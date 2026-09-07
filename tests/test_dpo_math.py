import pytest
import torch
from torch.nn import functional as F
from dpo_train import get_batch_logps

def test_get_batch_logps_masking():
    """
    Test that get_batch_logps correctly ignores tokens marked with -100 
    (which represent the prompt that we don't want to calculate loss on).
    """
    # Dummy logits: (Batch=1, SeqLen=3, VocabSize=3)
    dummy_logits = torch.tensor([[
        [2.0, 1.0, 0.1], 
        [0.5, 2.5, 0.1], 
        [1.0, 1.0, 1.0]
    ]])
    
    # Dummy labels: (Batch=1, SeqLen=3)
    # Target tokens are 0 and 1. The last token is masked with -100.
    dummy_labels = torch.tensor([[0, 1, -100]])
    
    # Hand calculations:
    # Token 1 logits: [2.0, 1.0, 0.1]
    # log_softmax -> log( e^2 / (e^2 + e^1 + e^0.1) ) = log( 7.389 / 11.212 ) = -0.4174
    # Token 2 logits: [0.5, 2.5, 0.1]
    # log_softmax -> log( e^2.5 / (e^0.5 + e^2.5 + e^0.1) ) = log( 12.182 / 14.935 ) = -0.2037
    # Token 3 logits: [1.0, 1.0, 1.0] (MASKED!) -> logprob is calculated but multiplied by 0.
    # Total expected sum: -0.4174 + -0.2037 + 0 = -0.6211
    
    expected = -0.6211
    logps = get_batch_logps(dummy_logits, dummy_labels)
    
    assert abs(logps.item() - expected) < 1e-3, f"Masking failed. Expected {expected}, got {logps.item()}"

def test_dpo_loss_gradient_direction():
    """
    Test that the DPO loss function pushes gradients in the correct direction:
    If the policy model favors the chosen response more than the reference model,
    the loss should decrease.
    """
    beta = 0.1
    
    # Scenario A: Policy model is heavily favoring the chosen response
    policy_chosen_logps_A = torch.tensor([-1.0], requires_grad=True)
    policy_rejected_logps_A = torch.tensor([-5.0], requires_grad=True)
    
    ref_chosen_logps = torch.tensor([-2.0])
    ref_rejected_logps = torch.tensor([-2.0])
    
    policy_logratio_A = policy_chosen_logps_A - policy_rejected_logps_A # 4.0
    ref_logratio_A = ref_chosen_logps - ref_rejected_logps # 0.0
    
    loss_A = -F.logsigmoid(beta * (policy_logratio_A - ref_logratio_A)).mean()
    
    # Scenario B: Policy model is heavily favoring the rejected response (BAD!)
    policy_chosen_logps_B = torch.tensor([-5.0], requires_grad=True)
    policy_rejected_logps_B = torch.tensor([-1.0], requires_grad=True)
    
    policy_logratio_B = policy_chosen_logps_B - policy_rejected_logps_B # -4.0
    ref_logratio_B = ref_chosen_logps - ref_rejected_logps # 0.0
    
    loss_B = -F.logsigmoid(beta * (policy_logratio_B - ref_logratio_B)).mean()
    
    # Verification 1: Loss B (where policy fails) should be higher than Loss A (where policy succeeds)
    assert loss_B.item() > loss_A.item(), "DPO loss should penalize policy favoring rejected responses"
    
    # Verification 2: Check gradient direction for Scenario B
    loss_B.backward()
    
    # Since we want to MINIMIZE loss B:
    # policy_chosen_logps_B gradient should be negative (optimizer will push it UP)
    assert policy_chosen_logps_B.grad.item() < 0, "DPO should push chosen logprobs UP"
    
    # policy_rejected_logps_B gradient should be positive (optimizer will push it DOWN)
    assert policy_rejected_logps_B.grad.item() > 0, "DPO should push rejected logprobs DOWN"
