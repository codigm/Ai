// =============================================================================
// NanoGPT — Chat Client (SSE Streaming)
// =============================================================================

(function () {
  'use strict';

  // ── DOM References ──────────────────────────────────────────────────────────
  const chatArea         = document.getElementById('chat-area');
  const welcomeScreen    = document.getElementById('welcome-screen');
  const userInput        = document.getElementById('user-input');
  const btnSend          = document.getElementById('btn-send');
  const btnClear         = document.getElementById('btn-clear');
  const btnSettings      = document.getElementById('btn-settings');
  const btnSettingsClose = document.getElementById('btn-settings-close');
  const btnClearSettings = document.getElementById('btn-clear-settings');
  const settingsPanel    = document.getElementById('settings-panel');
  const settingsOverlay  = document.getElementById('settings-overlay');
  const statusBar        = document.getElementById('status-bar');
  const statusMessage    = document.getElementById('status-message');

  // Sliders
  const sliderTemp       = document.getElementById('slider-temperature');
  const sliderMaxTokens  = document.getElementById('slider-max-tokens');
  const sliderTopK       = document.getElementById('slider-top-k');
  const valTemp          = document.getElementById('val-temperature');
  const valMaxTokens     = document.getElementById('val-max-tokens');
  const valTopK          = document.getElementById('val-top-k');

  // ── State ───────────────────────────────────────────────────────────────────
  let conversationHistory = []; // Array of {role: 'user'|'assistant', content: string}
  let isGenerating = false;
  let currentAbortController = null;

  // ── Settings ────────────────────────────────────────────────────────────────
  function getSettings() {
    return {
      temperature: parseFloat(sliderTemp.value),
      max_tokens: parseInt(sliderMaxTokens.value, 10),
      top_k: parseInt(sliderTopK.value, 10),
    };
  }

  // Bind slider display updates
  sliderTemp.addEventListener('input', () => {
    valTemp.textContent = parseFloat(sliderTemp.value).toFixed(2);
  });
  sliderMaxTokens.addEventListener('input', () => {
    valMaxTokens.textContent = sliderMaxTokens.value;
  });
  sliderTopK.addEventListener('input', () => {
    valTopK.textContent = sliderTopK.value;
  });

  // ── Settings Panel Toggle ───────────────────────────────────────────────────
  function openSettings() {
    settingsPanel.classList.add('open');
    settingsOverlay.classList.add('open');
    btnSettings.classList.add('active');
  }

  function closeSettings() {
    settingsPanel.classList.remove('open');
    settingsOverlay.classList.remove('open');
    btnSettings.classList.remove('active');
  }

  btnSettings.addEventListener('click', () => {
    settingsPanel.classList.contains('open') ? closeSettings() : openSettings();
  });
  btnSettingsClose.addEventListener('click', closeSettings);
  settingsOverlay.addEventListener('click', closeSettings);

  // ── Status Bar ──────────────────────────────────────────────────────────────
  function showError(msg) {
    statusMessage.textContent = msg;
    statusBar.classList.add('visible');
    setTimeout(() => statusBar.classList.remove('visible'), 6000);
  }

  // ── Auto-resize Textarea ───────────────────────────────────────────────────
  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
  });

  // ── Scroll to Bottom ───────────────────────────────────────────────────────
  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatArea.scrollTop = chatArea.scrollHeight;
    });
  }

  // ── Create Message Element ─────────────────────────────────────────────────
  function createMessageElement(role, content) {
    const msg = document.createElement('div');
    msg.className = `message message--${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message__avatar';
    avatar.textContent = role === 'user' ? '👤' : '⚡';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message__content';

    // Role label
    const label = document.createElement('div');
    label.className = 'message__label';
    label.textContent = role === 'user' ? 'You' : 'NanoGPT';

    const bubble = document.createElement('div');
    bubble.className = 'message__bubble';
    bubble.textContent = content || '';

    contentDiv.appendChild(label);
    contentDiv.appendChild(bubble);
    msg.appendChild(avatar);
    msg.appendChild(contentDiv);

    return { messageEl: msg, bubbleEl: bubble };
  }

  // ── Add User Message ───────────────────────────────────────────────────────
  function addUserMessage(text) {
    // Hide welcome screen
    if (welcomeScreen) {
      welcomeScreen.style.display = 'none';
    }

    conversationHistory.push({ role: 'user', content: text });
    const { messageEl } = createMessageElement('user', text);
    chatArea.appendChild(messageEl);
    scrollToBottom();
  }

  // ── Add Typing Indicator ───────────────────────────────────────────────────
  function addTypingIndicator() {
    const msg = document.createElement('div');
    msg.className = 'message message--assistant';
    msg.id = 'typing-indicator';

    const avatar = document.createElement('div');
    avatar.className = 'message__avatar';
    avatar.textContent = '⚡';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message__content';

    const label = document.createElement('div');
    label.className = 'message__label';
    label.textContent = 'NanoGPT';

    const bubble = document.createElement('div');
    bubble.className = 'message__bubble';

    const dots = document.createElement('div');
    dots.className = 'typing-indicator';
    dots.innerHTML = '<div class="typing-indicator__dot"></div><div class="typing-indicator__dot"></div><div class="typing-indicator__dot"></div>';

    bubble.appendChild(dots);
    contentDiv.appendChild(label);
    contentDiv.appendChild(bubble);
    msg.appendChild(avatar);
    msg.appendChild(contentDiv);
    chatArea.appendChild(msg);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  // ── Set Send Button State ─────────────────────────────────────────────────
  function setSendButtonState(generating) {
    if (generating) {
      btnSend.innerHTML = '■';
      btnSend.classList.add('is-stop');
      btnSend.title = 'Stop generating';
      btnSend.disabled = false;
    } else {
      btnSend.innerHTML = '➤';
      btnSend.classList.remove('is-stop');
      btnSend.title = 'Send message';
      btnSend.disabled = false;
    }
  }

  // ── Stream Response via SSE ────────────────────────────────────────────────
  async function streamResponse() {
    isGenerating = true;
    setSendButtonState(true);
    addTypingIndicator();

    const settings = getSettings();
    const body = {
      messages: conversationHistory,
      temperature: settings.temperature,
      max_tokens: settings.max_tokens,
      top_k: settings.top_k,
    };

    currentAbortController = new AbortController();

    try {
      const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: currentAbortController.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      // Remove typing indicator and create the assistant bubble
      removeTypingIndicator();
      const { messageEl, bubbleEl } = createMessageElement('assistant', '');
      chatArea.appendChild(messageEl);

      // Add streaming cursor
      const cursor = document.createElement('span');
      cursor.className = 'streaming-cursor';
      bubbleEl.appendChild(cursor);

      let fullContent = '';

      // Read the SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE lines
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);

            if (data === '[DONE]') {
              continue;
            }

            // Append token to the bubble
            fullContent += data;
            bubbleEl.textContent = fullContent;
            bubbleEl.appendChild(cursor); // Re-append cursor after textContent
            scrollToBottom();
          }
        }
      }

      // Clean up: remove cursor, finalize
      cursor.remove();
      conversationHistory.push({ role: 'assistant', content: fullContent });

    } catch (err) {
      removeTypingIndicator();
      if (err.name === 'AbortError') {
        // User cancelled — silently handled
      } else {
        showError(err.message || 'Failed to connect to the server.');
      }
    } finally {
      isGenerating = false;
      setSendButtonState(false);
      currentAbortController = null;
      userInput.focus();
    }
  }

  // ── Send Message / Stop ────────────────────────────────────────────────────
  function handleSendClick() {
    if (isGenerating) {
      // Stop generating
      if (currentAbortController) {
        currentAbortController.abort();
      }
      return;
    }
    sendMessage();
  }

  function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isGenerating) return;

    addUserMessage(text);
    userInput.value = '';
    userInput.style.height = 'auto';
    streamResponse();
  }

  // ── Clear Chat ─────────────────────────────────────────────────────────────
  function clearChat() {
    conversationHistory = [];

    // Remove all messages
    const messages = chatArea.querySelectorAll('.message');
    messages.forEach(m => m.remove());

    // Show welcome screen again
    if (welcomeScreen) {
      welcomeScreen.style.display = '';
    }

    // Abort if generating
    if (currentAbortController) {
      currentAbortController.abort();
    }

    closeSettings();
  }

  // ── Event Listeners ────────────────────────────────────────────────────────
  btnSend.addEventListener('click', handleSendClick);
  btnClear.addEventListener('click', clearChat);
  btnClearSettings.addEventListener('click', clearChat);

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Welcome chip buttons
  document.querySelectorAll('.welcome__chip').forEach(chip => {
    chip.addEventListener('click', () => {
      userInput.value = chip.dataset.prompt;
      sendMessage();
    });
  });

  // Close settings on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeSettings();
    }
  });

  // Focus input on load
  userInput.focus();
})();
