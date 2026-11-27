FROM python:3.14-slim

# Non-root user required/recommended for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user server.py model.py tokenizer.py generate.py tokenizer.json ./
COPY --chown=user static/ ./static/
COPY --chown=user model_sft_final.pt* model_step_9500.pt* ./

EXPOSE 7860

CMD ["python", "server.py"]
