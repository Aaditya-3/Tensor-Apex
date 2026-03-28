FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user . $HOME/app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

EXPOSE 7860 7861
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"

CMD ["python", "gradio_app.py"]
