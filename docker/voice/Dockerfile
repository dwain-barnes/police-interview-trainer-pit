FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install Python 3.11 + essentials (git needed for torch.hub, ffmpeg for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev curl ffmpeg git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/voice/requirements.txt .
RUN pip install --no-cache-dir "setuptools<72" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall resemble-perth

COPY docker/voice/serve_voice.py .
COPY docker/voice/entrypoint.sh .
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

RUN mkdir -p /app/static
COPY index.html /app/static/index.html
COPY scenarios.json /app/static/scenarios.json

EXPOSE 8001

ENTRYPOINT ["./entrypoint.sh"]
