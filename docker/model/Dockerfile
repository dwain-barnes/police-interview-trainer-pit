FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install Python 3.11 + build essentials for llama.cpp CUDA compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        curl gcc g++ cmake git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

# Build llama-server from source with CUDA
# Create libcuda.so.1 symlink — stubs dir only has libcuda.so, which
# causes the linker to fail inside Docker (no real GPU driver).
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    cmake --build build --config Release -j$(nproc) --target llama-server && \
    cp build/bin/llama-server /usr/local/bin/llama-server && \
    cp build/bin/lib*.so* /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/llama.cpp

WORKDIR /app

# Python deps for proxy + model download only
COPY docker/model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/model/entrypoint.sh docker/model/proxy.py docker/model/chat_template.jinja ./
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
