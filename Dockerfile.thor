# Jetson Thor (JetPack 7, arm64) image for training/eval. PyTorch only: the base
# image's TensorRT 11 does not support Jetson, so engine export runs natively.
# Build: docker build -f Dockerfile.thor -t yoloworld-thor .
FROM ultralytics/ultralytics:latest-nvidia-arm64

RUN uv pip install --system --no-cache-dir --break-system-packages \
        git+https://github.com/ultralytics/CLIP.git
