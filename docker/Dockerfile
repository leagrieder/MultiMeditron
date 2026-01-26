ARG NGC_VERSION="25.05-py3"
ARG ARCH="x86_64"

FROM nvcr.io/nvidia/pytorch:$NGC_VERSION

# To build documentation + cool utilies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        graphviz \
        ffmpeg \
        tmux \
        sudo \
        git \
        gosu \

COPY . /workspace/MultiMeditron
WORKDIR /workspace/MultiMeditron
RUN pip install -e ".[flash-attn]"
RUN pip install git+https://github.com/EPFLiGHT/lmms-eval.git 


COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

RUN ARCH="$(uname -m)" && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        curl -L https://github.com/a8m/envsubst/releases/download/v1.4.3/envsubst-Linux-arm64 -o envsubst \
    else \
        curl -L https://github.com/a8m/envsubst/releases/download/v1.4.3/envsubst-Linux-x86_64 -o envsubst \
    fi && \
    chmod +x envsubst && \
    sudo mv envsubst /usr/local/bin

ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
CMD ["/bin/bash"]
