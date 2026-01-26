ARG NGC_VERSION="25.05-py3"

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
	gettext

COPY . /workspace/MultiMeditron
WORKDIR /workspace/MultiMeditron
RUN pip install -e ".[flash-attn]"
RUN pip install git+https://github.com/EPFLiGHT/lmms-eval.git 

COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# A small python script that rebuild the envsubst utility to work with {{VAR}} syntax
COPY docker/envsubst.py /bin/envsubst
RUN chmod +x /bin/envsubst


ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
CMD ["/bin/bash"]
