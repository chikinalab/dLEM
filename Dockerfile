# DLEM image
FROM debian:stable-slim

ENV APT_PACKAGES=" \
		ca-certificates \
		zlib1g \
		zlib1g-dev \
		build-essential \
		curl \
		"

#Set up shell for install
USER root
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash","-c"]
ENV BASH_EN=/${USER}/.bashrc
ENV SHELL=/bin/bash
ENV TZ="America/New_York"

# Set working directory
ARG WORKDIR="/dlem"
WORKDIR ${WORKDIR}

#Get apt packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends ${APT_PACKAGES} && \
    rm -rf "/var/lib/apt/lists/*" && \
    apt-get clean && \
    rm -rf /var/cache/apt && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# UV options
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed
ENV PYTHON_VERSION=3.12

# Install the project's dependencies using the lockfile and settings
COPY README.md /dlem/README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv python install ${PYTHON_VERSION} && \
    uv sync --locked --no-install-project --no-dev 

COPY . /dlem
RUN --mount=type=cache,target=/root/.cache/uv \ 
    uv sync --locked
ENV PATH="/dlem/.venv/bin:$PATH"

ENTRYPOINT []
CMD []