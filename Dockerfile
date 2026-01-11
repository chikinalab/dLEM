# DLEM image
FROM debian:stable-slim

ENV APT_PACKAGES=" \
		apt-transport-https \
		ca-certificates \
		locales \
		fonts-liberation \
		wget \
        build-essential \
		"

#Set up shell for install
USER root
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash","-c"]
ENV BASH_EN=~/.bashrc
ENV SHELL=/bin/bash
ENV TZ="America/New_York"

#Get apt packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends ${APT_PACKAGES} && \
    rm -rf "/var/lib/apt/lists/*" && \
    apt-get clean && \
    rm -rf /var/cache/apt && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Set working directory
ARG WORKDIR="/dlem"
WORKDIR ${WORKDIR}

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

#Get micromamba and use it to get UV
COPY environment.yml /tmp/environment.yml
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba --strip-components=1 && \
	chmod 755 ./micromamba && \
	mkdir -p $(dirname $MAMBA_ROOT_PREFIX) && \
	./micromamba shell init -s bash -r $MAMBA_ROOT_PREFIX && \
	echo "micromamba activate base" >> /root/.bashrc && \
	source ~/.bashrc && \
	./micromamba install -y -n base -f /tmp/environment.yml && \
	ln -s /micromamba /usr/bin/conda && \
	ln -s /micromamba /opt/conda/bin/conda && \
	./micromamba clean --all --yes && \
	rm ${MAMBA_ROOT_PREFIX}/lib/*.a && \
	rm -rf ${MAMBA_ROOT_PREFIX}/pkgs

# UV options
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install the project's dependencies using the lockfile and settings
COPY README.md /dlem/README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Rest of source and install CLI
COPY . /dlem
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev && \
    uv tool install /dlem
ENV PATH="/dlem/.venv/bin:$PATH"

ENTRYPOINT []

USER nonroot

CMD ["dlem", "-h"]