# Multi-arch (linux/amd64, linux/arm64) image with the genomics tools and the gta CLI.
FROM mambaorg/micromamba:1.5.10-jammy

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

# MAMBA_DOCKERFILE_ACTIVATE makes RUN commands see the conda env (python, pip, java).
ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md LICENSE /tmp/build/
COPY --chown=$MAMBA_USER:$MAMBA_USER src /tmp/build/src
RUN pip install --no-cache-dir /tmp/build && rm -rf /tmp/build

USER root
RUN mkdir /work && chown $MAMBA_USER:$MAMBA_USER /work
USER $MAMBA_USER
WORKDIR /work
ENV GTA_ROOT=/work

# _entrypoint.sh activates the conda env before running gta.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "gta"]
CMD ["--help"]
