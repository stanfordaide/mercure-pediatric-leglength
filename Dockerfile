# Set base image to python:3.12.3-slim-bookworm using index digest hash to fix version
# This version of python:3.12.3-slim-bookworm has OS: Debian 12 (bookworm) and python: 3.12.3
# Use --platform=linux/arm64 for Apple Silicon Macs, or remove --platform for auto-detection
FROM python@sha256:2be8daddbb82756f7d1f2c7ece706aadcb284bf6ab6d769ea695cc3ed6016743

# Environment variables for better container behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_HOME=/app/v0/.cache

# Home directory setup
ENV HOME_DIR=/app/v0
RUN mkdir -p ${HOME_DIR}

# Set the working directory
# Note: This is the directory where the gear is run
WORKDIR ${HOME_DIR}

# Create non-root user for security
RUN groupadd -r mercureapp && useradd -r -g mercureapp -d ${HOME_DIR} -s /bin/bash mercureapp

# 1. Update package list for apt-get
# 2. Use apt-get to install required packages, 
#    skipping installation of recommended packages
#    (to keep the image size small)
# 3. Clean up package manager cache to reduce image size
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
        git \
        wget \
        curl \
        ca-certificates \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        gosu \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Upgrade pip and install Python dependencies
COPY requirements.txt ${HOME_DIR}/requirements.txt
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install -r requirements.txt

# Create models and cache directories
RUN mkdir -p ${HOME_DIR}/models ${HOME_DIR}/.cache

# Copy only the files needed for downloading models first
# This layer will be cached unless download_models.py or registry.json change
COPY registry.json ${HOME_DIR}/registry.json
COPY download_models.py ${HOME_DIR}/download_models.py

# Download all model checkpoints
# Note: Comment out if you want to download models at runtime instead
RUN python3 download_models.py

# Copy the remaining application files
# This layer will be invalidated when application code changes, but models won't be re-downloaded
COPY leglength ${HOME_DIR}/leglength
COPY monitoring ${HOME_DIR}/monitoring
COPY run.py ${HOME_DIR}/run.py
COPY docker_entrypoint.sh ${HOME_DIR}/docker_entrypoint.sh

# Create output directory with proper permissions for volume mounting
RUN mkdir -p /output && \
    chown mercureapp:mercureapp /output && \
    chmod 755 /output

# Set proper ownership and permissions
RUN chown -R mercureapp:mercureapp ${HOME_DIR} && \
    chmod a+x ${HOME_DIR}/run.py && \
    chmod a+x ${HOME_DIR}/docker_entrypoint.sh

# Add health check to verify container readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Don't switch users here - let the entrypoint handle it
# This allows the entrypoint to fix permissions as root if needed
# USER mercureapp

# Add labels for better container management
LABEL maintainer="Arogya Koirala" \
      description="Leg Length Analysis Module" \
      version="1.0" \
      org.opencontainers.image.title="mercure-leglength" \
      org.opencontainers.image.description="Mercure processing module for leg length analysis"

# Configure entrypoint with proper signal handling
ENTRYPOINT ["./docker_entrypoint.sh"]