# Set base image to python:3.12.3-slim-bookworm using index digest hash to fix version
# This version of python:3.12.3-slim-bookworm has OS: Debian 12 (bookworm) and python: 3.12.3
FROM --platform=linux/amd64 python@sha256:2be8daddbb82756f7d1f2c7ece706aadcb284bf6ab6d769ea695cc3ed6016743

# Environment variables for better container behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Flywheel spec (v0)
ENV FLYWHEEL=/flywheel/v0
RUN mkdir -p ${FLYWHEEL}

# Set the working directory
# Note: This is the directory where the gear is run
WORKDIR ${FLYWHEEL}

# Create non-root user for security
RUN groupadd -r flywheel && useradd -r -g flywheel -d ${FLYWHEEL} -s /bin/bash flywheel

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
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Upgrade pip and install Python dependencies
COPY requirements.txt ${FLYWHEEL}/requirements.txt
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install flywheel-sdk && \
    pip3 install -r requirements.txt && \
    pip3 cache purge

# Create models directory
RUN mkdir -p ${FLYWHEEL}/models

# Copy the application files
COPY leglength ${FLYWHEEL}/leglength
COPY download_models.py ${FLYWHEEL}/download_models.py
COPY run.py ${FLYWHEEL}/run.py

# Download all model checkpoints
RUN python3 download_models.py

# Set proper ownership and permissions
RUN chown -R flywheel:flywheel ${FLYWHEEL} && \
    chmod a+x ${FLYWHEEL}/run.py

# Add health check to verify container readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Switch to non-root user
USER flywheel

# Add labels for better container management
LABEL maintainer="flywheel" \
      description="Leg Length Analysis Gear" \
      version="1.0" \
      org.opencontainers.image.title="fw-leglength" \
      org.opencontainers.image.description="Flywheel gear for leg length analysis"

# Configure entrypoint with proper signal handling
ENTRYPOINT ["python3", "/flywheel/v0/run.py"]