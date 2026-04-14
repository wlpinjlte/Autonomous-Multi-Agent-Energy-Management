FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/NatLabRockies/EnergyPlus/releases/download/v25.1.0/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64.sh && \
    chmod +x EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64.sh && \
    echo "y\r" |./EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64.sh && \
    rm EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64.sh

ENV ENERGYPLUS_VERSION=25.1.0
ENV ENERGYPLUS_DIR=/usr/local/EnergyPlus-25-1-0
ENV PYTHONPATH="${ENERGYPLUS_DIR}"
ENV EPLUS_PATH="${ENERGYPLUS_DIR}"

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app

COPY . /app

RUN uv sync --frozen

# ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "env_setup.py"]