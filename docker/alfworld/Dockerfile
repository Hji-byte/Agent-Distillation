FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements/alfworld-eto.txt /tmp/alfworld-eto.txt
RUN python -m pip install --no-cache-dir -r /tmp/alfworld-eto.txt
RUN python -m pip install --no-cache-dir PyYAML==6.0.3

CMD ["python", "-m", "exps_research.alfworld_eto.integrity", "/workspace/_local/upstream/ETO"]
