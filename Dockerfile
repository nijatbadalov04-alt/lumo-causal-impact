# UK rail open-access — reproducibility image (CPU-only, Linux/amd64).
#
# Builds a self-contained CPU environment that runs the test suite and any
# pipeline stage whose input data is present. The GPU deep-counterfactual stage
# (src.models.deep.deep_counterfactual) is import-safe here: it imports a CPU
# torch and its main() aborts cleanly when no CUDA device is visible.
#
# Build:  docker build -t uk-rail-oa .
# Test:   docker run --rm uk-rail-oa                 # default CMD: pytest -q
# Shell:  docker run --rm -it uk-rail-oa bash
# Pipeline (with data mounted):
#         docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#                 uk-rail-oa python run_pipeline.py
#
# NOTE: raw data (~1.5 GB, user-provided) is NOT baked into the image — it is
# excluded via .dockerignore. Mount data/ at run time, or rely on the auto-download
# stages (ORR/CAA) for the bits that fetch themselves. See data/README.md and docs/.

FROM python:3.12-slim

# Fail fast and keep Python output unbuffered / UTF-8 everywhere.
ENV PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libs that a few wheels (matplotlib/pyarrow/scipy stacks) expect at runtime.
# Kept minimal; most scientific wheels are manylinux and need no build toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (better layer caching). requirements-ci.txt = pinned CPU stack
# + CPU torch; the CPU torch wheel is resolved from the PyTorch CPU index.
COPY requirements.txt requirements-ci.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements-ci.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the repo (data/, .venv/, results/, .git excluded via .dockerignore).
COPY . .

# Default: run the test suite. Data-dependent pipeline stages skip when data/ is
# absent (guard clauses), so the suite is the right CI/Docker smoke target.
CMD ["pytest", "-q"]
