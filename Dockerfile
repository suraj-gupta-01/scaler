# ── Adaptive Alert Triage — Optimised Dockerfile ─────────────────────────────
#
# Key change vs previous version:
#   - Installs ONLY core deps from pyproject.toml (no torch / stable-baselines3)
#   - torch + stable-baselines3 are in the [train] optional group — NOT here
#   - Build time: ~60-90s  (was ~2200s pulling 2GB of PyTorch)
#
# Build:
#   docker build -t adaptive-alert-triage:latest .
#
# Run server:
#   docker run -p 7860:7860 adaptive-alert-triage:latest
#
# Run inference baseline:
#   docker run --rm \
#     -e API_BASE_URL="https://api.openai.com/v1" \
#     -e MODEL_NAME="gpt-4o-mini" \
#     -e HF_TOKEN="hf_..." \
#     adaptive-alert-triage:latest \
#     python inference.py --n 3
#
# Environment variables (set at runtime, not here):
#   API_BASE_URL  — LLM endpoint  (default: https://api.openai.com/v1)
#   MODEL_NAME    — LLM model name (default: gpt-4o-mini)
#   HF_TOKEN      — Hugging Face / OpenAI API key
#   PORT          — Server port   (default: 7860)

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy dependency manifest first (maximises Docker layer cache) ─────────────
COPY pyproject.toml .

# ── Upgrade pip/setuptools/wheel ──────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Copy full project ─────────────────────────────────────────────────────────
COPY . .

# ── Install ONLY core deps (no [train], no [viz], no [gemini]) ────────────────
# This skips torch, stable-baselines3, matplotlib, seaborn, pandas, google-genai
# Those are only needed for local training/analysis — not the server or inference
RUN pip install --no-cache-dir -e .

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app
ENV PORT=7860

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Add non-root user ───────────────────────────────────────────────
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/weights /app/results \
    && chown -R appuser:appuser /app
USER appuser
# ── Default command: start the FastAPI server ─────────────────────────────────
CMD ["sh", "-c", \
    "python -m uvicorn adaptive_alert_triage.server:app \
    --host 0.0.0.0 \
    --port ${PORT:-7860} \
    --workers 1 \
    --log-level info"]
