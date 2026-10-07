# syntax=docker/dockerfile:1
# Multi-stage build. Targets:
#   serve      - FastAPI scoring service (slim: no sklearn/optuna/shap/prefect)
#   pipelines  - training, monitoring and Prefect flows
#   mlflow     - MLflow tracking server (same version as the client libraries)
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

# ---------- builders: dependencies into an isolated venv ----------
FROM base AS build-serve
RUN python -m venv /opt/venv
COPY requirements/serve.txt /tmp/requirements.txt
# optional build secret "extra_ca": corporate / proxy CA bundle (no-op when absent)
RUN --mount=type=secret,id=extra_ca \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt \
    && find /opt/venv -name "__pycache__" -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.11/site-packages/pyarrow/include \
              /opt/venv/lib/python3.11/site-packages/*/tests

FROM base AS build-pipelines
RUN python -m venv /opt/venv
COPY requirements/pipelines.txt /tmp/requirements.txt
RUN --mount=type=secret,id=extra_ca \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt \
    && find /opt/venv -name "__pycache__" -prune -exec rm -rf {} +

FROM base AS build-mlflow
RUN --mount=type=secret,id=extra_ca \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi \
    && python -m venv /opt/venv && /opt/venv/bin/pip install "mlflow==3.16.1"

# ---------- runtime images ----------
FROM base AS serve
COPY --from=build-serve /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app src/ src/
COPY --chown=app:app configs/ configs/
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src FRAUD_ROOT=/app MLFLOW_DISABLE_AGENT_HINT=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "fraud.serve.app:build", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM base AS pipelines
COPY --from=build-pipelines /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app src/ src/
COPY --chown=app:app configs/ configs/
COPY --chown=app:app scripts/ scripts/
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src FRAUD_ROOT=/app MLFLOW_DISABLE_AGENT_HINT=1 \
    MPLCONFIGDIR=/tmp/matplotlib
USER app
CMD ["python", "-m", "fraud.pipelines.serve_flows"]

FROM base AS mlflow
COPY --from=build-mlflow /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH MLFLOW_DISABLE_AGENT_HINT=1
USER app
EXPOSE 5000
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", \
     "--host", "0.0.0.0", "--port", "5000", "--workers", "2", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"]
