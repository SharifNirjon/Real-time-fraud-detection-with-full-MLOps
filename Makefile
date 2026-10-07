# Real-time fraud detection - common tasks. Python commands run in .venv on the host;
# `make up` starts the dockerised stack (api, redis, mlflow, prometheus, grafana, prefect).
SHELL := /bin/bash
PY ?= .venv/bin/python
SAMPLE ?=            # make data SAMPLE=--sample  for ~20% of cards
TRIALS ?=
API_URL ?= http://localhost:8000
export MLFLOW_TRACKING_URI ?= http://localhost:5000
export ADMIN_API_KEY ?= change-me-local-dev-key
export MLFLOW_DISABLE_AGENT_HINT = 1
COMPOSE ?= docker compose

.PHONY: help venv data features train evaluate serve warm simulate simulate-drift monitor retrain \
        test lint format bench up down logs all demo clean

help:
	@grep -E '^[a-z-]+:.*' Makefile | cut -d: -f1 | sort | tr '\n' ' '; echo

venv:
	uv venv -p 3.11 .venv && uv pip install --python .venv/bin/python -r requirements/dev.txt -e .

data:                       ## download (Kaggle API) + ingest + time split
	@test -f data/raw/train_transaction.csv || (mkdir -p data/raw && kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw && kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/raw && cd data/raw && for f in *.zip; do unzip -o $$f && rm $$f; done)
	$(PY) -m fraud.data.ingest $(SAMPLE)
	$(PY) -m fraud.data.split

features:                   ## V-column selection on train + offline features
	$(PY) -m fraud.features.select
	$(PY) -m fraud.features.build

train: features             ## baselines + LightGBM + calibration + thresholds + MLflow registry
	$(PY) -m fraud.train.train $(SAMPLE) $(if $(TRIALS),--trials $(TRIALS),)

evaluate:                   ## re-generate reports/ from saved scores
	$(PY) -m fraud.evaluate.report

serve:                      ## run the API on the host (needs redis + mlflow)
	REDIS_URL=redis://localhost:6379/0 $(PY) -m uvicorn fraud.serve.app:build --factory --port 8000

warm:                       ## load all pre-live history into the Redis online store
	$(PY) -m fraud.serve.warm

simulate: warm              ## replay the live stream (normal traffic)
	$(PY) scripts/simulate.py --api-url $(API_URL) $(SIMARGS)

simulate-drift: warm        ## replay the live stream with injected amount/device drift
	$(PY) scripts/simulate.py --api-url $(API_URL) --drift $(SIMARGS)

monitor:                    ## Evidently drift + live performance report (retrains if drift)
	API_URL=$(API_URL) $(PY) -m fraud.pipelines.flows monitor $(MONITORARGS)

retrain:                    ## champion/challenger retraining flow
	API_URL=$(API_URL) $(PY) -m fraud.pipelines.flows retrain --trigger manual

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests scripts && .venv/bin/black --check src tests scripts

format:
	.venv/bin/ruff check --fix src tests scripts && .venv/bin/black src tests scripts

bench:                      ## async load test: 2,000 requests
	$(PY) scripts/benchmark_latency.py --api-url $(API_URL) --requests 2000

up:                         ## build + start the full stack
	mkdir -p data/predictions data/labels reports/drift artifacts mlflow_data
	chmod -R a+rwX data/predictions data/labels reports artifacts mlflow_data
	$(COMPOSE) up -d --build
	@echo "API http://localhost:8000/docs  MLflow http://localhost:5000  Grafana http://localhost:3000  Prometheus http://localhost:9090  Prefect http://localhost:4200"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

all: data train test       ## full offline pipeline + tests

demo:                       ## drift demo: drifted replay, then monitor -> retrain -> promote
	$(MAKE) simulate-drift SIMARGS="--batch-size 25"
	$(MAKE) monitor MONITORARGS=--auto-retrain

clean:
	rm -rf data/predictions/* data/labels/* reports/drift/*
