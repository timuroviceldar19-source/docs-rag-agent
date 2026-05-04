.PHONY: install qdrant-up qdrant-down lint format typecheck test check help ui

help: ## Print each target with a one-line description
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## python -m pip install -e ".[dev]"
	python -m pip install -e ".[dev]"

qdrant-up: ## docker compose up -d qdrant
	docker compose up -d qdrant

qdrant-down: ## docker compose down
	docker compose down

lint: ## ruff check .
	ruff check .

format: ## ruff format .
	ruff format .

typecheck: ## mypy
	mypy

test: ## pytest
	pytest

check: lint typecheck test ## runs lint, typecheck, test in order

ui: ## streamlit run streamlit_app/app.py
	streamlit run streamlit_app/app.py
