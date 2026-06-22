# ============================================
# California Housing Pipeline Makefile
# ============================================

.PHONY: help install test test-fast lint format clean pipeline

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-fast    - Run unit tests only (skip integration)"
	@echo "  make lint         - Run linter (ruff)"
	@echo "  make format       - Format code (black + ruff)"
	@echo "  make clean        - Remove cache and build artifacts"
	@echo "  make pipeline     - Run full data pipeline"

install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not integration"

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

pipeline:
	python pipelines/run_pipeline.py