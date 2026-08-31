# ============================================
# California Housing Pipeline Makefile
# ============================================

.PHONY: help install install-dev test test-fast lint format clean pipeline pull push status

help:
	@echo "============================================================"
	@echo "  California Housing MLOps Pipeline"
	@echo "============================================================"
	@echo "Setup:"
	@echo "  make install       - Install core dependencies (from pyproject.toml)"
	@echo "  make install-dev   - Install core + dev tools (ruff, pytest, mypy)"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test          - Run all tests"
	@echo "  make test-fast     - Run unit tests only (skip integration)"
	@echo "  make lint          - Run linter (ruff check)"
	@echo "  make format        - Format code (ruff format + fix)"
	@echo ""
	@echo "Pipeline (DVC):"
	@echo "  make pipeline      - Run full data pipeline (dvc repro)"
	@echo "  make pull          - Pull data/models from DVC remote"
	@echo "  make push          - Push data/models to DVC remote"
	@echo "  make status        - Show DVC pipeline status"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Remove cache and build artifacts"
	@echo "============================================================"

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install --upgrade pip setuptools wheel
	pip install -e .

install-dev:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"

# ── Testing & Quality ────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not integration"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# ── Pipeline (DVC) ───────────────────────────────────────────────────────────

pipeline:
	dvc repro

pull:
	dvc pull

push:
	dvc push

status:
	dvc status

# ── Maintenance ──────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info .coverage coverage.xml