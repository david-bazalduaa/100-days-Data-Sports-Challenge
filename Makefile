# ============================================================================
# 100 Days of Data & Sports Challenge — Makefile
# ============================================================================
# Usage:
#   make setup      → Create virtual environment and install dependencies
#   make data       → Download and process raw datasets
#   make train      → Train all models
#   make test       → Run test suite
#   make lint       → Lint and format code
#   make clean      → Remove artifacts and caches
#   make profile    → Generate data profiling reports
#   make dashboard  → Launch Streamlit scouting dashboard
#   make api        → Launch FastAPI NFL analytics API
#   make web        → Serve GitHub Pages site locally
# ============================================================================

.PHONY: setup data train test lint clean profile dashboard api web help

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# ── Colors ──
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
RESET  := \033[0m

help: ## Show this help message
	@echo "$(CYAN)100 Days of Data & Sports Challenge$(RESET)"
	@echo "$(YELLOW)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}'

# ── Environment ──
setup: ## Create virtual environment and install all dependencies
	@echo "$(CYAN)→ Creating virtual environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[all]"
	$(VENV)/bin/pre-commit install || true
	$(VENV)/bin/nbstripout --install || true
	@echo "$(GREEN)✓ Environment ready. Activate with: source $(VENV)/bin/activate$(RESET)"

setup-minimal: ## Install core dependencies only (no dev/api/dashboard)
	@echo "$(CYAN)→ Creating minimal environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	@echo "$(GREEN)✓ Minimal environment ready.$(RESET)"

# ── Data ──
data: ## Download and process all datasets
	@echo "$(CYAN)→ Downloading datasets...$(RESET)"
	$(PY) src/data/download_statsbomb.py
	$(PY) src/data/download_nfl.py
	$(PY) src/data/download_fbref.py
	@echo "$(GREEN)✓ All datasets downloaded.$(RESET)"

data-football: ## Download football (soccer) datasets only
	$(PY) src/data/download_statsbomb.py

data-nfl: ## Download NFL datasets only
	$(PY) src/data/download_nfl.py

# ── Training ──
train: ## Train all models
	@echo "$(CYAN)→ Training models...$(RESET)"
	$(PY) src/models/train_xg.py
	$(PY) src/models/train_wp.py
	$(PY) src/models/train_match_predictor.py
	@echo "$(GREEN)✓ All models trained.$(RESET)"

# ── Testing ──
test: ## Run test suite with coverage
	@echo "$(CYAN)→ Running tests...$(RESET)"
	$(VENV)/bin/pytest tests/ --cov=src --cov-report=term-missing -v

test-quick: ## Run tests without coverage
	$(VENV)/bin/pytest tests/ -v --tb=short

# ── Code Quality ──
lint: ## Lint and format code
	@echo "$(CYAN)→ Linting...$(RESET)"
	$(VENV)/bin/ruff check src/ tests/ --fix
	$(VENV)/bin/ruff format src/ tests/
	@echo "$(GREEN)✓ Code quality checks passed.$(RESET)"

typecheck: ## Run type checking
	$(VENV)/bin/mypy src/ --ignore-missing-imports

# ── Profiling ──
profile: ## Generate data profiling reports
	@echo "$(CYAN)→ Generating profiling reports...$(RESET)"
	$(PY) -c "from ydata_profiling import ProfileReport; import pandas as pd; \
		print('Profiling module available.')"
	@echo "$(GREEN)✓ Run individual profiling notebooks for detailed reports.$(RESET)"

# ── Applications ──
dashboard: ## Launch Streamlit scouting dashboard
	@echo "$(CYAN)→ Launching Scouting Dashboard...$(RESET)"
	$(VENV)/bin/streamlit run apps/scouting_dashboard.py

api: ## Launch FastAPI NFL analytics API
	@echo "$(CYAN)→ Launching NFL Analytics API...$(RESET)"
	$(VENV)/bin/uvicorn api.app.main:app --reload --port 8000

# ── Web ──
web: ## Serve GitHub Pages site locally with Jekyll
	@echo "$(CYAN)→ Serving site locally...$(RESET)"
	cd docs && bundle exec jekyll serve --livereload

# ── Cleanup ──
clean: ## Remove generated artifacts and caches
	@echo "$(YELLOW)→ Cleaning...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf mlruns wandb
	@echo "$(GREEN)✓ Clean.$(RESET)"

clean-all: clean ## Remove everything including venv and data
	rm -rf $(VENV)
	rm -rf data/raw/**/*.csv data/raw/**/*.parquet
	rm -rf data/processed/**/*.csv data/processed/**/*.parquet
	@echo "$(GREEN)✓ Full clean complete.$(RESET)"
