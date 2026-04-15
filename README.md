# 🏟️ 100 Days of Data & Sports Challenge

> **ML Engineer | Data Scientist** — Sports Analytics Specialization
>
> Building professional-grade data products for football ⚽ and American football 🏈.

[![Progress](https://img.shields.io/badge/Progress-Day%201%2F100-brightgreen?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)]()

---

## 🎯 What Is This?

A **100-day intensive challenge** to build a professional sports analytics portfolio from scratch. Six end-to-end ML projects covering predictive modeling, player scouting, win probability, match forecasting, interactive visualization, and draft analytics — all deployed and publicly accessible.

**This is not a learning diary.** Every day produces a tangible deliverable: a pipeline, a model, a visualization, a dashboard, or a documented insight. The goal is to build **production-grade evidence** of ML engineering skills applied to real sports problems.

---

## 📊 Projects

| # | Project | Sport | Type | Status |
|---|---------|-------|------|--------|
| 1 | **FootballScout** — Player Similarity Engine | ⚽ Football | Scouting / Unsupervised ML | 🔲 |
| 2 | **xG-Lab** — Expected Goals from Scratch | ⚽ Football | Predictive Model | 🔲 |
| 3 | **GridironML** — NFL Win Probability API | 🏈 NFL | ML API / Deployment | 🔲 |
| 4 | **MatchOracle** — Match Outcome Forecaster | ⚽ Football | Forecasting System | 🔲 |
| 5 | **GamePulse** — Interactive Game Flow Analyzer | ⚽🏈 Multi | Visualization Product | 🔲 |
| 6 | **DraftEdge** — NFL Draft Success Predictor | 🏈 NFL | Prediction / Talent Eval | 🔲 |

---

## 🗺️ Roadmap

| Phase | Days | Focus | Status |
|-------|------|-------|--------|
| **1 — Foundations** | 1–15 | Infrastructure, data pipelines, viz library | 🟡 In Progress |
| **2 — EDA & Features** | 16–35 | Exploration, feature engineering, storytelling | ⚪ Pending |
| **3 — Modeling** | 36–60 | Predictive models, validation, experiment tracking | ⚪ Pending |
| **4 — Products** | 61–85 | Dashboards, APIs, automation, MLOps | ⚪ Pending |
| **5 — Polish** | 86–100 | Portfolio, branding, interview prep, launch | ⚪ Pending |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Core** | Python 3.11 · pandas · polars · NumPy |
| **ML** | scikit-learn · XGBoost · LightGBM · Optuna · SHAP |
| **Sports Data** | StatsBomb · nflverse · FBref · Metrica Sports |
| **Visualization** | matplotlib · mplsoccer · Plotly · seaborn |
| **Deployment** | FastAPI · Streamlit · Docker · Railway |
| **MLOps** | MLflow · GitHub Actions · DVC |
| **Quality** | ruff · black · pytest · pre-commit · nbstripout |

---

## 📁 Repository Structure

```
100-days-Data-Sports-Challenge/
├── data/
│   ├── raw/                # Original datasets (not tracked by git)
│   │   ├── statsbomb/      # StatsBomb open data
│   │   ├── nfl/            # nflverse play-by-play & rosters
│   │   ├── fbref/          # FBref scraped stats
│   │   └── tracking/       # Tracking data (Metrica Sports)
│   ├── processed/          # Clean, feature-engineered datasets
│   │   ├── football/
│   │   └── nfl/
│   └── external/           # Third-party reference data
├── notebooks/
│   ├── phase1_foundations/  # Data exploration & quality
│   ├── phase2_eda_features/ # EDA, feature engineering
│   ├── phase3_modeling/    # Model training & evaluation
│   ├── phase4_products/    # Product development
│   └── phase5_polish/      # Final polish & demos
├── src/                    # Source code package
│   ├── data/               # Data loaders & downloaders
│   ├── features/           # Feature engineering pipelines
│   ├── models/             # Model training & inference
│   ├── viz/                # Visualization utilities
│   ├── validation/         # Temporal CV frameworks
│   ├── dashboard/          # Dashboard backends
│   └── scrapers/           # Web scraping utilities
├── models/                 # Serialized trained models
│   ├── football/
│   └── nfl/
├── apps/                   # Streamlit dashboards
├── api/                    # FastAPI services
│   └── app/
├── reports/                # Generated reports
│   ├── figures/
│   └── html/
├── tests/                  # Unit & integration tests
├── docs/                   # GitHub Pages site source
├── .github/workflows/      # CI/CD pipelines
├── pyproject.toml          # Dependencies & tool config
├── Makefile                # Automation targets
├── environment.yml         # Conda environment (alternative)
└── README.md               # ← You are here
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/david-bazaldua/100-days-Data-Sports-Challenge.git
cd 100-days-Data-Sports-Challenge

# Option A: pip (recommended)
make setup

# Option B: conda
conda env create -f environment.yml
conda activate sports-analytics
pip install -e .

# Download datasets
make data

# Run tests
make test

# Launch dashboard (after Phase 4)
make dashboard

# Launch API (after Phase 4)
make api
```

---

## 📅 Daily Log

| Week | Days | Milestone | Highlight |
|------|------|-----------|-----------|
| 1 | 1–7 | 🏗️ Infrastructure | Repo + Web + Data Pipelines |
| 2 | 8–14 | 📊 Data Foundation | Quality reports + schemas + viz |
| 3 | 15–21 | 🔍 EDA Kickoff | Shot analysis + passing networks |
| ... | ... | ... | ... |

*Full daily tracker available on the [portfolio website]()*

---

## 📖 Key Decisions

- **Temporal validation only** — No random splits for time-series sports data. All models use season-based or expanding-window CV.
- **Per-90 normalization** — Player stats normalized per 90 minutes to account for playing time differences.
- **Calibrated probabilities** — All probabilistic models are calibrated (Platt/isotonic) and evaluated with Brier score.
- **SHAP for every model** — Interpretability is not optional when communicating with sports stakeholders.

---

## 👤 Author

**David Bazaldua** — ML Engineer building data products for the sports industry.

- 🌐 [Portfolio]()
- 💼 [LinkedIn]()
- 📧 [Email]()

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.