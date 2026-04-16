// ============================================================
// 100 Days of Data & Sports — App Logic
// ============================================================

const PROGRESS_DATA = [
    { day: 1, phase: 1, objective: "Repo Architecture", status: "completed" },
    { day: 2, phase: 1, objective: "GitHub Pages Setup", status: "completed" },
    { day: 3, phase: 1, objective: "Progress Tracker", status: "completed" },
    { day: 4, phase: 1, objective: "StatsBomb Open Data", status: "pending" },
    { day: 5, phase: 1, objective: "nflverse Data", status: "pending" },
    { day: 6, phase: 1, objective: "FBref Scraper", status: "pending" },
    { day: 7, phase: 1, objective: "Ingestion Pipeline", status: "pending" },
    { day: 8, phase: 1, objective: "Football Data Quality", status: "pending" },
    { day: 9, phase: 1, objective: "NFL Data Quality", status: "pending" },
    { day: 10, phase: 1, objective: "Football Schema", status: "pending" },
    { day: 11, phase: 1, objective: "NFL Schema", status: "pending" },
    { day: 12, phase: 1, objective: "Tracking Data Intro", status: "pending" },
    { day: 13, phase: 1, objective: "Visualization Library", status: "pending" },
    { day: 14, phase: 1, objective: "Publish Phase 1", status: "pending" },
    { day: 15, phase: 1, objective: "Phase 1 Retrospective", status: "pending" },
    { day: 16, phase: 2, objective: "Shot Analysis EDA", status: "pending" },
    { day: 17, phase: 2, objective: "xG Features Part 1", status: "pending" },
    { day: 18, phase: 2, objective: "xG Geometrical Features", status: "pending" },
    { day: 19, phase: 2, objective: "Passing Networks", status: "pending" },
    { day: 20, phase: 2, objective: "NFL Offensive EDA", status: "pending" },
    { day: 21, phase: 2, objective: "NFL Player Features", status: "pending" },
    { day: 22, phase: 2, objective: "Rolling Features", status: "pending" },
    { day: 23, phase: 2, objective: "Per-90 and Radar Charts", status: "pending" },
    { day: 24, phase: 2, objective: "Pressing Analysis", status: "pending" },
    { day: 25, phase: 2, objective: "NFL Situational Analysis", status: "pending" },
    { day: 26, phase: 2, objective: "Player Heatmaps KDE", status: "pending" },
    { day: 27, phase: 2, objective: "Player Clustering", status: "pending" },
    { day: 28, phase: 2, objective: "Player Similarity Engine", status: "pending" },
    { day: 29, phase: 2, objective: "Project 1 README", status: "pending" },
    { day: 30, phase: 2, objective: "NFL Draft Combine EDA", status: "pending" },
    { day: 31, phase: 2, objective: "Draft Features", status: "pending" },
    { day: 32, phase: 2, objective: "Possession Analysis", status: "pending" },
    { day: 33, phase: 2, objective: "Formation Analysis", status: "pending" },
    { day: 34, phase: 2, objective: "Journal Post Phase 2", status: "pending" },
    { day: 35, phase: 2, objective: "Phase 2 Retrospective", status: "pending" },
    { day: 36, phase: 3, objective: "Temporal Validation Framework", status: "pending" },
    { day: 37, phase: 3, objective: "xG Baseline Logistic", status: "pending" },
    { day: 38, phase: 3, objective: "xG Boosting and SHAP", status: "pending" },
    { day: 39, phase: 3, objective: "xG Calibration and Errors", status: "pending" },
    { day: 40, phase: 3, objective: "xG Packaging Project 2", status: "pending" },
    { day: 41, phase: 3, objective: "NFL WP Features", status: "pending" },
    { day: 42, phase: 3, objective: "NFL WP Baseline", status: "pending" },
    { day: 43, phase: 3, objective: "NFL WP Advanced", status: "pending" },
    { day: 44, phase: 3, objective: "NFL WP Packaging Project 3", status: "pending" },
    { day: 45, phase: 3, objective: "Match Result Prediction", status: "pending" },
    { day: 46, phase: 3, objective: "RPS and Odds Analysis", status: "pending" },
    { day: 47, phase: 3, objective: "Elo Power Rankings", status: "pending" },
    { day: 48, phase: 3, objective: "Elo as Feature", status: "pending" },
    { day: 49, phase: 3, objective: "MLflow Setup", status: "pending" },
    { day: 50, phase: 3, objective: "NFL Player Projection", status: "pending" },
    { day: 51, phase: 3, objective: "Bayesian Estimation", status: "pending" },
    { day: 52, phase: 3, objective: "xG Chain Value", status: "pending" },
    { day: 53, phase: 3, objective: "Simplified VAEP", status: "pending" },
    { day: 54, phase: 3, objective: "Ensemble and Stacking", status: "pending" },
    { day: 55, phase: 3, objective: "Draft Value Predictor", status: "pending" },
    { day: 56, phase: 3, objective: "Aging Curves", status: "pending" },
    { day: 57, phase: 3, objective: "Market Value Prediction", status: "pending" },
    { day: 58, phase: 3, objective: "Journal Post Phase 3", status: "pending" },
    { day: 59, phase: 3, objective: "Code Review and Refactoring", status: "pending" },
    { day: 60, phase: 3, objective: "Phase 3 Retrospective", status: "pending" },
    { day: 61, phase: 4, objective: "Scouting Dashboard Wireframe", status: "pending" },
    { day: 62, phase: 4, objective: "Scouting Dashboard Backend", status: "pending" },
    { day: 63, phase: 4, objective: "Scouting Dashboard UI Part 1", status: "pending" },
    { day: 64, phase: 4, objective: "Scouting Dashboard UI Part 2", status: "pending" },
    { day: 65, phase: 4, objective: "Scouting Dashboard Deploy", status: "pending" },
    { day: 66, phase: 4, objective: "NFL API Design", status: "pending" },
    { day: 67, phase: 4, objective: "NFL API Implementation", status: "pending" },
    { day: 68, phase: 4, objective: "NFL API Docker Deploy", status: "pending" },
    { day: 69, phase: 4, objective: "NFL API Docs and Demo", status: "pending" },
    { day: 70, phase: 4, objective: "Match Forecaster Design", status: "pending" },
    { day: 71, phase: 4, objective: "Match Forecaster Pipeline", status: "pending" },
    { day: 72, phase: 4, objective: "Match Forecaster Frontend", status: "pending" },
    { day: 73, phase: 4, objective: "Match Forecaster Backtesting", status: "pending" },
    { day: 74, phase: 4, objective: "Match Forecaster Packaging", status: "pending" },
    { day: 75, phase: 4, objective: "Game Flow Visualizer Football", status: "pending" },
    { day: 76, phase: 4, objective: "Game Flow Visualizer NFL", status: "pending" },
    { day: 77, phase: 4, objective: "Game Flow Deploy", status: "pending" },
    { day: 78, phase: 4, objective: "Route Classification NFL", status: "pending" },
    { day: 79, phase: 4, objective: "Pitch Control Model", status: "pending" },
    { day: 80, phase: 4, objective: "Automated Weekly Reports", status: "pending" },
    { day: 81, phase: 4, objective: "MLflow Model Registry", status: "pending" },
    { day: 82, phase: 4, objective: "Model Drift Monitoring", status: "pending" },
    { day: 83, phase: 4, objective: "Journal Post Phase 4", status: "pending" },
    { day: 84, phase: 4, objective: "Final Code Quality Pass", status: "pending" },
    { day: 85, phase: 4, objective: "Phase 4 Retrospective", status: "pending" },
    { day: 86, phase: 5, objective: "Portfolio Grade READMEs", status: "pending" },
    { day: 87, phase: 5, objective: "Hero Images and GIFs", status: "pending" },
    { day: 88, phase: 5, objective: "Web Project Pages", status: "pending" },
    { day: 89, phase: 5, objective: "Web Journal and Timeline", status: "pending" },
    { day: 90, phase: 5, objective: "About Me Page", status: "pending" },
    { day: 91, phase: 5, objective: "CV Update", status: "pending" },
    { day: 92, phase: 5, objective: "LinkedIn Post 1", status: "pending" },
    { day: 93, phase: 5, objective: "LinkedIn Post 2", status: "pending" },
    { day: 94, phase: 5, objective: "LinkedIn Post 3 Draft", status: "pending" },
    { day: 95, phase: 5, objective: "Interview Prep Questions", status: "pending" },
    { day: 96, phase: 5, objective: "Interview Prep Case Study", status: "pending" },
    { day: 97, phase: 5, objective: "Portfolio One Pager", status: "pending" },
    { day: 98, phase: 5, objective: "Web Final QA", status: "pending" },
    { day: 99, phase: 5, objective: "Final Cleanup", status: "pending" },
    { day: 100, phase: 5, objective: "Final Publication", status: "pending" }
];

// ── Build Grid ──
function buildGrid() {
    const grid = document.getElementById('grid-100');
    if (!grid) return;

    grid.innerHTML = '';
    PROGRESS_DATA.forEach(item => {
        const cell = document.createElement('div');
        cell.className = `day-cell phase-${item.phase} ${item.status}`;
        cell.textContent = item.day;
        cell.dataset.day = item.day;
        cell.addEventListener('click', () => showDetail(item));
        grid.appendChild(cell);
    });
}

// ── Show Detail ──
function showDetail(item) {
    const detail = document.getElementById('day-detail');
    const title = document.getElementById('detail-title');
    const obj = document.getElementById('detail-objective');
    const status = document.getElementById('detail-status');

    title.textContent = `Day ${item.day}`;
    obj.textContent = item.objective;
    status.textContent = item.status === 'completed' ? 'Completed' : 'Pending';
    status.className = `detail-status ${item.status === 'pending' ? 'pending-status' : ''}`;

    detail.classList.add('active');
}

// ── Update Stats ──
function updateStats() {
    const completed = PROGRESS_DATA.filter(d => d.status === 'completed').length;
    const pct = Math.round((completed / 100) * 100);

    const statDays = document.getElementById('stat-days');
    const progressBar = document.getElementById('progress-bar');
    const progressPct = document.getElementById('progress-pct');

    if (statDays) statDays.textContent = completed;
    if (progressBar) progressBar.style.width = pct + '%';
    if (progressPct) progressPct.textContent = pct;
}

// ── Nav Scroll ──
function initNav() {
    const nav = document.getElementById('nav');
    if (!nav) return;
    
    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 20);
    });
}

// ── Close Detail ──
function initDetailClose() {
    const closeBtn = document.getElementById('day-detail-close');
    const detail = document.getElementById('day-detail');
    if (closeBtn && detail) {
        closeBtn.addEventListener('click', () => {
            detail.classList.remove('active');
        });
    }
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    buildGrid();
    updateStats();
    initNav();
    initDetailClose();
});
