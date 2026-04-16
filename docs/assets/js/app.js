// ============================================================
// 100 Days of Data & Sports — App Logic
// ============================================================

const STORAGE_KEY = 'sports100_completed';

const PROGRESS_DATA = [
    { day: 1, phase: 1, objective: "Repo Architecture" },
    { day: 2, phase: 1, objective: "GitHub Pages Setup" },
    { day: 3, phase: 1, objective: "Progress Tracker" },
    { day: 4, phase: 1, objective: "StatsBomb Open Data" },
    { day: 5, phase: 1, objective: "nflverse Data" },
    { day: 6, phase: 1, objective: "FBref Scraper" },
    { day: 7, phase: 1, objective: "Ingestion Pipeline" },
    { day: 8, phase: 1, objective: "Football Data Quality" },
    { day: 9, phase: 1, objective: "NFL Data Quality" },
    { day: 10, phase: 1, objective: "Football Schema" },
    { day: 11, phase: 1, objective: "NFL Schema" },
    { day: 12, phase: 1, objective: "Tracking Data Intro" },
    { day: 13, phase: 1, objective: "Visualization Library" },
    { day: 14, phase: 1, objective: "Publish Phase 1" },
    { day: 15, phase: 1, objective: "Phase 1 Retrospective" },
    { day: 16, phase: 2, objective: "Shot Analysis EDA" },
    { day: 17, phase: 2, objective: "xG Features Part 1" },
    { day: 18, phase: 2, objective: "xG Geometrical Features" },
    { day: 19, phase: 2, objective: "Passing Networks" },
    { day: 20, phase: 2, objective: "NFL Offensive EDA" },
    { day: 21, phase: 2, objective: "NFL Player Features" },
    { day: 22, phase: 2, objective: "Rolling Features" },
    { day: 23, phase: 2, objective: "Per-90 and Radar Charts" },
    { day: 24, phase: 2, objective: "Pressing Analysis" },
    { day: 25, phase: 2, objective: "NFL Situational Analysis" },
    { day: 26, phase: 2, objective: "Player Heatmaps KDE" },
    { day: 27, phase: 2, objective: "Player Clustering" },
    { day: 28, phase: 2, objective: "Player Similarity Engine" },
    { day: 29, phase: 2, objective: "Project 1 README" },
    { day: 30, phase: 2, objective: "NFL Draft Combine EDA" },
    { day: 31, phase: 2, objective: "Draft Features" },
    { day: 32, phase: 2, objective: "Possession Analysis" },
    { day: 33, phase: 2, objective: "Formation Analysis" },
    { day: 34, phase: 2, objective: "Journal Post Phase 2" },
    { day: 35, phase: 2, objective: "Phase 2 Retrospective" },
    { day: 36, phase: 3, objective: "Temporal Validation Framework" },
    { day: 37, phase: 3, objective: "xG Baseline Logistic" },
    { day: 38, phase: 3, objective: "xG Boosting and SHAP" },
    { day: 39, phase: 3, objective: "xG Calibration and Errors" },
    { day: 40, phase: 3, objective: "xG Packaging Project 2" },
    { day: 41, phase: 3, objective: "NFL WP Features" },
    { day: 42, phase: 3, objective: "NFL WP Baseline" },
    { day: 43, phase: 3, objective: "NFL WP Advanced" },
    { day: 44, phase: 3, objective: "NFL WP Packaging Project 3" },
    { day: 45, phase: 3, objective: "Match Result Prediction" },
    { day: 46, phase: 3, objective: "RPS and Odds Analysis" },
    { day: 47, phase: 3, objective: "Elo Power Rankings" },
    { day: 48, phase: 3, objective: "Elo as Feature" },
    { day: 49, phase: 3, objective: "MLflow Setup" },
    { day: 50, phase: 3, objective: "NFL Player Projection" },
    { day: 51, phase: 3, objective: "Bayesian Estimation" },
    { day: 52, phase: 3, objective: "xG Chain Value" },
    { day: 53, phase: 3, objective: "Simplified VAEP" },
    { day: 54, phase: 3, objective: "Ensemble and Stacking" },
    { day: 55, phase: 3, objective: "Draft Value Predictor" },
    { day: 56, phase: 3, objective: "Aging Curves" },
    { day: 57, phase: 3, objective: "Market Value Prediction" },
    { day: 58, phase: 3, objective: "Journal Post Phase 3" },
    { day: 59, phase: 3, objective: "Code Review and Refactoring" },
    { day: 60, phase: 3, objective: "Phase 3 Retrospective" },
    { day: 61, phase: 4, objective: "Scouting Dashboard Wireframe" },
    { day: 62, phase: 4, objective: "Scouting Dashboard Backend" },
    { day: 63, phase: 4, objective: "Scouting Dashboard UI Part 1" },
    { day: 64, phase: 4, objective: "Scouting Dashboard UI Part 2" },
    { day: 65, phase: 4, objective: "Scouting Dashboard Deploy" },
    { day: 66, phase: 4, objective: "NFL API Design" },
    { day: 67, phase: 4, objective: "NFL API Implementation" },
    { day: 68, phase: 4, objective: "NFL API Docker Deploy" },
    { day: 69, phase: 4, objective: "NFL API Docs and Demo" },
    { day: 70, phase: 4, objective: "Match Forecaster Design" },
    { day: 71, phase: 4, objective: "Match Forecaster Pipeline" },
    { day: 72, phase: 4, objective: "Match Forecaster Frontend" },
    { day: 73, phase: 4, objective: "Match Forecaster Backtesting" },
    { day: 74, phase: 4, objective: "Match Forecaster Packaging" },
    { day: 75, phase: 4, objective: "Game Flow Visualizer Football" },
    { day: 76, phase: 4, objective: "Game Flow Visualizer NFL" },
    { day: 77, phase: 4, objective: "Game Flow Deploy" },
    { day: 78, phase: 4, objective: "Route Classification NFL" },
    { day: 79, phase: 4, objective: "Pitch Control Model" },
    { day: 80, phase: 4, objective: "Automated Weekly Reports" },
    { day: 81, phase: 4, objective: "MLflow Model Registry" },
    { day: 82, phase: 4, objective: "Model Drift Monitoring" },
    { day: 83, phase: 4, objective: "Journal Post Phase 4" },
    { day: 84, phase: 4, objective: "Final Code Quality Pass" },
    { day: 85, phase: 4, objective: "Phase 4 Retrospective" },
    { day: 86, phase: 5, objective: "Portfolio Grade READMEs" },
    { day: 87, phase: 5, objective: "Hero Images and GIFs" },
    { day: 88, phase: 5, objective: "Web Project Pages" },
    { day: 89, phase: 5, objective: "Web Journal and Timeline" },
    { day: 90, phase: 5, objective: "About Me Page" },
    { day: 91, phase: 5, objective: "CV Update" },
    { day: 92, phase: 5, objective: "LinkedIn Post 1" },
    { day: 93, phase: 5, objective: "LinkedIn Post 2" },
    { day: 94, phase: 5, objective: "LinkedIn Post 3 Draft" },
    { day: 95, phase: 5, objective: "Interview Prep Questions" },
    { day: 96, phase: 5, objective: "Interview Prep Case Study" },
    { day: 97, phase: 5, objective: "Portfolio One Pager" },
    { day: 98, phase: 5, objective: "Web Final QA" },
    { day: 99, phase: 5, objective: "Final Cleanup" },
    { day: 100, phase: 5, objective: "Final Publication" }
];

// ── LocalStorage helpers ──
function getCompletedDays() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : [1, 2, 3]; // days 1-3 completed by default
    } catch {
        return [1, 2, 3];
    }
}

function saveCompletedDays(days) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(days));
}

function isDayCompleted(day) {
    return getCompletedDays().includes(day);
}

function toggleDay(day) {
    const days = getCompletedDays();
    const idx = days.indexOf(day);
    if (idx === -1) {
        days.push(day);
    } else {
        days.splice(idx, 1);
    }
    days.sort((a, b) => a - b);
    saveCompletedDays(days);
}

// ── Build Grid ──
function buildGrid() {
    const grid = document.getElementById('grid-100');
    if (!grid) return;

    grid.innerHTML = '';
    PROGRESS_DATA.forEach(item => {
        const isCompleted = isDayCompleted(item.day);
        const cell = document.createElement('div');
        cell.className = `day-cell phase-${item.phase} ${isCompleted ? 'completed' : 'pending'}`;
        cell.textContent = item.day;
        cell.dataset.day = item.day;
        cell.title = `Day ${item.day}: ${item.objective} — Click to toggle`;
        cell.addEventListener('click', () => {
            toggleDay(item.day);
            buildGrid();
            updateStats();
            showDetail(item);
        });
        grid.appendChild(cell);
    });
}

// ── Show Detail ──
let currentDetailDay = null;

function showDetail(item) {
    currentDetailDay = item.day;
    const isCompleted = isDayCompleted(item.day);

    const detail = document.getElementById('day-detail');
    const title = document.getElementById('detail-title');
    const obj = document.getElementById('detail-objective');
    const status = document.getElementById('detail-status');
    const toggleBtn = document.getElementById('detail-toggle');

    title.textContent = `Day ${item.day}`;
    obj.textContent = item.objective;

    status.textContent = isCompleted ? 'Completed' : 'Pending';
    status.className = `detail-status ${isCompleted ? '' : 'pending-status'}`;

    toggleBtn.textContent = isCompleted ? 'Mark as Pending' : 'Mark as Completed';
    toggleBtn.className = `detail-toggle-btn ${isCompleted ? 'is-completed' : ''}`;

    detail.classList.add('active');
}

// ── Update Stats ──
function updateStats() {
    const completed = getCompletedDays().length;
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

// ── Toggle from Detail Panel ──
function initDetailToggle() {
    const toggleBtn = document.getElementById('detail-toggle');
    if (!toggleBtn) return;
    toggleBtn.addEventListener('click', () => {
        if (currentDetailDay === null) return;
        toggleDay(currentDetailDay);
        buildGrid();
        updateStats();
        // Re-show the detail with updated state
        const item = PROGRESS_DATA.find(d => d.day === currentDetailDay);
        if (item) showDetail(item);
    });
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    // Seed localStorage on first visit
    if (!localStorage.getItem(STORAGE_KEY)) {
        saveCompletedDays([1, 2, 3]);
    }
    buildGrid();
    updateStats();
    initNav();
    initDetailClose();
    initDetailToggle();
});
