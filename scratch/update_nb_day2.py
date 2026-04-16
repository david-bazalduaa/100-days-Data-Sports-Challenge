import json
import os

notebook_path = '/Users/davidbazalduamendez/Documents/GitHub/100-days-Data-Sports-Challenge/notebooks/phase1_foundations/day002_github_pages_setup.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# The cell to replace is the second one (index 1)
new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "## Day 2: Implementation Log\n",
     "\n",
     "### Tech Stack Decisions\n",
     "- **Custom Static Site:** Decided to move away from a standard Jekyll theme to a pure HTML/CSS/JS architecture. This allows for a more tailored, minimalist design and the implementation of custom interactive features like the 100-day tracker without Ruby dependencies.\n",
     "- **Aesthetics:** Implemented a 'Sports-Analytics' theme with a clean hero section, subtle field-line decorations, and high-impact typography (Inter).\n",
     "\n",
     "### Key Features Implemented\n",
     "1.  **100-Day Tracker Grid:** A 10x10 responsive grid representing the full course roadmap. Each cell is color-coded by phase (Foundations, EDA, Modeling, Products, Polish).\n",
     "2.  **Persistence Logic:** Added a JavaScript helper using `localStorage`. This allows me to mark days as completed by clicking on them, and the state persists even after closing the browser.\n",
     "3.  **Live Stats:** The hero section now features a dynamic counter that shows real-time progress based on the tracker state.\n",
     "4.  **NoJekyll Deployment:** Added a `.nojekyll` file to the `/docs` directory to ensure GitHub Pages serves the raw static assets correctly.\n",
     "\n",
     "### Execution Steps\n",
     "```bash\n",
     "# Created index.html, style.css, and app.js\n",
     "# Cleaned up legacy Jekyll files from /docs\n",
     "# Added .nojekyll to /docs\n",
     "# Committed and pushed to origin main\n",
     "```\n",
     "\n",
     "**Status:** Functional and Deployed."
    ]
}

nb['cells'][1] = new_cell

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
