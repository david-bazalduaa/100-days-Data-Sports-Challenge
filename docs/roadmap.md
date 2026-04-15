---
layout: default
title: Roadmap & Tracker
---
<script src="{{ '/assets/js/progress.js' | relative_url }}"></script>

# 100-Day Interactive Tracker

<div class="progress-container" style="margin: 2rem 0; padding: 1.5rem; background: var(--card-bg); border-radius: 8px;">
    <h2>Overall Progress</h2>
    <progress id="overall-progress" value="0" max="100" style="width: 100%; height: 20px;"></progress>
    <p id="progress-text" style="text-align: right; font-weight: bold; margin-top: 0.5rem;">Calculating...</p>
</div>

## Phase 1: Foundations & Infrastructure (Days 1–15)

<ul class="tracker-list" style="list-style-type: none; padding-left: 0;">
    {% for item in site.data.progress %}
    <li class="status-{{ item.status }}" style="margin-bottom: 0.8rem; padding: 1rem; border-left: 4px solid {% if item.status == 'completed' %}#10b981{% elsif item.status == 'in-progress' %}#f59e0b{% else %}#334155{% endif %}; background: rgba(0,0,0,0.2);">
        <strong>Day {{ item.day }}:</strong> {{ item.objective }} 
        <span style="float: right; font-size: 0.9em; padding: 2px 6px; border-radius: 4px; background: {% if item.status == 'completed' %}#064e3b{% else %}#1e293b{% endif %};">
            {{ item.status | capitalize }}
        </span>
    </li>
    {% endfor %}
</ul>
