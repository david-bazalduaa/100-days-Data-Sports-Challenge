document.addEventListener('DOMContentLoaded', () => {
    // Interactive progress logic (if needed to enhance Jekyll static output)
    const items = document.querySelectorAll('.tracker-list li');
    let completedCount = 0;
    
    items.forEach(item => {
        if (item.classList.contains('status-completed')) {
            completedCount++;
        }
    });
    
    const progressBar = document.getElementById('overall-progress');
    const progressText = document.getElementById('progress-text');
    
    if (progressBar && progressText && items.length > 0) {
        const percentage = Math.round((completedCount / items.length) * 100);
        progressBar.value = percentage;
        progressText.innerText = `${completedCount} / ${items.length} Days Completed (${percentage}%)`;
    }
});
