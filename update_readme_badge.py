import re
import datetime
import os

START_DATE = datetime.date(2026, 4, 15)
README_PATH = 'README.md'

def main():
    today = datetime.date.today()
    days_passed = (today - START_DATE).days + 1
    
    # Cap between 1 and 100
    if days_passed < 1:
        days_passed = 1
    elif days_passed > 100:
        days_passed = 100
        
    print(f"Current Day: {days_passed}")
    
    with open(README_PATH, 'r') as f:
        content = f.read()
        
    # Regex to find the progress badge
    # Example: [![Progress](https://img.shields.io/badge/Progress-Day%201%2F100-brightgreen?style=for-the-badge)]()
    new_content = re.sub(
        r'\[\!\[Progress\]\(https://img\.shields\.io/badge/Progress-Day%20\d+%2F100-brightgreen\?style=for-the-badge\)\]\(\)',
        f'[![Progress](https://img.shields.io/badge/Progress-Day%20{days_passed}%2F100-brightgreen?style=for-the-badge)]()',
        content
    )
    
    with open(README_PATH, 'w') as f:
        f.write(new_content)
        
    print("README.md updated.")

if __name__ == "__main__":
    main()
