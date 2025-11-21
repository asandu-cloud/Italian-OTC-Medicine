import shutil
from pathlib import Path

# Folder where THIS script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root = go one directory up from /Scripts
PROJECT_ROOT = SCRIPT_DIR.parent

# The actual cache directory
CACHE_DIR = PROJECT_ROOT / ".cache"

def clear_cache():
    print(f"Script directory:      {SCRIPT_DIR}")
    print(f"Project root detected: {PROJECT_ROOT}")
    print(f"Cache directory:       {CACHE_DIR}")

    if CACHE_DIR.exists():
        print(f"\nDeleting cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
    else:
        print(f"\nNo cache directory found at: {CACHE_DIR}")

    # Recreate empty cache folder
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Re-created empty cache directory: {CACHE_DIR}")

if __name__ == "__main__":
    clear_cache()

