import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import time
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Importing recommender...")
from recommender import ContentBasedRecommender

print("Initializing Recommender...")
def progress(p, m):
    print(f"[{p}%] {m}")

try:
    rec = ContentBasedRecommender(progress_callback=progress)
    print("Initialization successful!")
except Exception as e:
    print(f"Initialization failed: {e}")
    import traceback
    traceback.print_exc()
