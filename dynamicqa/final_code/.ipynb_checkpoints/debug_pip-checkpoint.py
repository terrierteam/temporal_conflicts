import sys
import huggingface_hub
import os

print("--- Python Executable ---")
print(sys.executable)
print("\n--- huggingface_hub version ---")
print(huggingface_hub.__version__)
print("\n--- huggingface_hub location ---")
print(huggingface_hub.__file__)
print("\n--- sys.path (Search Paths) ---")
for path in sys.path:
    print(path)

print("\n--- PYTHONPATH Environment Variable ---")
print(os.environ.get('PYTHONPATH'))