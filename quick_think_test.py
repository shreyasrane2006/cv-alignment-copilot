# quick_think_test.py
import ollama
import time

start = time.time()
response = ollama.chat(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    think=False,
)
print(response["message"]["content"])
print(f"Took {time.time() - start:.1f}s")