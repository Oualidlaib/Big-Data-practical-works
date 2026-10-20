import os
import time
import threading
import psutil
import pandas as pd

proc = psutil.Process(os.getpid())

class MemSampler(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()   # renamed

    def run(self):
        while not self._stop_event.is_set():
            self.samples.append(proc.memory_info().rss / 1024**2)  # MB
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join()

baseline = proc.memory_info().rss / 1024**2
sampler = MemSampler(interval=0.1)

total = 0
rows = 0

t = time.time()
sampler.start()
for chunk in pd.read_csv("dataset/itineraries.csv", usecols=["totalFare"], chunksize=100_000):
    total += chunk["totalFare"].sum()
    rows += len(chunk)
sampler.stop()
elapsed = time.time() - t

peak = max(sampler.samples)
avg = sum(sampler.samples) / len(sampler.samples)

print(f"Time: {elapsed:.1f} s")
print(f"Baseline RAM: {baseline:.0f} MB")
print(f"Peak RAM: {peak:.0f} MB (+{peak - baseline:.0f} MB over baseline)")
print(f"Avg RAM: {avg:.0f} MB")
print("Sum of totalFare:", total)
print("Mean of totalFare:", total / rows)