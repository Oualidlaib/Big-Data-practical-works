import time
import dask
import dask.dataframe as dd
from dask.diagnostics import ResourceProfiler, ProgressBar

path = "dataset/itineraries.csv"

ddf = dd.read_csv(path, usecols=["totalFare", "baseFare"], blocksize="128MB",
                  dtype={"totalFare": "float32", "baseFare": "float32"})
print("Partitions:", ddf.npartitions)

mean_fare = ddf["totalFare"].mean()
sum_fare = ddf["totalFare"].sum()

t = time.time()
with ResourceProfiler(dt=0.25) as rprof, ProgressBar():
    a, b = dask.compute(mean_fare, sum_fare)
elapsed = time.time() - t

mem = [r.mem for r in rprof.results]   # MB, sampled every 0.25 s
cpu = [r.cpu for r in rprof.results]   # % CPU

print(f"Single compute: {elapsed:.1f} s")
print(f"Peak RAM: {max(mem):.0f} MB | Avg RAM: {sum(mem)/len(mem):.0f} MB")
print(f"Peak CPU: {max(cpu):.0f}%")
print("Mean:", a, "| Sum:", b)