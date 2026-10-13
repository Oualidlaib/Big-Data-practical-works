import time
import dask
import dask.dataframe as dd

path = "tp2/dataset/itineraries.csv"


ddf = dd.read_csv(path, usecols=["totalFare", "baseFare"], blocksize="128MB",
                  dtype={"totalFare": "float32", "baseFare": "float32"})
print("Partitions:", ddf.npartitions)


mean_fare = ddf["totalFare"].mean()
sum_fare = ddf["totalFare"].sum()
print(type(mean_fare))        

t = time.time()
a = mean_fare.compute()
b = sum_fare.compute()
print("Separate computes:", round(time.time() - t, 1), "s")

t = time.time()
a, b = dask.compute(mean_fare, sum_fare)
print("Single compute:", round(time.time() - t, 1), "s")

print("Mean:", a, "| Sum:", b)