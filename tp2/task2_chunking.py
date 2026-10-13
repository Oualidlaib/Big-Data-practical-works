import pandas as pd

total = 0
rows = 0

for chunk in pd.read_csv("tp2/dataset/itineraries.csv", usecols=["totalFare"], chunksize=100_000):
    total += chunk["totalFare"].sum()
    rows += len(chunk)

print("Sum of totalFare:", total)
print("Mean of totalFare:", total / rows)