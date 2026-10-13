import resource
import pandas as pd
import psutil

LIMIT = 4 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))

path = "tp2/dataset/itineraries.csv"

print("RAM before:", psutil.virtual_memory().percent, "%")

try:
    df = pd.read_csv(path, engine="python")
    print("Loaded:", df.shape)
except MemoryError:
    print("MemoryError: the file does not fit in 4 GB of RAM")