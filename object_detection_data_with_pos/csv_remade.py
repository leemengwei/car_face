import numpy as np
import pandas as pd
from IPython import embed
import sys

filename = sys.argv[1]

data = pd.read_csv(filename, header=None)

for i in [1,2,3,4]:
    data[i] = np.round(data[i].values).astype(int)

data.to_csv(filename, header=None, index=None)
print("Done")
