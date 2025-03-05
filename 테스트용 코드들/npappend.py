import numpy as np

b=np.array([[1,2],[3,4]])
tmp = np.stack((b, b), axis=2)
print(tmp)
