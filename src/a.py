import numpy as np

im = (np.indices((10, 10)).sum(axis=0) % 2).astype(np.float32)
print(im * 255)