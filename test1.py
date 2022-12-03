import torch
import numpy as np

a=np.array([[2,2],[1,1]])
b=np.array([[2,3],[1,4]])
a=torch.from_numpy(a)
# b=torch.from_numpy(b)
print('a',a)
print('b',b)
print((a==b).sum().item())
