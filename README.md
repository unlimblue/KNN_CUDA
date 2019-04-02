# KNN_CUDA

+ ref: [kNN-CUDA](https://github.com/vincentfpgarcia/kNN-CUDA)
+ ref: [pytorch knn cuda](https://github.com/chrischoy/pytorch_knn_cuda)
+ author: [sli@mail.bnu.edu.cn](sli@mail.bnu.edu.cn)

#### Modifications 
+ Aten support
+ pytorch v1.0+ support
+ pytorch c++ extention need [ninja](https://github.com/ninja-build/ninja) 

#### Performance

+ dim   = 5
+ k     = 100
+ ref   = 224
+ query = 224
+ Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
+ NVIDIA GeForce 940MX

| Loop   | sklearn | CUDA    | Memory   |
| :---:  | :---:   | :---:   | :---:    |
| 100    | 2.34 ms | 0.06 ms | 652/1024 |
| 1000   | 2.30 ms | 1.40 ms | 652/1024 |


#### Usage

```python
import torch

# Make sure your CUDA is available.
assert torch.cuda.is_available()

import numpy as np
from knn_cuda import KNN
"""
if transpose_mode is True, 
    ref   is Tensor [nr x dim]
    query is Tensor [nq x dim]
    
    return 
        dist is Tensor [nq x k]
        indx is Tensor [nq x k]
else
    ref   is Tensor [dim x nr]
    query is Tensor [dim x nq]
    
    return 
        dist is Tensor [k x nq]
        indx is Tensor [k x nq]
"""

knn = KNN(k=10, transpose_mode=True)

ref = torch.from_numpy(np.random.random(1000, 5)).cuda()
query = torch.from_numpy(np.random.random(50, 5)).cuda()

dist, indx = knn(ref, query)
```
