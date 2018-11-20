import torch
import numpy as np
from sklearn.neighbors import KDTree
from knn_cuda import KNN
import time
import numpy.testing as npt
import traceback


def timeit(n, func, *args, **kwargs):
    n = max(n, 1)
    s = time.time()
    for _ in range(n):
        func(*args, **kwargs)
    t = 1000 * (time.time() - s) / n
    return func.__name__, t


def test_time(n):
    print("timing......")
    dim = 5
    k = 100
    np_ref = np.random.random((224, dim))
    np_query = np.random.random((224, dim))

    torch_ref = torch.from_numpy(np_ref).float().cuda()
    torch_query = torch.from_numpy(np_query).float().cuda()

    def np_func():
        tree = KDTree(np_ref, leaf_size=100)
        return tree.query(np_query, k=k)

    def cu_func():
        knn = KNN(k)
        return knn(torch_ref, torch_query)

    print("{} use: {:.2f} ms".format(*timeit(n, np_func)))
    print("{} use: {:.2f} ms".format(*timeit(n, cu_func)))
    print("Mem", torch.cuda.max_memory_allocated() / 1024, torch.cuda.max_memory_cached() / 1024)


def t2n(t):
    return t.detach().cpu().numpy()


def assert_eq():
    dim = 5
    k = 100
    np_ref = np.random.random((224, dim))
    np_query = np.random.random((224, dim))

    torch_ref = torch.from_numpy(np_ref).float().cuda()
    torch_query = torch.from_numpy(np_query).float().cuda()
    tree = KDTree(np_ref, leaf_size=100)
    np_d, np_i = tree.query(np_query, k=k)
    np_d, np_i = np.sort(np_d), np.sort(np_i)

    knn = KNN(k, transpose_mode=True)
    t_d, t_i = knn(torch_ref, torch_query)
    t_d, t_i = np.sort(t2n(t_d)), np.sort(t2n(t_i))
    
    npt.assert_almost_equal(np_d, t_d, decimal=4)
    diff = np.abs(np_i - t_i)
    assert diff.sum() == 0, "DIFF {}, {}".format(np_i[diff > 0], t_i[diff > 0])


for i in range(100):
    try:
        assert_eq()
        print("\033[32mPASS\033[0m")
    except:
        traceback.print_exc()
        print("\033[31mERROR\033[0m")
        input("Type enter")

test_time(1000)
