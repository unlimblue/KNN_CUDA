import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


__version__ = "0.2"


def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    ext_csrc = os.path.join(root_dir, "csrc")
    ext_path = os.path.join(ext_csrc, "_ext", ext_name)
    os.makedirs(ext_path, exist_ok=True)
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False."
    ext_sources = [
        os.path.join(ext_csrc, "cuda", "{}.cpp".format(ext_name)),
        os.path.join(ext_csrc, "cuda", "{}.cu".format(ext_name))
    ]
    extra_cuda_cflags = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    ext = load(
        name=ext_name,
        sources=ext_sources,
        extra_cflags=["-O2"],
        build_directory=ext_path,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True
    )
    return ext


_knn = load_cpp_ext("knn")


def knn(ref, query, k):
    d, i = _knn.knn(ref, query, k)
    i -= 1
    return d, i


def _T(t, mode=False):
    if mode:
        return t.transpose(0, 1).contiguous()
    else:
        return t


class KNN(nn.Module):

    def __init__(self, k, transpose_mode=False):
        super(KNN, self).__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):
        assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
        with torch.no_grad():
            batch_size = ref.size(0)
            D, I = [], []
            for bi in range(batch_size):
                r, q = _T(ref[bi], self._t), _T(query[bi], self._t)
                d, i = knn(r.float(), q.float(), self.k)
                d, i = _T(d, self._t), _T(i, self._t)
                D.append(d)
                I.append(i)
            D = torch.stack(D, dim=0)
            I = torch.stack(I, dim=0)
        return D, I

