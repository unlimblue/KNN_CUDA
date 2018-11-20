#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, t) AT_ASSERTM(x.dtype() == t, #x " must be " #t)
#define CHECK_CUDA(x) AT_ASSERTM(x.device().type() == at::Device::Type::CUDA, #x " must be on CUDA")
#define CHECK_INPUT(x, t) CHECK_CONTIGUOUS(x); CHECK_TYPE(x, t); CHECK_CUDA(x)


void knn_device(
    float* ref_dev, 
    int ref_nb, 
    float* query_dev, 
    int query_nb, 
    int dim, 
    int k, 
    float* dist_dev, 
    long* ind_dev, 
    cudaStream_t stream
    );

std::vector<at::Tensor> knn(
    at::Tensor & ref, 
    at::Tensor & query, 
    const int k
    ){

    CHECK_INPUT(ref, at::kFloat);
    CHECK_INPUT(query, at::kFloat);
    int dim = ref.size(0);
    int ref_nb = ref.size(1);
    int query_nb = query.size(1);
    float * ref_dev = ref.data<float>();
    float * query_dev = query.data<float>();
    auto dist = at::empty({ref_nb, query_nb}, query.options().dtype(at::kFloat));
    auto ind = at::empty({k, query_nb}, query.options().dtype(at::kLong));
    float * dist_dev = dist.data<float>();
    long * ind_dev = ind.data<long>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knn_device(
        ref_dev,
        ref_nb,
        query_dev,
        query_nb,
        dim,
        k,
        dist_dev,
        ind_dev,
        stream
    );

    return {dist.slice(0, 0, k), ind};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn", &knn, "KNN cuda version");
}

