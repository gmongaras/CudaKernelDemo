#include <cuda_runtime.h> // For cudaMemcpy and cudaFree
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/autocast_mode.h>
// #include <torch/extension.h>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <chrono>

#include <cuda_fp16.h> // Include CUDA half-precision definitions



// General AtomicAdd_
template<typename T>
__device__ void AtomicAdd_(T* address, T val) {
    atomicAdd(address, val);
}
// Specialization for half precision
template<>
__device__ void AtomicAdd_(at::Half* address, at::Half val) {
    atomicAdd(reinterpret_cast<__half*>(address), *reinterpret_cast<__half*>(&val));
}
// Specialization for bfloat16 half precision
template<>
__device__ void AtomicAdd_(at::BFloat16* address, at::BFloat16 val) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), *reinterpret_cast<__nv_bfloat16*>(&val));
}




// General __shfl_down_sync
template<typename T>
__device__ T __shfl_down_sync_(unsigned mask, T val, int delta, int width = warpSize) {
    return __shfl_down_sync(mask, val, delta, width);
}
// Specialization for half precision
template<>
__device__ at::Half __shfl_down_sync_(unsigned mask, at::Half val, int delta, int width) {
    return __shfl_down_sync(mask, *reinterpret_cast<__half*>(&val), delta, width);
}
// Specialization for bfloat16 half precision
template<>
__device__ at::BFloat16 __shfl_down_sync_(unsigned mask, at::BFloat16 val, int delta, int width) {
    return __shfl_down_sync(mask, *reinterpret_cast<__nv_bfloat16*>(&val), delta, width);
}





template<typename T>
__global__ void forward_kernel(
    const T* A, const T* B,
    T* output,
    int N, int M, int d
    ) {

    int block_size_x = 32;
    int block_size_y = 32;
    
    // Block indices
    int block_idx_x = blockIdx.x * block_size_x;
    int block_idx_y = blockIdx.y * block_size_y;

    // Thread indices
    int thread_idx_x = threadIdx.x; // thread in the block on the x-axis
    int thread_idx_y = threadIdx.y; // thread in the block on the y-axis


    // Calculate the block-thread index
    int block_thread_idx_x = block_idx_x + thread_idx_x;
    int block_thread_idx_y = block_idx_y + thread_idx_y;


    // Shared memory for reduction
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];T *shared_mem = reinterpret_cast<T *>(shared_memory_uchar);

    // Sum of the product of the elements
    T sum = 0;

    // Iterate over the dimension
    for (int d_idx = 0; d_idx < d; d_idx++) {
        // Multiply the elements, add to the sum of this output element
        sum += A[block_thread_idx_x * d + d_idx] * B[block_thread_idx_y * d + d_idx];
    }

    // Store the sum in the output matrix at output[A_idx, B_idx]
    output[block_thread_idx_x * M + block_thread_idx_y] = sum;
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward(
    const T* A, const T* B, T* output,
    int N, int M, int d,
    cudaStream_t stream = 0) {
    
    dim3 grid(N/32, M/32); // Number of "blocks" is how many groups of 32x32 threads we want.
    dim3 block(32, 32); // Numebr of threads per "block" will be 32x32 - the max number of threads.
    int shared_memory_size = 0; // No shared memory being used. We will add this later :)
    
    // Luanch the kernel and do the computation
    forward_kernel<T><<<grid, block, shared_memory_size, stream>>>(A, B, output, N, M, d);
}




// C++ interface
template<typename dtype_>
torch::Tensor DotProduct(torch::Tensor& A, torch::Tensor& B) {
    // Must be a CUDA tensor
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");

    // Get tensor dimension
    int N = A.size(0);
    int M = B.size(0);
    int d = A.size(1);

    // B should be the same size as A
    TORCH_CHECK(B.size(1) == d, "B must have the same dimension size as A");

    // Get the data type, could be auto casted
    auto data_type = at::autocast::is_enabled() && A.scalar_type() == at::kFloat ? at::kHalf : A.scalar_type();

    // Ensure the tensors are contiguous
    A = A.contiguous().to(data_type);
    B = B.contiguous().to(data_type);

    // Create the output tensor
    torch::Tensor output = torch::zeros({N, M}, torch::TensorOptions().dtype(data_type).device(A.device()));

    // https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)A.get_device()};

    // Call the CUDA kernel. The kernel is templated to handle different data types such as half, float, double.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "forward_cuda", ([&] {
        forward<scalar_t>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, d);
    }));

    // Device syc for debugging
    cudaDeviceSynchronize();

    return output;
}


TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, Autocast, m) {
    m.impl("float32", DotProduct<float>);
    m.impl("float64", DotProduct<double>);
    m.impl("float16", DotProduct<at::Half>);
    try {
        m.impl("bfloat16", DotProduct<at::BFloat16>);
    } catch (const std::exception& e) {
        std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float32", &DotProduct<float>);
    m.def("float64", &DotProduct<double>);
    m.def("float16", &DotProduct<at::Half>);
    try {
        m.def("bfloat16", &DotProduct<at::BFloat16>);
    } catch (const std::exception& e) {
        std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
    }
}