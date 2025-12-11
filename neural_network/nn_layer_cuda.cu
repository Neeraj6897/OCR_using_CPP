#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "nn_layer.h"

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(code) + " at " + file + ":" + std::to_string(line));
}
#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

__global__ void dense_forward_kernel(
    const float* __restrict__ W,
    const float* __restrict__ b,
    const float* __restrict__ x,
    float* __restrict__ y,      
    int in_size, int out_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_size) return;

    float sum = b[i];
    const float* wrow = W + i * in_size;
    for (int j = 0; j < in_size; ++j) {
        sum += wrow[j] * x[j];
    }
    y[i] = sum;
}

__global__ void k_grad_weights(const float* __restrict__ dY,
                               const float* __restrict__ x,
                               float* __restrict__ dW,
                               int out_size, int in_size)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size && j < in_size) {
        dW[i * in_size + j] = dY[i] * x[j];
    }
}

__global__ void k_grad_bias(const float* __restrict__ dY,
                            float* __restrict__ db,
                            int out_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size) {
        db[i] = dY[i];
    }
}

__global__ void k_grad_input(const float* __restrict__ W,  
                             const float* __restrict__ dY, 
                             float* __restrict__ dX,       
                             int out_size, int in_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= in_size) return;

    float sum = 0.0f;

    for (int i = 0; i < out_size; ++i) {
        sum += dY[i] * W[i * in_size + j];
    }
    dX[j] = sum;
}

__global__ void k_update_weights(float* __restrict__ W,
                                 const float* __restrict__ dW,
                                 float lr, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        W[k] -= lr * dW[k];
    }
}

__global__ void k_update_biases(float* __restrict__ b,
                                const float* __restrict__ db,
                                float lr, int out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out) {
        b[i] -= lr * db[i];
    }
}

#ifdef USE_CUDA
void NN_Layer::cudaInit_() {
    CUDA_CHECK(cudaMalloc(&d_W_, output_size_ * input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_, output_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_, input_size_  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_, output_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dY_, output_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW_, output_size_ * input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_, output_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dX_, input_size_  * sizeof(float)));
}

void NN_Layer::cudaFree_() {
    if (d_W_) cudaFree(d_W_);
    if (d_b_) cudaFree(d_b_);
    if (d_x_) cudaFree(d_x_);
    if (d_y_) cudaFree(d_y_);
    d_W_ = d_b_ = d_x_ = d_y_ = nullptr;
    if (d_dY_) cudaFree(d_dY_);
    if (d_dW_) cudaFree(d_dW_);
    if (d_db_) cudaFree(d_db_);
    if (d_dX_) cudaFree(d_dX_);
    d_dY_ = d_dW_ = d_db_ = d_dX_ = nullptr;
}

void NN_Layer::cudaUploadParams_() {
    CUDA_CHECK(cudaMemcpy(d_W_, weights_.data(),
                          weights_.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_, biases_.data(),
                          biases_.size()*sizeof(float), cudaMemcpyHostToDevice));
}

void NN_Layer::cudaForward(const float* x_host, float* y_host) const {
    CUDA_CHECK(cudaMemcpy(d_x_, x_host, input_size_*sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (output_size_ + threads - 1) / threads;
    dense_forward_kernel<<<blocks, threads>>>(d_W_, d_b_, d_x_, d_y_, input_size_, output_size_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(y_host, d_y_, output_size_*sizeof(float), cudaMemcpyDeviceToHost));
}

void NN_Layer::cudaBackward(const float* gradY_host,
                             float* gradX_host)
{
    CUDA_CHECK(cudaMemcpy(d_dY_, gradY_host,
                          output_size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block2d(16, 16);
    dim3 grid2d((input_size_  + block2d.x - 1) / block2d.x,
                (output_size_ + block2d.y - 1) / block2d.y);
    k_grad_weights<<<grid2d, block2d>>>(d_dY_, d_x_, d_dW_, output_size_, input_size_);
    CUDA_CHECK(cudaGetLastError());

    int threads = 256;
    int blocks_b = (output_size_ + threads - 1) / threads;
    k_grad_bias<<<blocks_b, threads>>>(d_dY_, d_db_, output_size_);
    CUDA_CHECK(cudaGetLastError());

    int blocks_x = (input_size_ + threads - 1) / threads;
    k_grad_input<<<blocks_x, threads>>>(d_W_, d_dY_, d_dX_,
                                        output_size_, input_size_);
    CUDA_CHECK(cudaGetLastError());

    if (!update_on_gpu_) {
        CUDA_CHECK(cudaMemcpy(gradient_weights_.data(), d_dW_,
                            gradient_weights_.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gradient_biases_.data(), d_db_,
                            gradient_biases_.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(gradX_host, d_dX_,
                          input_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void NN_Layer::cudaUpdate(float lr)
{
    int n_w = output_size_ * input_size_;
    int threads = 256;
    int blocks_w = (n_w + threads - 1) / threads;
    k_update_weights<<<blocks_w, threads>>>(d_W_, d_dW_, lr, n_w);
    CUDA_CHECK(cudaGetLastError());

    int blocks_b = (output_size_ + threads - 1) / threads;
    k_update_biases<<<blocks_b, threads>>>(d_b_, d_db_, lr, output_size_);
    CUDA_CHECK(cudaGetLastError());

    host_params_dirty_ = true;
}

void NN_Layer::syncDeviceToHost_() {
  if (!host_params_dirty_) return;
  CUDA_CHECK(cudaMemcpy(weights_.data(), d_W_,
                        weights_.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(biases_.data(), d_b_,
                        biases_.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  host_params_dirty_ = false;
}

#endif