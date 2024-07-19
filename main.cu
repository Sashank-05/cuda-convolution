#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void convolutionKernel(const unsigned char* input, unsigned char* output,
    int width, int height, int channels,
    const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            int halfKernel = kernelSize / 2;

            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;

                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        float kernelValue = kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                        sum += input[(iy * width + ix) * channels + c] * kernelValue;
                    }
                }
            }

            int clamped = __float2int_rd(sum);
            clamped = max(0, min(255, clamped));
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(clamped);
        }
    }
}

void convolveImage(const unsigned char* input, unsigned char* output,
    int width, int height, int channels,
    const float* kernel, int kernelSize) {
    unsigned char* d_input, * d_output;
    float* d_kernel;

    size_t imageSize = width * height * channels * sizeof(unsigned char);
    size_t kernelBytes = kernelSize * kernelSize * sizeof(float);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelBytes);

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    convolutionKernel<<<gridSize, blockSize>>>(
        d_input, d_output, width, height, channels, d_kernel, kernelSize
    );

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}