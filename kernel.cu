#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// STB Image Library
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<int>(err), cudaGetErrorName(err), func);
        exit(EXIT_FAILURE);
    }
}

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

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelBytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, kernel, kernelBytes, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    convolutionKernel << <gridSize, blockSize >> > (
        d_input, d_output, width, height, channels, d_kernel, kernelSize
        );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
}

// Define common convolution kernels
const float SOBEL_X[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

const float SOBEL_Y[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

const float LAPLACIAN[9] = {
     0,  1,  0,
     1, -4,  1,
     0,  1,  0
};

const float mine[9] = {
    0, 0, 0,
    0, 0, 1,
    0, 0, 0

};

int main() {
    const char* inputFilename = "input.bmp";  // Change this to your input image file
    int width, height, channels;
    unsigned char* inputImage = stbi_load(inputFilename, &width, &height, &channels, 0);

    if (!inputImage) {
        fprintf(stderr, "Error loading image %s\n", inputFilename);
        return -1;
    }

    printf("Loaded image: %dx%d with %d channels\n", width, height, channels);

    unsigned char* outputImage = new unsigned char[width * height * channels];

    // Apply Sobel X filter
    convolveImage(inputImage, outputImage, width, height, channels, SOBEL_X, 3);
    stbi_write_png("output_sobel_x.png", width, height, channels, outputImage, width * channels);

    // Apply Sobel Y filter
    convolveImage(inputImage, outputImage, width, height, channels, SOBEL_Y, 3);
    stbi_write_png("output_sobel_y.png", width, height, channels, outputImage, width * channels);

    // Apply Laplacian filter
    convolveImage(inputImage, outputImage, width, height, channels, LAPLACIAN, 3);
    stbi_write_png("output_laplacian.png", width, height, channels, outputImage, width * channels);

    convolveImage(inputImage, outputImage, width, height, channels, mine, 3);
    stbi_write_png("output_mine.png", width, height, channels, outputImage, width * channels);

    // Clean up
    stbi_image_free(inputImage);
    delete[] outputImage;

    printf("Image processing complete. Check the output files.\n");

    return 0;
 }