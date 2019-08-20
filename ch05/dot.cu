#include <cstdio>

#define N 100000
#define blocksPerGrid 256
#define threadsPerBlock 128

__global__ void dot(float* a, float* b, float* partial_c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    float temp = 0;
    while (tid < N)
    {
        temp += (a[tid] + b[tid]);
        tid += blockDim.x*gridDim.x;
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    // Reduction
    int stride = blockDim.x/2;
    
    while (stride != 0)
    {
        if (threadIdx.x < stride)
        {
            cache[threadIdx.x] += cache[threadIdx.x+stride];
        }
        __syncthreads();
        stride /= 2;
    }

    if (threadIdx.x == 0)
    {
        partial_c[blockIdx.x] = cache[0];
    }
}

int main()
{
    float *a, *b, *partial_c;
    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i*2;
    }

    float *dev_a, *dev_b, *dev_partial_c;
    cudaMalloc((void**)&dev_a, N*sizeof(float));
    cudaMalloc((void**)&dev_b, N*sizeof(float));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));

    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    float c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

    printf("dotmul result: %.2f\n", c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    free(a);
    free(b);
    free(partial_c);

    return 0;
}

