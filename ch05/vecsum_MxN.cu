#include <cstdio>

#define N 200

__global__ void add(int* a, int* b, int* c)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i*i;
    }

    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));

    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    add<<<(N+127)/128, 128>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}

