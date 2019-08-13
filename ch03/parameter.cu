#include <iostream>
#include "../common/book.h"

__global__ void add(int a, int b, int* c)
{
    *c = a + b;
}

int main()
{
    int c;
    int* device_c;
    cudaMalloc((void**)&device_c, sizeof(int));
    add<<<1, 1>>>(2, 8, device_c);
    cudaMemcpy(&c, device_c, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "2+8=" << c << std::endl;
    cudaFree(device_c);

    return 0;
}

