#include <cmath>
#include "../common/cpu_bitmap.h"

#define INF 2e10f
#define DIM 1024
#define NSPHERES 20

struct Sphere
{
    float x, y, z;
    float r, g, b;
    float radius;

    __device__ float hit(int ox, int oy, float* n)
    {
        float dx = ox - x;
        float dy = oy - y;

        if (dx*dx + dy*dy < radius*radius)
        {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz/radius;
            return dz + z;
        }
        return -INF;
    }
};

__global__ void kernel(Sphere* spheres, unsigned char* bitmap)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int offset = x + y*blockDim.x*gridDim.x;
    float ox = x - DIM/2;
    float oy = y - DIM/2;

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < NSPHERES; i++)
    {
        float n;
        float z = spheres[i].hit(ox, oy, &n);
        if (z > maxz)
        {
            r = spheres[i].r*n;
            g = spheres[i].g*n;
            b = spheres[i].b*n;
            maxz = z;
        }
    }

    bitmap[offset*4 + 0] = (int)(r*255);
    bitmap[offset*4 + 1] = (int)(g*255);
    bitmap[offset*4 + 2] = (int)(b*255);
    bitmap[offset*4 + 3] = 255;
}

#define rnd(x) (x*rand() / RAND_MAX)

int main()
{
    CPUBitmap bitmap(DIM, DIM);

    unsigned char* dev_bitmap;
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

    Sphere* dev_spheres;
    cudaMalloc((void**)&dev_spheres, NSPHERES*sizeof(Sphere));

    Sphere* spheres = (Sphere*)malloc(NSPHERES*sizeof(Sphere));
    for (int i = 0; i < NSPHERES; i++)
    {
        spheres[i].r = rnd(1.0f);
        spheres[i].g = rnd(1.0f);
        spheres[i].b = rnd(1.0f);
        spheres[i].x = rnd(1000.0f) - 500;
        spheres[i].y = rnd(1000.0f) - 500;
        spheres[i].z = rnd(1000.0f) - 500;
        spheres[i].radius = rnd(100.0f) + 20;
    }
    cudaMemcpy(dev_spheres, spheres, NSPHERES*sizeof(Sphere), cudaMemcpyHostToDevice);

    dim3 gridDim(DIM/16, DIM/16);
    dim3 blockDim(16, 16);

    kernel<<<gridDim, blockDim>>>(dev_spheres, dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    bitmap.display_and_exit();

    free(spheres);
    cudaFree(dev_spheres);
    cudaFree(dev_bitmap);

    return 0;
}

