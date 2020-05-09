#include "header.cuh"
#define N 16384
int a[N], b[N];

int main() {

    for (int i = 0; i < N; ++i) a[i] = 1;
    int *d_a, *d_b;
    cudaMalloc((int **)&d_a, N * 4);
    cudaMalloc((int **)&d_b, N * 4);
    cudaMemcpy(d_a, a, N * 4, cudaMemcpyHostToDevice);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_a, d_b, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_a, d_b, N);
    cudaMemcpy(b, d_b, N * 4, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 10; ++i) {
    //     printf("# %d %d\n", N * i / 10, b[N * i / 10]);
    // }
    for (int i = 0; i < N; ++i) {
        if (b[i] != i + 1) {
            printf("# break point is %d\n", i);
            break;
        }
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_temp_storage);
    return 0;
}