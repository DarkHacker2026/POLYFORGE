// saxpy_demo.cu — POLYFORGE demo kernel
// SAXPY: Single-precision A·X Plus Y
// y[i] = a * x[i] + y[i]
// Classic GPU "hello world" — runs end-to-end through the POLYFORGE pipeline.

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
