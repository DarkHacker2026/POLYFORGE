// vectorAdd.cu — POLYFORGE demo kernel
// Vector addition: C[i] = A[i] + B[i]
// Classic CUDA "hello world" — runs end-to-end through the POLYFORGE pipeline.

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}