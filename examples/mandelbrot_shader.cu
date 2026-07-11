// mandelbrot_shader.cu - Classic GPU Compute Workload
// Computes a Mandelbrot fractal pixel-by-pixel using 2D thread indexing.
// If your compiler can successfully compile and run this mathematically intensive
// "pixel shader" equivalent, it can handle standard graphics/compute pipelines!

__global__ void mandelbrot_shader(int width, int height, float zoom, float moveX, float moveY, int maxIter, int *output) {
    // 2D Thread Indexing (Standard for image/shader processing)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Map pixel coordinates to the complex plane
        float pr = 1.5f * (x - width / 2.0f) / (0.5f * zoom * width) + moveX;
        float pi = (y - height / 2.0f) / (0.5f * zoom * height) + moveY;
        float newRe = 0.0f;
        float newIm = 0.0f;
        float oldRe, oldIm;
        
        int iter;
        for (iter = 0; iter < maxIter; iter++) {
            oldRe = newRe;
            oldIm = newIm;
            // z = z^2 + c
            newRe = oldRe * oldRe - oldIm * oldIm + pr;
            newIm = 2.0f * oldRe * oldIm + pi;
            
            // If the complex magnitude escapes the radius of 2, break
            if ((newRe * newRe + newIm * newIm) > 4.0f) {
                break;
            }
        }
        
        // Write the iteration count to the output buffer (represents the pixel's color/intensity)
        int pixelIndex = y * width + x;
        output[pixelIndex] = iter;
    }
}
