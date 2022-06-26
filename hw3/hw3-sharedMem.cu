#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <chrono>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define MASK_LEN 5
#define SD_WIDTH 260 // numOfThreads + MASK_LEN - 1
#define SCALE 8
#define xBound 2
#define yBound 2

// clang-format off
__constant__ char mask[MASK_N][MASK_X][MASK_Y] = {
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0},
     {  2,  8, 12,  8,  2},
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1},
     { -4, -8,  0,  8,  4},
     { -6,-12,  0, 12,  6},
     { -4, -8,  0,  8,  4},
     { -1, -2,  0,  2,  1}}
};
// clang-format on

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
    unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
    const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    // int x, y, i, v, u;
    // int R, G, B;
    float val[MASK_N][3] = {0.0};
    // float val[MASK_N * 3] = {0.0};
    // int adjustX, adjustY, xBound, yBound;

    // thread idx
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width) return ;

    // block shared memory
    __shared__ unsigned char sharedR[MASK_LEN][SD_WIDTH];
    __shared__ unsigned char sharedG[MASK_LEN][SD_WIDTH];
    __shared__ unsigned char sharedB[MASK_LEN][SD_WIDTH];

    int sIdx;

    for (int y = 0; y < height; ++y) {
        // move data from global memory to per-block memory
        // height = 0, we just need to get the first 3 rows
        if (y == 0) {
            for (int v = 0; v<= yBound; ++v) {
                if (y + v >= 0 && y + v < height) {
                    // thread's responsible row
                    sIdx = channels * (width * (y + v) + x);
                    sharedR[v + yBound][threadIdx.x + xBound] = s[sIdx + 2];
                    sharedG[v + yBound][threadIdx.x + xBound] = s[sIdx + 1];
                    sharedB[v + yBound][threadIdx.x + xBound] = s[sIdx + 0];

                    // boundary case (leftmost & rightmost)
                    // leftmost
                    if (threadIdx.x == 0) {
                        if (blockIdx.x != 0) {
                            // x - 2
                            sIdx = channels * (width * (y + v) + (x - 2));
                            sharedR[v + yBound][0] = s[sIdx + 2];
                            sharedG[v + yBound][0] = s[sIdx + 1];
                            sharedB[v + yBound][0] = s[sIdx + 0];
                            
                            // x - 1
                            sIdx = channels * (width * (y + v) + (x - 1));
                            sharedR[v + yBound][1] = s[sIdx + 2];
                            sharedG[v + yBound][1] = s[sIdx + 1];
                            sharedB[v + yBound][1] = s[sIdx + 0];
                        }
                    }

                    // rightmost 
                    if (threadIdx.x == blockDim.x - 1) {
                        // x + 2
                        if (x + 2 < width) {
                            sIdx = channels * (width * (y + v) + (x + 2));
                            sharedR[v + yBound][SD_WIDTH - 1] = s[sIdx + 2];
                            sharedG[v + yBound][SD_WIDTH - 1] = s[sIdx + 1];
                            sharedB[v + yBound][SD_WIDTH - 1] = s[sIdx + 0];
                        }

                        // x + 1
                        if (x + 1 < width) {
                            sIdx = channels * (width * (y + v) + (x + 1));
                            sharedR[v + yBound][SD_WIDTH - 2] = s[sIdx + 2];
                            sharedG[v + yBound][SD_WIDTH - 2] = s[sIdx + 1];
                            sharedB[v + yBound][SD_WIDTH - 2] = s[sIdx + 0];
                        }
                    }
                }
            }
        } else {
            // remove the first row of sharedRGB by moving up each row
            for (int i=1; i<=4; ++i) {
                // thread's responsible row
                sharedR[i-1][threadIdx.x + xBound] = sharedR[i][threadIdx.x + xBound];
                sharedG[i-1][threadIdx.x + xBound] = sharedG[i][threadIdx.x + xBound];
                sharedB[i-1][threadIdx.x + xBound] = sharedB[i][threadIdx.x + xBound];

                // boundary case (leftmost & rightmost)
                // leftmost
                if (threadIdx.x == 0) {
                    for (int left = 0; left <= 1; ++left) {
                        sharedR[i-1][left] = sharedR[i][left];
                        sharedG[i-1][left] = sharedG[i][left];
                        sharedB[i-1][left] = sharedB[i][left];
                    }
                }

                // rightmost
                if (threadIdx.x == blockDim.x - 1) {
                    for (int right = SD_WIDTH-1; right >= SD_WIDTH-2; --right) {
                        sharedR[i-1][right] = sharedR[i][right];
                        sharedG[i-1][right] = sharedG[i][right];
                        sharedB[i-1][right] = sharedB[i][right];
                    }
                }
            }

            // add the new height to last row of sharedRGB
            if (y + yBound >= 0 && y + yBound < height) {
                // thread's responsible row
                sIdx = channels * (width * (y + yBound) + x);
                sharedR[MASK_LEN-1][threadIdx.x + xBound] = s[sIdx + 2];
                sharedG[MASK_LEN-1][threadIdx.x + xBound] = s[sIdx + 1];
                sharedB[MASK_LEN-1][threadIdx.x + xBound] = s[sIdx + 0];

                // boundary case (leftmost & rightmost)
                // leftmost
                if (threadIdx.x == 0) {
                    if (blockIdx.x != 0) {
                        // x - 2
                        sIdx = channels * (width * (y + yBound) + (x - 2));
                        sharedR[MASK_LEN-1][0] = s[sIdx + 2];
                        sharedG[MASK_LEN-1][0] = s[sIdx + 1];
                        sharedB[MASK_LEN-1][0] = s[sIdx + 0];
                        
                        // x - 1
                        sIdx = channels * (width * (y + yBound) + (x - 1));
                        sharedR[MASK_LEN-1][1] = s[sIdx + 2];
                        sharedG[MASK_LEN-1][1] = s[sIdx + 1];
                        sharedB[MASK_LEN-1][1] = s[sIdx + 0];
                    }
                }

                // rightmost 
                if (threadIdx.x == blockDim.x - 1) {
                    // x + 2
                    if (x + 2 < width) {
                        sIdx = channels * (width * (y + yBound) + (x + 2));
                        sharedR[MASK_LEN-1][SD_WIDTH - 1] = s[sIdx + 2];
                        sharedG[MASK_LEN-1][SD_WIDTH - 1] = s[sIdx + 1];
                        sharedB[MASK_LEN-1][SD_WIDTH - 1] = s[sIdx + 0];
                    }

                    // x + 1
                    if (x + 1 < width) {
                        sIdx = channels * (width * (y + yBound) + (x + 1));
                        sharedR[MASK_LEN-1][SD_WIDTH - 2] = s[sIdx + 2];
                        sharedG[MASK_LEN-1][SD_WIDTH - 2] = s[sIdx + 1];
                        sharedB[MASK_LEN-1][SD_WIDTH - 2] = s[sIdx + 0];
                    }
                }
            }
        }

        // sync all the threads in the block
        __syncthreads();

        // masking
        for (int i = 0; i < MASK_N; ++i) {
            val[i][2] = 0.0;
            val[i][1] = 0.0;
            val[i][0] = 0.0;

            for (int v = -yBound; v <= yBound; ++v) {
                for (int u = -xBound; u <= xBound; ++u) {
                    if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                        unsigned char R = sharedR[v + yBound][threadIdx.x + xBound + u];
                        unsigned char G = sharedG[v + yBound][threadIdx.x + xBound + u];
                        unsigned char B = sharedB[v + yBound][threadIdx.x + xBound + u];
                        val[i][2] += R * mask[i][u + xBound][v + yBound];
                        val[i][1] += G * mask[i][u + xBound][v + yBound];
                        val[i][0] += B * mask[i][u + xBound][v + yBound];
                    }
                }
            }
        }

        // same as template code 
        float totalR = 0.0;
        float totalG = 0.0;
        float totalB = 0.0;
        for (int i = 0; i < MASK_N; ++i) {
            totalR += val[i][2] * val[i][2];
            totalG += val[i][1] * val[i][1];
            totalB += val[i][0] * val[i][0];
        }

        // same as template code
        totalR = sqrt(totalR) / SCALE;
        totalG = sqrt(totalG) / SCALE;
        totalB = sqrt(totalB) / SCALE;
        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
        t[channels * (width * y + x) + 2] = cR;
        t[channels * (width * y + x) + 1] = cG;
        t[channels * (width * y + x) + 0] = cB;

        // sync all the threads in the block
        __syncthreads();
    } 
}

int main(int argc, char** argv) {
    assert(argc == 3);

    unsigned height, width, channels;
    unsigned char* src_img = NULL;

    // cuda
    unsigned char *dsrc_img, *ddst_img;

    read_png(argv[1], &src_img, &height, &width, &channels);
    assert(channels == 3);

    unsigned char* dst_img =
        (unsigned char*)malloc(height * width * channels * sizeof(unsigned char));


    // malloc for GPU device
    cudaMalloc(&dsrc_img, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst_img, height * width * channels * sizeof(unsigned char));

    // mem copy src image to GPU 
    cudaMemcpy(dsrc_img, src_img, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    const int numOfThreads = 256;
    const int numOfBlocks = (numOfThreads + width - 1) / numOfThreads;

    // cudaMallocManaged
    // cudaMallocManaged(&ddst_img, height * width * channels * sizeof(unsigned char));

    sobel<<<numOfBlocks, numOfThreads>>>(dsrc_img, ddst_img, height, width, channels);

    // mem copy dst image back to host
    cudaMemcpy(dst_img, ddst_img, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    write_png(argv[2], dst_img, height, width, channels);

    free(src_img);
    free(dst_img);
    cudaFree(dsrc_img);
    cudaFree(ddst_img);

    return 0;
}
