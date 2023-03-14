#include <stdint.h>
#include "timer.hpp"

void printGpuProperties();

namespace GpuSeedTable {
struct DeviceArrays {
    uint32_t* d_compressedSeq;
    uint32_t d_seqLen;
    size_t* d_array1;
    size_t* d_array2;
    size_t* d_array3;
    size_t* d_array4;
    size_t* d_SA;
    size_t* d_done;
    // HINT: if needed, you add more device arrays for the GPU here (make sure to allocate and dellocate them in appropriate functions!)

    void allocateDeviceArrays (uint32_t* compressedSeq, uint32_t seqLen, uint32_t kmerSize);
    void printValues(uint32_t numValues,uint32_t kmerSize);
    void deallocateDeviceArrays ();
};

static DeviceArrays deviceArrays;

void seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    size_t* array1,
    size_t* array2,
    size_t* array3,
    size_t* array4,
    size_t* SA,
    size_t* done);
}

