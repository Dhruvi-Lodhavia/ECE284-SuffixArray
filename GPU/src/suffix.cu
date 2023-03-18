#include "suffix.cuh"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <math.h>

/**
 * Prints information for each available GPU device on stdout
 */
void printGpuProperties () {
    int nDevices;

    // Store the number of available GPU device in nDevicess
    cudaError_t err = cudaGetDeviceCount(&nDevices);

    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaGetDeviceCount failed!\n");
        exit(1);
    }

    // For each GPU device found, print the information (memory, bandwidth etc.)
    // about the device
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device memory: %lu\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

/**
 * Allocates arrays on the GPU device for (i) storing the compressed sequence
  (ii) storing B, B2,suffix array values and other intermediate arrays for operations
 * Size of the arrays depends on the input sequence length and kmer size
 */
void Gpusuffix::DeviceArrays::allocateDeviceArrays (uint32_t* compressedSeq, uint32_t seqLen, uint32_t kmerSize) {
    cudaError_t err;

    d_seqLen = seqLen;
    uint32_t compressedSeqLen = (seqLen+15)/16;
    uint32_t maxKmers = (uint32_t) pow(4,kmerSize)+1;
    

    // Only (1)allocate and (2)transfer the 2-bit compressed sequence to GPU.
    // This reduces the memory transfer and storage overheads
    // 1. Allocate memory
    err = cudaMalloc(&d_compressedSeq, compressedSeqLen*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // 2. Transfer compressed sequence
    err = cudaMemcpy(d_compressedSeq, compressedSeq, compressedSeqLen*sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

  
    err = cudaMalloc(&d_array2, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }


    err = cudaMalloc(&d_intermediate_array, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    
    err = cudaMalloc(&d_intermediate_array2, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // Allocate memory on GPU device for storing the kmer position array
    // Each element is size_t (64-bit) because an intermediate step uses the
    // first 32-bits for kmer value and the last 32-bits for kmer positions
    err = cudaMalloc(&d_array1, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }


    err = cudaMalloc(&d_array3, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_suffix_array, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_done, (1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    cudaDeviceSynchronize();
}

/**
 * Free allocated GPU device memory for different arrays
 */
void Gpusuffix::DeviceArrays::deallocateDeviceArrays () {
    cudaFree(d_compressedSeq);
    cudaFree(d_array2);
    cudaFree(d_intermediate_array);
    cudaFree(d_intermediate_array2);
    cudaFree(d_array1);
    cudaFree(d_array3);
    cudaFree(d_suffix_array);
    cudaFree(d_done);
}

/**
 * Finds kmers for the compressed sequence creates an array with elements
 * containing the 64-bit concatenated value consisting of the kmer value in the
 * first 32 bits and the kmer position in the last 32 bits. The values are
 * stored in the arrary d_array1 with i-th element corresponding to the i-th
 * kmer in the sequence
 *
 * ASSIGNMENT 2 TASK: parallelize this function
 */
__global__ void kmerPosConcat(
    uint32_t* d_compressedSeq,
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_array1) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    
    int bs = blockDim.x;
    int gs = gridDim.x;

    int i = bs*bx+tx;
    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    // Helps mask the non kmer bits from compressed sequence. E.g. for k=2,
    // mask=0x1111 and for k=3, mask=0x111111
    uint32_t mask = (1 << 2*k)-1;
    size_t kmer = 0;
    
    
    while(i<=N-k){
        uint32_t index = i/16;
        uint32_t shift1 = 2*(i%16);
        if (shift1 > 0) {
            uint32_t shift2 = 32-shift1;
            kmer = ((d_compressedSeq[index] >> shift1) | (d_compressedSeq[index+1] << shift2)) & mask;
        } else {
            kmer = d_compressedSeq[index] & mask;
        }

        // Concatenate kmer value (first 32-bits) with its position (last
        // 32-bits)
        size_t arrayConcat;
        
        arrayConcat = (kmer << 32) + i;
        
        d_array1[i] = arrayConcat;
        i+=bs*gs;
    }
    
}

/**
 * This is the first part of rebucketing stage where threads at index i and i+1
 * compare and give i if they are unequal and 0 if they are equal
 */
__global__ void kmerOffsetFill(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* d_array2,
    size_t* d_array1) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;
    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;
   
    size_t mask = ((size_t) 1 << 32)-1;
    uint32_t kmer = 0;
    uint32_t lastKmer = 0;

     
    for (uint32_t i = (bx * bs + tx); i < N-k; i+=bs*gs){
            lastKmer = (d_array1[i] >> 32) & mask;
            kmer = (d_array1[i+1] >> 32) & mask;
        
        if(kmer == lastKmer){
            d_array2[i+1] = 0;
        }
        else{
            d_array2[i+1] = i+1;
        }   
    }
}

//This kernel computes prefix scan with max by using shared memory. it calculates 
//per block prefix scan
__global__ void prefixsum(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* d_array2,
    size_t* d_array3,
    uint32_t range) {

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;
    

    for(uint32_t index = bx; index < ((range+bs-1)/bs); index+=gs){ 
        
        __shared__ size_t array_shared[2048]; 
        //transfer contents from global to shared
        uint32_t startAddress = index*(bs);
        if((startAddress+tx) < range){
            array_shared[tx] = d_array2[startAddress + tx];
        }
        else{
            array_shared[tx] = 0;
        }
        __syncthreads();
        int n = bs;


        //use upsweep and downsweep for computing prefix scan
        for (int offset=1; offset<n; offset*=2) {
            int val = (tx + 1) * offset * 2 - 1;
            if (val< n) {
                array_shared[val] = max(array_shared[val - offset], array_shared[val]);
            }
            __syncthreads();
        }

        
        for (int offset=n/2; offset>0; offset>>= 1) {
            __syncthreads();
            int val = (tx + 1) * offset * 2 - 1;
            if (val < n) {
                array_shared[val+offset] = max(array_shared[val],array_shared[val+offset]);
            }
        }
        __syncthreads();
        //store back to global
        if((startAddress+ tx) < N-k+1){
            d_array2[startAddress + tx] = array_shared[tx];
        }
        //store max value of each block in another array
        d_array3[index] = array_shared[n-1];
    }
}

//in this step we compute max of previous block group with present block group
__global__ void reductionStep(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* d_array2,
    size_t* d_array3,
    uint32_t range) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;
    for(uint32_t index = bx; index< (range/bs)+1; index+=gs){ 
        uint32_t startAddress = index*(bs);
        if(((startAddress+tx) < range) && (index!=0)){
            d_array2[startAddress + tx] = max(d_array3[index-1],d_array2[startAddress + tx]);
        }
        else{
            d_array2[startAddress + tx] += 0;
        }        
    }
}

    /**
    * Masks the first 32 bits of the elements in the array1 array
    *
    */
__global__ void kmerPosMask(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_array1) {

    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    int i = bs*bx+tx;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    size_t mask = ((size_t) 1 << 32)-1;
    
    while(i<=N-k){
        d_array1[i] = (d_array1[i] & mask);
        i+=bs*gs;
        }
}

//SA to ISA conversion using map parallelism
__global__ void reordering(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* d_array2,,
    size_t* d_array1,
    size_t* d_array3) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;
    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;
    
    for (uint32_t i = (bx * bs + tx); i <= N-k; i+=bs*gs){
        uint32_t new_index = d_array1[i];
        d_array3[new_index] = d_array2[i];
    } 
}

//shifting by an offset shift_val using map parallelism
//assigning value i at index i in suffix array for sorting later
__global__ void shifting(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* d_array1,
    size_t* d_array3,
    uint32_t shift_val,
    size_t* d_suffix_array) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    for (uint32_t i = (bx * bs + tx); i <= N-k; i+=bs*gs){
        d_suffix_array[i] = i;
        if(i<=N-k-shift_val){
            d_array1[i] = d_array3[i+shift_val];
        }
        else{
            d_array1[i] = 0;
        }
    } 
}

//merging array B and B2 so they can be sorted later
__global__ void merging(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* d_array1,
    size_t* d_array3) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;
    for (uint32_t i = (bx * bs + tx); i <= N-k; i+=bs*gs){
        d_array1[i] += d_array3[i]<<32;
    } 
}

//performs bucketing operation just like kmeroffsetfill
//there we performed bucketing on lower 32 bits
//here we do it on all 64 bits
__global__ void kmerOffsetFill2(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* d_array2,
    size_t* d_array1) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;
   
    size_t kmer = 0;
    size_t lastKmer = 0;

    

    
    for (uint32_t i = (bx * bs + tx); i < N-k; i+=bs*gs){
        lastKmer = d_array1[i];
        kmer = d_array1[i+1];
        
        if(kmer == lastKmer){
            d_array2[i+1] = 0;
        }
        else{
            d_array2[i+1] = i+1;
        }   
    }   
}

//checks if all values are unique 
__global__ void singleton(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* d_array2,
    size_t* d_done){

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    uint32_t kmer = 0;
    uint32_t lastKmer = 0;
    //initialize done to be 0 initially
    if((bx==0) && (tx==0))
    {
        d_done[0] = 1;
    }
    //if consecutive values are same. break and done =0
    //many threads can compete and try to write done = 0
    //but does not matter as it can only overwrite 0
    for (uint32_t i = (bx * bs + tx); i < N-k; i+=bs*gs){
        lastKmer = d_array2[i];
        kmer = d_array2[i+1];
        if(kmer == lastKmer){
            d_done[0] = 0;
            return;
        }
        
    }

}



/**
 * Constructs suffix array on gpu
*/
void Gpusuffix::suffixOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    size_t* array2,
    size_t* intermediate_array,
    size_t* intermediate_array2,
    size_t* array1,
    size_t* array3,
    size_t* suffix_array,
    size_t* done) {


    int numBlocks =  65535; // i.e. number of thread blocks on the GPU
    int blockSize = 1024; // i.e. number of GPU threads per thread block



    kmerPosConcat<<<numBlocks, blockSize>>>(compressedSeq, seqLen, kmerSize, suffix_array);

    // Parallel sort the suffixrray array on the GPU device using the thrust
    // library (https://thrust.github.io/)
    // thrust::device_ptr<size_t> array1Ptr(array1);
    thrust::device_ptr<size_t> interPtr(suffix_array);
    thrust::sort(interPtr, interPtr+seqLen-kmerSize+1);
    uint32_t numKmers = pow(4, kmerSize);
    uint32_t range = (seqLen-kmerSize+1);

    //rebucketing start
    kmerOffsetFill<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array2,suffix_array);
    prefixsum<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array2,array3,range);
    uint32_t num = ((range+blockSize-1)/blockSize);
    prefixsum<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array3,intermediate_array,num);
    uint32_t num2 = ((num+blockSize-1)/blockSize);
    prefixsum<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, intermediate_array,intermediate_array2,num2);
    reductionStep<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, intermediate_array,intermediate_array2,((range/blockSize)/blockSize));
    reductionStep<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array3,intermediate_array,(range/blockSize));
    reductionStep<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array2,array3,range);
    kmerPosMask<<<numBlocks, blockSize>>>(seqLen, kmerSize, suffix_array);

    size_t* done2 = new size_t[1];
    uint32_t shift_val = 1;
    //loop repeats till done = 1
    do{ 

        reordering<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array2,suffix_array,array3);
        shifting<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers,array1,array3,shift_val,suffix_array);
        shift_val = shift_val<<1;
        merging<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers,array1,array3);

        thrust::device_ptr<size_t> array1Ptr(array1);
        thrust::device_ptr<size_t> suffixPtr(suffix_array);
        thrust::sort_by_key(array1Ptr, array1Ptr+seqLen-kmerSize+1,suffixPtr);

        kmerOffsetFill2<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array2,array1);
        prefixsum<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array2,array3,range);

        prefixsum<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array3,intermediate_array,num);
        prefixsum<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, intermediate_array,intermediate_array2,num2);
        reductionStep<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, intermediate_array,intermediate_array2,((range/blockSize)/blockSize));
        reductionStep<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array3,intermediate_array,(range/blockSize));
        reductionStep<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, array2,array3,range);
        
        
        singleton<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers,array2,done);
        
        cudaMemcpy(done2, done, sizeof(size_t), cudaMemcpyDeviceToHost);
    } while(done2[0] == 0);
    


    // Wait for all computation on GPU device to finish. Needed to ensure
    // correct runtime profiling results for this function.
    cudaDeviceSynchronize();
}

void Gpusuffix::DeviceArrays::printValues(uint32_t numValues,uint32_t kmerSize) {
    size_t* array2 = new size_t[numValues-kmerSize+1];
    size_t* array1 = new size_t[numValues-kmerSize+1];
    size_t* array3 = new size_t[numValues-kmerSize+1];
    size_t* intermediate_array = new size_t[numValues-kmerSize+1];
    size_t* intermediate_array2 = new size_t[numValues-kmerSize+1];
    size_t* suffix_array = new size_t[numValues-kmerSize+1];
    size_t* done = new size_t[1];
    cudaError_t err;

   
 
    err = cudaMemcpy(suffix_array, d_suffix_array, (numValues-kmerSize+1)*sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!!!!\n");
        exit(1);
    }

    //writing suffix array in a file 
    FILE *fp;
    fp = fopen("out.txt", "w");

    for (uint32_t i = 0; i < (numValues-kmerSize+1); i++) {
    	fprintf(fp, "%lu\n",suffix_array[i]);
    }
}

