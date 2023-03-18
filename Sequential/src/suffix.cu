#include "suffix.cuh"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <math.h>
// #include <lock.h>

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

    err = cudaMalloc(&d_array1, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_array2, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_array3, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_array4, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_SA, (seqLen-kmerSize+1)*sizeof(size_t));
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
    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_array3);
    cudaFree(d_array4);
    cudaFree(d_SA);
    cudaFree(d_done);
}

/**
 * Finds kmers for the compressed sequence creates an array with elements
 * containing the 64-bit concatenated value consisting of the kmer value in the
 * first 32 bits and the kmer position in the last 32 bits. The values are
 * stored in the arrary d_array1, with i-th element corresponding to the i-th
 * kmer in the sequence
 */
__global__ void kmerPosConcat(
    uint32_t* d_compressedSeq,
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_array1) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    // Helps mask the non kmer bits from compressed sequence. E.g. for k=2,
    // mask=0x1111 and for k=3, mask=0x111111
    uint32_t mask = (1 << 2*k)-1;
    size_t kmer = 0;
    
    // the if statement below ensures only the first thread of the first
    // block does all the computation. 

    if ((bx == 0) && (tx == 0)) {
    	for (uint32_t i = 0; i <= N-k; i++) {
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
        	size_t arrayConcat = (kmer << 32) + i;
        	d_array1[i] = arrayConcat;
	    }
    }

    printf("0 done\n");
}

//this performs rebucketing sequentially
__global__ void reBucketInitial(
	uint32_t d_seqLen,
	uint32_t kmerSize,
	size_t* d_array1,
	size_t* d_array2) {

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    size_t mask = ((size_t) 1 << 32)-1;
    uint32_t kmer = 0;
    uint32_t lastKmer = 0;
    uint32_t lastIndex = 0;
    d_array2[0] = 0;
    for (uint32_t i = 1; i <= N-k; i++) {
        lastKmer = (d_array1[i-1] >> 32) & mask;
        kmer = (d_array1[i] >> 32) & mask;
        if(kmer == lastKmer){
            d_array2[i] = lastIndex;
        }
        else{
            d_array2[i] = i;
            lastIndex = i;
        }
    }
}


//this reorders (SA to ISa conversion) sequentially
__global__ void reordering(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_array1,
    size_t* d_array2,
    size_t* d_array3) {

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    for (uint32_t i = 0; i <= N-k; i++){
        uint32_t new_index = d_array1[i];
        d_array3[new_index] = d_array2[i];
    }
}

//performs shifting by offset shift_val sequentially
__global__ void shifting(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_array3,
    size_t* d_array1,
    uint32_t shift_val) {

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    for (uint32_t i = 0; i <= N-k; i++){
        if(i<=N-k-shift_val){
            d_array1[i] = d_array3[i+shift_val];
        }
        else{
            d_array1[i] = 0;
        }
    }
}

//merges array B and B2
//assign indexes (i) in suffix array as i
__global__ void merge(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_array3,
    size_t* d_array1,
    size_t* d_SA,
    size_t* d_array4) {

	uint32_t N = d_seqLen;
	uint32_t k = kmerSize;

	size_t mask_R = ((size_t) 1 << 32)-1;
	size_t mask_L = mask_R << 32;

	for (uint32_t i = 0; i <= N-k; i++)
		d_array4[i] = ((d_array3[i] << 32) & mask_L) | (d_array1[i] & mask_R);
	
	
	for (uint32_t i = 0; i <= N-k; i++) {
		d_SA[i] = i;
	}

}

//performs bucketing operation just like rebucketinitial
//there we performed bucketing on lower 32 bits
//here we do it on all 64 bits
__global__ void reBucket(
	uint32_t d_seqLen,
	uint32_t kmerSize,
	size_t* d_array4,
	size_t* d_array2,
	size_t* d_array1,
	size_t* d_SA) {

		uint32_t N = d_seqLen;
		uint32_t k = kmerSize;

		for (uint32_t i = 0; i <= N-k; i++) {
			d_array1[i] = d_SA[i];
		}

		uint32_t lastIndex = 0;
		d_array2[0] = 0;
		for (uint32_t i = 1; i <= N-k; i++) {
			if(d_array4[i-1] == d_array4[i]){
            			d_array2[i] = lastIndex;
       			}
        		else{
            			d_array2[i] = i;
				lastIndex = i;
        		}
		}

}


//checks if consecutive values are different. if not, done =0
__global__ void singleton(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_array2,
    size_t* d_done) {

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    uint32_t kmer = 0;
    uint32_t lastKmer = 0;
    size_t mask = ((size_t) 1 << 32)-1;

    d_done[0] = 1;


    
    for (uint32_t i = 0; i < N-k; i++){
        lastKmer = (d_array2[i]) & mask;
        kmer = (d_array2[i+1]) & mask;
        if(kmer == lastKmer){
            d_done[0] = 0;
            break;
        }
    }

}



/**
 * Constructs suffix array on GPU
*/
void Gpusuffix::suffixOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    size_t* array1,
    size_t* array2,
    size_t* array3,
    size_t* array4,
    size_t* SA,
    size_t* done) {


    int numBlocks =  1; // i.e. number of thread blocks on the GPU
    int blockSize = 1; // i.e. number of GPU threads per thread block

 

    uint32_t N = seqLen;
    uint32_t k = kmerSize;
    int iter = 0;

    kmerPosConcat<<<numBlocks, blockSize>>>(compressedSeq, seqLen, kmerSize, array1);
    printf("N=%u\n", N);
    printf("k=%u\n", k);
    

    thrust::device_ptr<size_t> array1Ptr(array1);
    thrust::sort(array1Ptr, array1Ptr+seqLen-kmerSize+1);

    reBucketInitial<<<numBlocks, blockSize>>>(seqLen, kmerSize, array1, array2);
 
    uint32_t shift_val = 1;
    size_t* done2 = new size_t[1];
    do {
 
    reordering<<<numBlocks, blockSize>>>(seqLen, kmerSize, array1, array2, array3);
    shifting<<<numBlocks, blockSize>>>(seqLen, kmerSize, array3, array1, shift_val);
    shift_val = shift_val << 1;
    merge<<<numBlocks, blockSize>>>(seqLen, kmerSize, array3, array1, SA, array4);
    thrust::device_ptr<size_t> array4Ptr(array4);
    thrust::device_ptr<size_t> SAPtr(SA);
    thrust::sort_by_key(array4Ptr, array4Ptr+seqLen-kmerSize+1, SAPtr);
    reBucket<<<numBlocks, blockSize>>>(seqLen, kmerSize, array4, array2, array1, SA);
    singleton<<<numBlocks, blockSize>>>(seqLen, kmerSize, array2, done);

    cudaMemcpy(done2, done, (1)*sizeof(size_t), cudaMemcpyDeviceToHost);

    iter = iter+1;

    } while(done2[0] == 0);

    cudaDeviceSynchronize();
}



void Gpusuffix::DeviceArrays::printValues(uint32_t numValues, uint32_t kmerSize) {
    size_t* array2 = new size_t[numValues-kmerSize+1];
    size_t* array1 = new size_t[numValues-kmerSize+1];
    size_t* array3 = new size_t[numValues-kmerSize+1];
    // size_t* intermediate_array = new size_t[numValues];
    // size_t* intermediate_array2 = new size_t[numValues];
    size_t* SA = new size_t[numValues-kmerSize+2];
    cudaError_t err;

    err = cudaMemcpy(SA, d_SA, (numValues-kmerSize+1)*sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!!\n");
        exit(1);
    }

    
    
    err = cudaMemcpy(array1, d_array1, (numValues-kmerSize+1)*sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!!!\n");
        exit(1);
    }
    err = cudaMemcpy(array2, d_array2, (numValues-kmerSize+1)*sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!!!\n");
        exit(1);
    }
    err = cudaMemcpy(array3, d_array3, (numValues-kmerSize+1)*sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!!!\n");
        exit(1);
    }

   
    //prints out suffix array 
    FILE *fp;
    fp = fopen("out_suffix.txt", "w");

    for (uint32_t i = 0; i <= numValues-kmerSize; i++) {
    	fprintf(fp, "%lu\n",SA[i]);
    }

}



