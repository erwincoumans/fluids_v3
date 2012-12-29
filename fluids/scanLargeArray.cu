/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>

// includes, kernels
#include <scan.cu>  // defines prescanArray()

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// regression test functionality
extern "C" 
unsigned int compare( const float* reference, const float* data, 
                     const unsigned int len);
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

#ifndef __DEVICE_EMULATION__
    unsigned int num_test_iterations = 100;
    unsigned int num_elements = 1000000; // can support large, non-power-of-2 arrays!
#else
    unsigned int num_test_iterations = 1;
    unsigned int num_elements = 10000; // can support large, non-power-of-2 arrays!
#endif
    
    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int*)&num_elements);
    cutGetCmdLineArgumenti( argc, (const char**) argv, "i", (int*)&num_test_iterations);

    unsigned int mem_size = sizeof( float) * num_elements;
    
    unsigned int timerGPU, timerCPU;
    cutilCheckError(cutCreateTimer(&timerCPU));
    cutilCheckError(cutCreateTimer(&timerGPU));

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( mem_size);
      
    // initialize the input data on the host
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        h_data[i] = 1.0f;//(int)(10 * rand()/32768.f);
    }

    // compute reference solution
    float* reference = (float*) malloc( mem_size); 
    cutStartTimer(timerCPU);
    for (unsigned int i = 0; i < num_test_iterations; i++)
    {
        computeGold( reference, h_data, num_elements);
    }
    cutStopTimer(timerCPU);

    // allocate device memory input and output arrays
    float* d_idata = NULL;
    float* d_odata = NULL;

    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));
    
    // copy host memory to device input array
    cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );
    // initialize all the other device arrays to be safe
    cutilSafeCall( cudaMemcpy( d_odata, h_data, mem_size, cudaMemcpyHostToDevice) );

    printf("Running parallel prefix sum (prescan) of %d elements\n", num_elements);
    printf("This version is work efficient (O(n) adds)\n");
    printf("and has very few shared memory bank conflicts\n\n");

    preallocBlockSums(num_elements);

    // run once to remove startup overhead
    prescanArray(d_odata, d_idata, num_elements);


    // Run the prescan
    cutStartTimer(timerGPU);
    for (unsigned int i = 0; i < num_test_iterations; i++)
    {
        //printf("prescanArray\n");
        prescanArray(d_odata, d_idata, num_elements);
    }
    cutStopTimer(timerGPU);

    deallocBlockSums();    

    // copy result from device to host
    cutilSafeCall(cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, 
                               cudaMemcpyDeviceToHost));

    // If this is a regression test write the results to a file
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test 
        cutWriteFilef( "./data/result.dat", h_data, num_elements, 0.0);
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        unsigned int result_regtest = cutComparef( reference, h_data, num_elements);
        printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
        printf( "Average GPU execution time: %f ms\n", cutGetTimerValue(timerGPU) / num_test_iterations);
        printf( "CPU execution time:         %f ms\n", cutGetTimerValue(timerCPU) / num_test_iterations);
    }

    printf("\nCheck out the CUDA Data Parallel Primitives Library for more on scan.\n");
    printf("http://www.gpgpu.org/developer/cudpp\n");

    // cleanup memory
    cutDeleteTimer(timerCPU);
    cutDeleteTimer(timerGPU);
    free( h_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);

    cudaThreadExit();
}
