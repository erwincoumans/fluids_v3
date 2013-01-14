/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012. Rama Hoetzlein, http://fluids3.com

  Fluids-ZLib license (* see part 1 below)
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. Acknowledgement of the
	 original author is required if you publish this in a paper, or use it
	 in a product. (See fluids3.com for details)
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/


#include <conio.h>
#include "cutil.h"				// cutil32.lib
#include "cutil_math.h"				// cutil32.lib
#include <string.h>
#include <assert.h>

#if defined(__APPLE__) || defined(MACOSX)
	#include <GLUT/glut.h>
#else
	#include "GL/glut.h"
#endif
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <math.h>
		
#include "fluid_system_host.cuh"		
#include "fluid_system_kern.cuh"

FluidParams		fcuda;
bufList			fbuf;

void cudaExit (int argc, char **argv)
{
	CUT_EXIT(argc, argv); 
}

// Initialize CUDA
void cudaInit(int argc, char **argv)
{   
    CUT_DEVICE_INIT(argc, argv);
 
	cudaDeviceProp p;
	cudaGetDeviceProperties ( &p, 0);
	
	printf ( "-- CUDA --\n" );
	printf ( "Name:       %s\n", p.name );
	printf ( "Revision:   %d.%d\n", p.major, p.minor );
	printf ( "Global Mem: %d\n", p.totalGlobalMem );
	printf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	printf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	printf ( "Warp Size:  %d\n", p.warpSize );
	printf ( "Mem Pitch:  %d\n", p.memPitch );
	printf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	printf ( "Const Mem:  %d\n", p.totalConstMem );
	printf ( "Clock Rate: %d\n", p.clockRate );	

	fbuf.mgridactive = 0x0;

	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mpos, sizeof(float)*3 ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mvel, sizeof(float)*3) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mveleval, sizeof(float)*3) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mforce, sizeof(float)*3) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mpress, sizeof(float) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mdensity, sizeof(float) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgcell, sizeof(uint)) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgndx, sizeof(uint)) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mclr, sizeof(uint)) );	

	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.msortbuf, sizeof(uint) ) );	

	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgrid, 1 ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgridcnt, 1 ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgridoff, 1 ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgridactive, 1 ) );

	//CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mcluster, sizeof(uint) ) );	

	preallocBlockSumsInt ( 1 );
};
	
// Compute number of blocks to create
int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int maxThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( maxThreads, numPnts );
    numBlocks = iDivUp ( numPnts, numThreads );
}

void FluidClearCUDA ()
{
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mpos ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mvel ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mveleval ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mforce ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mpress ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mdensity ) );		
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mgcell ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mgndx ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mclr ) );	
	//CUDA_SAFE_CALL ( cudaFree ( fbuf.mcluster ) );	

	CUDA_SAFE_CALL ( cudaFree ( fbuf.msortbuf ) );	

	CUDA_SAFE_CALL ( cudaFree ( fbuf.mgrid ) );
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mgridcnt ) );
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mgridoff ) );
	CUDA_SAFE_CALL ( cudaFree ( fbuf.mgridactive ) );
}


void FluidSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk )
{	
	fcuda.pnum = num;	
	fcuda.gridRes = res;
	fcuda.gridSize = size;
	fcuda.gridDelta = delta;
	fcuda.gridMin = gmin;
	fcuda.gridMax = gmax;
	fcuda.gridTotal = total;
	fcuda.gridSrch = gsrch;
	fcuda.gridAdjCnt = gsrch*gsrch*gsrch;
	fcuda.gridScanMax = res;
	fcuda.gridScanMax -= make_int3( fcuda.gridSrch, fcuda.gridSrch, fcuda.gridSrch );
	fcuda.chk = chk;

	// Build Adjacency Lookup
	int cell = 0;
	for (int y=0; y < gsrch; y++ ) 
		for (int z=0; z < gsrch; z++ ) 
			for (int x=0; x < gsrch; x++ ) 
				fcuda.gridAdj [ cell++]  = ( y * fcuda.gridRes.z+ z )*fcuda.gridRes.x +  x ;			
	
	printf ( "CUDA Adjacency Table\n");
	for (int n=0; n < fcuda.gridAdjCnt; n++ ) {
		printf ( "  ADJ: %d, %d\n", n, fcuda.gridAdj[n] );
	}	

	// Compute number of blocks and threads
    computeNumBlocks ( fcuda.pnum, 384, fcuda.numBlocks, fcuda.numThreads);			// particles
    computeNumBlocks ( fcuda.gridTotal, 384, fcuda.gridBlocks, fcuda.gridThreads);		// grid cell
    
	// Allocate particle buffers
    fcuda.szPnts = (fcuda.numBlocks  * fcuda.numThreads);     
    printf ( "CUDA Allocate: \n" );
	printf ( "  Pnts: %d, t:%dx%d=%d, Size:%d\n", fcuda.pnum, fcuda.numBlocks, fcuda.numThreads, fcuda.numBlocks*fcuda.numThreads, fcuda.szPnts);
    printf ( "  Grid: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", fcuda.gridTotal, fcuda.gridBlocks, fcuda.gridThreads, fcuda.gridBlocks*fcuda.gridThreads, fcuda.szGrid, (int) fcuda.gridRes.x, (int) fcuda.gridRes.y, (int) fcuda.gridRes.z );		
	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mpos,		fcuda.szPnts*sizeof(float)*3 ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mvel,		fcuda.szPnts*sizeof(float)*3 ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mveleval,	fcuda.szPnts*sizeof(float)*3 ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mforce,	fcuda.szPnts*sizeof(float)*3 ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mpress,	fcuda.szPnts*sizeof(float) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mdensity,	fcuda.szPnts*sizeof(float) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgcell,	fcuda.szPnts*sizeof(uint) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgndx,		fcuda.szPnts*sizeof(uint)) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mclr,		fcuda.szPnts*sizeof(uint) ) );	
	//CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mcluster,	fcuda.szPnts*sizeof(uint) ) );	

	int temp_size = 4*(sizeof(float)*3) + 2*sizeof(float) + 2*sizeof(int) + sizeof(uint);
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.msortbuf,	fcuda.szPnts*temp_size ) );	

	// Allocate grid
	fcuda.szGrid = (fcuda.gridBlocks * fcuda.gridThreads);  
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgrid,		fcuda.szPnts*sizeof(int) ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgridcnt,	fcuda.szGrid*sizeof(int) ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgridoff,	fcuda.szGrid*sizeof(int) ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.mgridactive, fcuda.szGrid*sizeof(int) ) );
		
	CUDA_SAFE_CALL ( cudaMemcpyToSymbol ( "simData", &fcuda, sizeof(FluidParams) ) );
	cudaThreadSynchronize ();

	// Prefix Sum - Preallocate Block sums for Sorting
	deallocBlockSumsInt ();
	preallocBlockSumsInt ( fcuda.gridTotal );
}

void FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl )
{
	fcuda.psimscale = ss;
	fcuda.psmoothradius = sr;
	fcuda.pradius = pr;
	fcuda.r2 = sr * sr;
	fcuda.pmass = mass;
	fcuda.prest_dens = rest;	
	fcuda.pboundmin = bmin;
	fcuda.pboundmax = bmax;
	fcuda.pextstiff = estiff;
	fcuda.pintstiff = istiff;
	fcuda.pvisc = visc;
	fcuda.pdamp = damp;
	fcuda.pforce_min = fmin;
	fcuda.pforce_max = fmax;
	fcuda.pforce_freq = ffreq;
	fcuda.pground_slope = gslope;
	fcuda.pgravity = make_float3( gx, gy, gz );
	fcuda.AL = al;
	fcuda.AL2 = al * al;
	fcuda.VL = vl;
	fcuda.VL2 = vl * vl;

	printf ( "Bound Min: %f %f %f\n", bmin.x, bmin.y, bmin.z );
	printf ( "Bound Max: %f %f %f\n", bmax.x, bmax.y, bmax.z );

	fcuda.pdist = pow ( fcuda.pmass / fcuda.prest_dens, 1/3.0f );
	fcuda.poly6kern = 315.0f / (64.0f * 3.141592 * pow( sr, 9.0f) );
	fcuda.spikykern = -45.0f / (3.141592 * pow( sr, 6.0f) );
	fcuda.lapkern = 45.0f / (3.141592 * pow( sr, 6.0f) );	

	CUDA_SAFE_CALL( cudaMemcpyToSymbol ( "simData", &fcuda, sizeof(FluidParams) ) );
	cudaThreadSynchronize ();
}

void CopyToCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr )
{
	// Send particle buffers
	int numPoints = fcuda.pnum;
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.mpos,		pos,			numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );	
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.mvel,		vel,			numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.mveleval, veleval,		numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.mforce,	force,			numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.mpress,	pressure,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.mdensity, density,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.mclr,		clr,			numPoints*sizeof(uint), cudaMemcpyHostToDevice ) );

	cudaThreadSynchronize ();	
}

void CopyFromCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr )
{
	// Return particle buffers
	int numPoints = fcuda.pnum;
	if ( pos != 0x0 ) CUDA_SAFE_CALL( cudaMemcpy ( pos,		fbuf.mpos,			numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	if ( clr != 0x0 ) CUDA_SAFE_CALL( cudaMemcpy ( clr,		fbuf.mclr,			numPoints*sizeof(uint),  cudaMemcpyDeviceToHost ) );
	/*CUDA_SAFE_CALL( cudaMemcpy ( vel,		fbuf.mvel,			numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy ( veleval,	fbuf.mveleval,		numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy ( force,		fbuf.mforce,		numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy ( pressure,	fbuf.mpress,		numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy ( density,	fbuf.mdensity,		numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );*/
	
	cudaThreadSynchronize ();	
}


void InsertParticlesCUDA ( uint* gcell, uint* ccell, int* gcnt )
{
	cudaMemset ( fbuf.mgridcnt, 0,			fcuda.gridTotal * sizeof(int));

	insertParticles<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: InsertParticlesCUDA: %s\n", cudaGetErrorString(error) );
	}  
	cudaThreadSynchronize ();
	// Transfer data back if requested (for validation)
	if (gcell != 0x0) {
		CUDA_SAFE_CALL( cudaMemcpy ( gcell,	fbuf.mgcell,	fcuda.pnum*sizeof(uint),		cudaMemcpyDeviceToHost ) );		
		CUDA_SAFE_CALL( cudaMemcpy ( gcnt,	fbuf.mgridcnt,	fcuda.gridTotal*sizeof(int),	cudaMemcpyDeviceToHost ) );
		//CUDA_SAFE_CALL( cudaMemcpy ( ccell,	fbuf.mcluster,	fcuda.pnum*sizeof(uint),		cudaMemcpyDeviceToHost ) );
	}
	
}

void PrefixSumCellsCUDA ( int* goff )
{
	// Prefix Sum - determine grid offsets
    prescanArrayRecursiveInt ( fbuf.mgridoff, fbuf.mgridcnt, fcuda.gridTotal, 0);
	cudaThreadSynchronize ();

	// Transfer data back if requested
	if ( goff != 0x0 ) {
		CUDA_SAFE_CALL( cudaMemcpy ( goff,	fbuf.mgridoff, fcuda.gridTotal * sizeof(int),  cudaMemcpyDeviceToHost ) );
	}
}

void CountingSortIndexCUDA ( uint* ggrid )
{	
	// Counting Sort - pass one, determine grid counts
	cudaMemset ( fbuf.mgrid,	GRID_UCHAR,	fcuda.pnum * sizeof(int) );

	countingSortIndex <<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );		
	cudaThreadSynchronize ();

	// Transfer data back if requested
	if ( ggrid != 0x0 ) {
		CUDA_SAFE_CALL( cudaMemcpy ( ggrid,	fbuf.mgrid, fcuda.pnum * sizeof(uint), cudaMemcpyDeviceToHost ) );
	}
}

void CountingSortFullCUDA ( uint* ggrid )
{
	// Transfer particle data to temp buffers
	int n = fcuda.pnum;
	cudaMemcpy ( fbuf.msortbuf + n*BUF_POS,		fbuf.mpos,		n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_VEL,		fbuf.mvel,		n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_VELEVAL,	fbuf.mveleval,	n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_FORCE,	fbuf.mforce,	n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_PRESS,	fbuf.mpress,	n*sizeof(float),	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_DENS,	fbuf.mdensity,	n*sizeof(float),	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_GCELL,	fbuf.mgcell,	n*sizeof(uint),		cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_GNDX,	fbuf.mgndx,		n*sizeof(uint),		cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_CLR,		fbuf.mclr,		n*sizeof(uint),		cudaMemcpyDeviceToDevice );

	// Counting Sort - pass one, determine grid counts
	cudaMemset ( fbuf.mgrid,	GRID_UCHAR,	fcuda.pnum * sizeof(int) );

	countingSortFull <<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );		
	cudaThreadSynchronize ();
}

void ComputePressureCUDA ()
{
	computePressure<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );	
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: ComputePressureCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaThreadSynchronize ();
}

void CountActiveCUDA ()
{
	int threads = 1;
	int blocks = 1;
	
	assert ( fbuf.mgridactive != 0x0 );
	
	cudaMemcpyToSymbol ( "gridActive", &fcuda.gridActive, sizeof(int) );

	countActiveCells<<< blocks, threads >>> ( fbuf, fcuda.gridTotal );
	cudaThreadSynchronize ();

	cudaMemcpyFromSymbol ( &fcuda.gridActive, "gridActive", sizeof(int) );
	
	printf ( "Active cells: %d\n", fcuda.gridActive );
}

void ComputePressureGroupCUDA ()
{
	if ( fcuda.gridActive > 0 ) {

		int threads = 128;		// should be based on maximum occupancy
		uint3 blocks;
		blocks.x = 4096;
		blocks.y = (fcuda.gridActive / 4096 )+1;
		blocks.z = 1;

		computePressureGroup<<< blocks, threads >>> ( fbuf, fcuda.pnum );	
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr, "CUDA ERROR: ComputePressureGroupCUDA: %s\n", cudaGetErrorString(error) );
		}   
		cudaThreadSynchronize ();
	}
}

void ComputeForceCUDA ()
{
	computeForce<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: ComputeForceCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaThreadSynchronize ();
}

void AdvanceCUDA ( float tm, float dt, float ss )
{
	advanceParticles<<< fcuda.numBlocks, fcuda.numThreads>>> ( tm, dt, ss, fbuf, fcuda.pnum );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: AdvanceCUDA: %s\n", cudaGetErrorString(error) );
	}    
    cudaThreadSynchronize ();
}



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

// includes, kernels
#include <assert.h>

inline bool isPowerOfTwo(int n) { return ((n&(n-1))==0) ; }

inline int floorPow2(int n) {
	#ifdef WIN32
		return 1 << (int)logb((float)n);
	#else
		int exp;
		frexp((float)n, &exp);
		return 1 << (exp - 1);
	#endif
}

#define BLOCK_SIZE 256

float**			g_scanBlockSums;
int**			g_scanBlockSumsInt;
unsigned int	g_numEltsAllocated = 0;
unsigned int	g_numLevelsAllocated = 0;

void preallocBlockSums(unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {       
        unsigned int numBlocks =   max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) level++;
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (float**) malloc(level * sizeof(float*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do {       
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) CUDA_SAFE_CALL ( cudaMalloc((void**) &g_scanBlockSums[level++], numBlocks * sizeof(float)) );
        numElts = numBlocks;
    } while (numElts > 1);
}
void preallocBlockSumsInt (unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {       
        unsigned int numBlocks =   max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) level++;
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSumsInt = (int**) malloc(level * sizeof(int*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do {       
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) CUDA_SAFE_CALL ( cudaMalloc((void**) &g_scanBlockSumsInt[level++], numBlocks * sizeof(int)) );
        numElts = numBlocks;
    } while (numElts > 1);
}

void deallocBlockSums()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++) cudaFree(g_scanBlockSums[i]);
    
    free( (void**)g_scanBlockSums );

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}
void deallocBlockSumsInt()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++) cudaFree(g_scanBlockSumsInt[i]);    
    free( (void**)g_scanBlockSumsInt );

    g_scanBlockSumsInt = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}



void prescanArrayRecursive (float *outArray, const float *inArray, int numElements, int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;
        if(!isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = floorPow2(numEltsLastBlock);            
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

	#ifdef DEBUG
		if (numBlocks > 1) assert(g_numEltsAllocated >= numElements);
	#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // execute the scan
    if (numBlocks > 1) {
        prescan<true, false><<< grid, threads, sharedMemSize >>> (outArray, inArray,  g_scanBlockSums[level], numThreads * 2, 0, 0);
        if (np2LastBlock) {
            prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>> (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be added to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive (g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level+1);

        uniformAdd<<< grid, threads >>> (outArray, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);
        if (np2LastBlock) {
            uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        prescan<false, false><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numThreads * 2, 0, 0);
    } else {
        prescan<false, true><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numElements, 0, 0);
    }
}

void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;
        if(!isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = floorPow2(numEltsLastBlock);            
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

	#ifdef DEBUG
		if (numBlocks > 1) assert(g_numEltsAllocated >= numElements);
	#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // execute the scan
    if (numBlocks > 1) {
        prescanInt <true, false><<< grid, threads, sharedMemSize >>> (outArray, inArray,  g_scanBlockSumsInt[level], numThreads * 2, 0, 0);
        if (np2LastBlock) {
            prescanInt <true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>> (outArray, inArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be added to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursiveInt (g_scanBlockSumsInt[level], g_scanBlockSumsInt[level], numBlocks, level+1);

        uniformAddInt <<< grid, threads >>> (outArray, g_scanBlockSumsInt[level], numElements - numEltsLastBlock, 0, 0);
        if (np2LastBlock) {
            uniformAddInt <<< 1, numThreadsLastBlock >>>(outArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        prescanInt <false, false><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numThreads * 2, 0, 0);
    } else {
        prescanInt <false, true><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numElements, 0, 0);
    }
}


void prescanArray ( float *d_odata, float *d_idata, int num )
{	
	// preform prefix sum
	preallocBlockSums( num );
    prescanArrayRecursive ( d_odata, d_idata, num, 0);
	deallocBlockSums();
}
void prescanArrayInt ( int *d_odata, int *d_idata, int num )
{	
	// preform prefix sum
	preallocBlockSumsInt ( num );
    prescanArrayRecursiveInt ( d_odata, d_idata, num, 0);
	deallocBlockSumsInt ();
}

char* d_idata = NULL;
char* d_odata = NULL;

void prefixSum ( int num )
{
	prescanArray ( (float*) d_odata, (float*) d_idata, num );
}

void prefixSumInt ( int num )
{	
	prescanArrayInt ( (int*) d_odata, (int*) d_idata, num );
}

void prefixSumToGPU ( char* inArray, int num, int siz )
{
    CUDA_SAFE_CALL ( cudaMalloc( (void**) &d_idata, num*siz ));
    CUDA_SAFE_CALL ( cudaMalloc( (void**) &d_odata, num*siz ));
    CUDA_SAFE_CALL ( cudaMemcpy( d_idata, inArray, num*siz, cudaMemcpyHostToDevice) );
}
void prefixSumFromGPU ( char* outArray, int num, int siz )
{		
	CUDA_SAFE_CALL ( cudaMemcpy( outArray, d_odata, num*siz, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL ( cudaFree( (void**) &d_idata ));
    CUDA_SAFE_CALL ( cudaFree( (void**) &d_odata ));
	d_idata = NULL;
	d_odata = NULL;
}
