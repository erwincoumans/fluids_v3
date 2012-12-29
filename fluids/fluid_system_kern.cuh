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

#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <stdio.h>
	#include <math.h>

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;

	// Particle & Grid Buffers
	struct bufList {
		float3*			mpos;
		float3*			mvel;
		float3*			mveleval;
		float3*			mforce;
		float*			mpress;
		float*			mdensity;		
		uint*			mgcell;
		uint*			mgndx;
		uint*			mclr;			// 4 byte color

		uint*			mcluster;

		char*			msortbuf;

		uint*			mgrid;	
		int*			mgridcnt;
		int*			mgridoff;
		int*			mgridactive;
	};

	// Temporary sort buffer offsets
	#define BUF_POS			0
	#define BUF_VEL			(sizeof(float3))
	#define BUF_VELEVAL		(BUF_VEL + sizeof(float3))
	#define BUF_FORCE		(BUF_VELEVAL + sizeof(float3))
	#define BUF_PRESS		(BUF_FORCE + sizeof(float3))
	#define BUF_DENS		(BUF_PRESS + sizeof(float))
	#define BUF_GCELL		(BUF_DENS + sizeof(float))
	#define BUF_GNDX		(BUF_GCELL + sizeof(uint))
	#define BUF_CLR			(BUF_GNDX + sizeof(uint))

	// Fluid Parameters (stored on both host and device)
	struct FluidParams {
		int				numThreads, numBlocks;
		int				gridThreads, gridBlocks;	

		int				szPnts, szHash, szGrid;
		int				stride, pnum;
		int				chk;
		float			pdist, pmass, prest_dens;
		float			pextstiff, pintstiff;
		float			pradius, psmoothradius, r2, psimscale, pvisc;
		float			pforce_min, pforce_max, pforce_freq, pground_slope;
		float			pvel_limit, paccel_limit, pdamp;
		float3			pboundmin, pboundmax, pgravity;
		float			AL, AL2, VL, VL2;
		
		float			poly6kern, spikykern, lapkern;

		float3			gridSize, gridDelta, gridMin, gridMax;
		int3			gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;

		int				gridAdj[64];
	};

	// Prefix Sum defines - 16 banks on G80
	#define NUM_BANKS		16
	#define LOG_NUM_BANKS	 4

	#ifndef CUDA_KERNEL
		extern "C" {
			FluidParams	simData;
			uint		gridActive;
		}		
		__global__ void insertParticles ( bufList buf, int pnum );
		__global__ void countingSortIndex ( bufList buf, int pnum );		
		__global__ void countingSortFull ( bufList buf, int pnum );		
		__global__ void computePressure ( bufList buf, int pnum );		
		__global__ void computeForce ( bufList buf, int pnum );
		__global__ void computePressureGroup ( bufList buf, int pnum );
		__global__ void advanceParticles ( float time, float dt, float ss, bufList buf, int numPnts );

		__global__ void countActiveCells ( bufList buf, int pnum );		

		// Prefix Sum
		#include "prefix_sum.cu"
		// NOTE: Template functions must be defined in the header
		template <bool storeSum, bool isNP2> __global__ void prescan(float *g_odata, const float *g_idata, float *g_blockSums, int n, int blockIndex, int baseIndex) {
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ float s_data[];
			loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
			prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
			storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
		}
		template <bool storeSum, bool isNP2> __global__ void prescanInt (int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ int s_dataInt [];
			loadSharedChunkFromMemInt <isNP2>(s_dataInt, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
			prescanBlockInt<storeSum>(s_dataInt, blockIndex, g_blockSums); 
			storeSharedChunkToMemInt <isNP2>(g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
		}
		__global__ void uniformAddInt (int*  g_data, int *uniforms, int n, int blockOffset, int baseIndex);	
		__global__ void uniformAdd    (float*g_data, float *uniforms, int n, int blockOffset, int baseIndex);	
	#endif
	

	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295

	
#endif
