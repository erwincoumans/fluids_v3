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

#ifndef DEF_HOST_CUDA
	#define DEF_HOST_CUDA

	#include <vector_types.h>	
	#include <driver_types.h>			// for cudaStream_t

	#define TOTAL_THREADS			1000000
	#define BLOCK_THREADS			256
	#define MAX_NBR					80	
	
	#define COLOR(r,g,b)	( (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )
	#define COLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

	typedef unsigned int		uint;
	typedef unsigned short		ushort;
	typedef unsigned char		uchar;

	#define OFFSET_POS		0
	#define OFFSET_VEL		12
	#define OFFSET_VELEVAL	24
	#define OFFSET_FORCE	36
	#define OFFSET_PRESS	48
	#define OFFSET_DENS		52
	#define OFFSET_CELL		56
	#define OFFSET_GCONT	60
	#define OFFSET_CLR		64

	extern "C"
	{

	void cudaInit(int argc, char **argv);
	void cudaExit(int argc, char **argv);

	void FluidClearCUDA ();
	void FluidSetupCUDA (  int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk );
	void FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl );

	void CopyToCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr );
	void CopyFromCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr );
	
	void InsertParticlesCUDA ( uint* gcell, uint* ccell, int* gcnt );	
	void PrefixSumCellsCUDA ( int* goff );
	//void RadixSortIndexCUDA ( uint* ggrid );
	void CountingSortIndexCUDA ( uint* ggrid );	
	void CountingSortFullCUDA ( uint* ggrid );
	void ComputePressureCUDA ();
	void ComputeForceCUDA ();	
	
	void CountActiveCUDA ();						// Clustering method (shared mem)
	void ComputePressureGroupCUDA ();				// Clustering method (shared mem)

	void AdvanceCUDA ( float time, float dt, float ss );

	void prefixSumToGPU ( char* inArray, int num, int siz );
	void prefixSumFromGPU ( char* outArray, int num, int siz );
	void prefixSum ( int num );
	void prefixSumInt ( int num );
	void preallocBlockSumsInt(unsigned int num);
	void deallocBlockSumsInt();
	void prescanArray ( float* outArray, float* inArray, int numElements );
	void prescanArrayInt ( int* outArray, int* inArray, int numElements );
	void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level);

	}

#endif