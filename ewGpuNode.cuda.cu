/*
 * EasyWave - A realtime tsunami simulation program with GPU support.
 * Copyright (C) 2014  Andrey Babeyko, Johannes Spazier
 * GFZ German Research Centre for Geosciences (http://www.gfz-potsdam.de)
 *
 * Parts of this program (especially the GPU extension) were developed
 * within the context of the following publicly funded project:
 * - TRIDEC, EU 7th Framework Programme, Grant Agreement 258723
 *   (http://www.tridec-online.eu)
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence"),
 * complemented with the following provision: For the scientific transparency
 * and verification of results obtained and communicated to the public after
 * using a modified version of the work, You (as the recipient of the source
 * code and author of this modified version, used to produce the published
 * results in scientific communications) commit to make this modified source
 * code available in a repository that is easily and freely accessible for a
 * duration of five years after the communication of the obtained results.
 *
 * You may not use this work except in compliance with the Licence.
 *
 * You may obtain a copy of the Licence at:
 * https://joinup.ec.europa.eu/software/page/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

#include <string>

#include "cuda.h"
#include "ewGpuNode.hpp"
#include "ewKernels.cuda.cuh"

#define CUDA_CALL(x) if( (x) != cudaSuccess ) { throw std::runtime_error("Error in file " __FILE__ ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString( cudaGetLastError() ) ); }

CGpuNode::CGpuNode() {

	pitch = 0;
	copied = true;

	int device;
	struct cudaDeviceProp props;

	CUDA_CALL(cudaGetDevice(&device));
	CUDA_CALL(cudaGetDeviceProperties(&props, device));

	std::cout << "Selected device: [" << device << "]: " << props.name << std::endl;

	have_profiling = true;
	for (auto &kd: kernel_duration) {
		kd = 0;
	}
}

CGpuNode::~CGpuNode()
{
	if (have_profiling) {
		dumpProfilingData();
	}
}

int CGpuNode::mallocMem() {

	CArrayNode::mallocMem();

	Params& dp = data.params;

	/* fill in some fields here */
	dp.nI = NLon;
	dp.nJ = NLat;
	dp.sshArrivalThreshold = Par.sshArrivalThreshold;
	dp.sshClipThreshold = Par.sshClipThreshold;
	dp.sshZeroThreshold = Par.sshZeroThreshold;
	dp.lpad = 31;

	size_t nJ_aligned = dp.nJ + dp.lpad;

	/* 2-dim */
	CUDA_CALL( cudaMallocPitch( &(data.d), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	CUDA_CALL( cudaMallocPitch( &(data.h), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	CUDA_CALL( cudaMallocPitch( &(data.hMax), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	CUDA_CALL( cudaMallocPitch( &(data.fM), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	CUDA_CALL( cudaMallocPitch( &(data.fN), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	CUDA_CALL( cudaMallocPitch( &(data.cR1), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	CUDA_CALL( cudaMallocPitch( &(data.cR2), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	CUDA_CALL( cudaMallocPitch( &(data.cR4), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	CUDA_CALL( cudaMallocPitch( &(data.tArr), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	/* TODO: cR3, cR5 for coriolis */

	/* 1-dim */
	CUDA_CALL( cudaMalloc( &(data.cR6), dp.nJ * sizeof(float) ) );
	CUDA_CALL( cudaMalloc( &(data.cB1), dp.nI * sizeof(float) ) );
	CUDA_CALL( cudaMalloc( &(data.cB2), dp.nJ * sizeof(float) ) );
	CUDA_CALL( cudaMalloc( &(data.cB3), dp.nI * sizeof(float) ) );
	CUDA_CALL( cudaMalloc( &(data.cB4), dp.nJ * sizeof(float) ) );

	CUDA_CALL( cudaMalloc( &(data.g_MinMax), sizeof(int4) ) );

	/* TODO: make sure that pitch is a multiple of 4 and the same for each cudaMallocPitch() call */
	dp.pI = pitch / sizeof(float);

	return 0;
}

int CGpuNode::copyToGPU() {

	Params& dp = data.params;

	/* align left grid boundary to a multiple of 32 with an offset 1 */
        Jmin -= (Jmin-2) % MEM_ALIGN;

        /* fill in further fields here */
        dp.iMin = Imin;
	dp.iMax = Imax;
        dp.jMin = Jmin;
	dp.jMax = Jmax;

	/* add offset to data.d to guarantee alignment: data.d + LPAD */
	/* 2-dim */
	CUDA_CALL( cudaMemcpy2D( data.d + dp.lpad, pitch, d, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy2D( data.h + dp.lpad, pitch, h, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy2D( data.hMax + dp.lpad, pitch, hMax, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy2D( data.fM + dp.lpad, pitch, fM, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy2D( data.fN + dp.lpad, pitch, fN, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy2D( data.cR1 + dp.lpad, pitch, cR1, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy2D( data.cR2 + dp.lpad, pitch, cR2, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy2D( data.cR4 + dp.lpad, pitch, cR4, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy2D( data.tArr + dp.lpad, pitch, tArr, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice ) );

	/* FIXME: move global variables into data structure */
	/* 1-dim */
	CUDA_CALL( cudaMemcpy( data.cR6, R6, dp.nJ * sizeof(float), cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy( data.cB1, C1, dp.nI * sizeof(float), cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy( data.cB2, C2, dp.nJ * sizeof(float), cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy( data.cB3, C3, dp.nI * sizeof(float), cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy( data.cB4, C4, dp.nJ * sizeof(float), cudaMemcpyHostToDevice ) );

	return 0;
}
int CGpuNode::copyFromGPU() {

	Params& dp = data.params;

	CUDA_CALL( cudaMemcpy2D( hMax, dp.nJ * sizeof(float), data.hMax + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy2D( tArr, dp.nJ * sizeof(float), data.tArr + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, cudaMemcpyDeviceToHost ) );

	return 0;
}

int CGpuNode::copyIntermediate() {

	/* ignore copy requests if data already present on CPU side */
	if( copied )
		return 0;

	Params& dp = data.params;

	CUDA_CALL( cudaMemcpy2D( h, dp.nJ * sizeof(float), data.h + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, cudaMemcpyDeviceToHost ) );

	/* copy finished */
	copied = true;

	return 0;
}

int CGpuNode::copyPOIs() {

	Params& dp = data.params;

	if( copied )
		return 0;

	for( int n = 0; n < NPOIs; n++ ) {

		int i = idxPOI[n] / dp.nJ + 1;
		int j = idxPOI[n] % dp.nJ + 1;

		int id = data.idx( i, j );

		CUDA_CALL( cudaMemcpy( h + idxPOI[n], data.h + dp.lpad + id, sizeof(float), cudaMemcpyDeviceToHost ) );
	}

	return 0;
}

int CGpuNode::freeMem() {

	/* 2-dim */
	CUDA_CALL( cudaFree( data.d ) );
	CUDA_CALL( cudaFree( data.h ) );
	CUDA_CALL( cudaFree( data.hMax ) );
	CUDA_CALL( cudaFree( data.fM ) );
	CUDA_CALL( cudaFree( data.fN ) );
	CUDA_CALL( cudaFree( data.cR1 ) );
	CUDA_CALL( cudaFree( data.cR2 ) );
	CUDA_CALL( cudaFree( data.cR4 ) );
	CUDA_CALL( cudaFree( data.tArr ) );

	/* 1-dim */
	CUDA_CALL( cudaFree( data.cR6 ) );
	CUDA_CALL( cudaFree( data.cB1 ) );
	CUDA_CALL( cudaFree( data.cB2 ) );
	CUDA_CALL( cudaFree( data.cB3 ) );
	CUDA_CALL( cudaFree( data.cB4 ) );

	CUDA_CALL( cudaFree( data.g_MinMax ) );

	CArrayNode::freeMem();

	return 0;
}

int CGpuNode::run() {

	Params& dp = data.params;

	static cudaEvent_t evtStart[NUM_TIMED_KERNELS];
	static cudaEvent_t evtEnd[NUM_TIMED_KERNELS];
	static bool events_initialized = false;

	if (!events_initialized) {
	        for (int i = 0; i < NUM_TIMED_KERNELS; i++) {
        	        cudaEventCreate(&(evtStart[i]));
                	cudaEventCreate(&(evtEnd[i]));
	        }

		events_initialized = true;
	}

	int nThreads = 256;
	int xThreads = 32;
	int yThreads = nThreads / xThreads;

	int NJ = dp.jMax - dp.jMin + 1;
	int NI = dp.iMax - dp.iMin + 1;
	int xBlocks = ceil( (float)NJ / (float)xThreads );
	int yBlocks = ceil( (float)NI / (float)yThreads );

	dim3 threads( xThreads, yThreads );
	dim3 blocks( xBlocks, yBlocks );

	int nBlocks = ceil( (float)max(dp.nI,dp.nJ) / (float)nThreads );

	dp.mTime = Par.time;

	CUDA_CALL( cudaEventRecord( evtStart[KERNEL_WAVE_UPDATE], 0 ) );
	waveUpdateKernel<<<blocks,threads>>>( data );
	CUDA_CALL( cudaEventRecord( evtEnd[KERNEL_WAVE_UPDATE], 0 ) );

	CUDA_CALL( cudaEventRecord( evtStart[KERNEL_WAVE_BOUND], 0 ) );
	waveBoundaryKernel<<<nBlocks,nThreads>>>( data );
	CUDA_CALL( cudaEventRecord( evtEnd[KERNEL_WAVE_BOUND], 0 ) );

	CUDA_CALL( cudaEventRecord( evtStart[KERNEL_FLUX_UPDATE], 0 ) );
	fluxUpdateKernel<<<blocks,threads>>>( data );
	CUDA_CALL( cudaEventRecord( evtEnd[KERNEL_FLUX_UPDATE], 0 ) );

	CUDA_CALL( cudaEventRecord( evtStart[KERNEL_FLUX_BOUND], 0 ) );
	fluxBoundaryKernel<<<nBlocks,nThreads>>>( data );
	CUDA_CALL( cudaEventRecord( evtEnd[KERNEL_FLUX_BOUND], 0 ) );

	CUDA_CALL( cudaEventRecord( evtStart[KERNEL_MEMSET], 0 ) );
	CUDA_CALL( cudaMemset( data.g_MinMax, 0, sizeof(data.g_MinMax) ) );
	CUDA_CALL( cudaEventRecord( evtEnd[KERNEL_MEMSET], 0 ) );

	CUDA_CALL( cudaEventRecord( evtStart[KERNEL_EXTEND], 0 ) );
	gridExtendKernel<<<nBlocks,nThreads>>>( data );
	CUDA_CALL( cudaEventRecord( evtEnd[KERNEL_EXTEND], 0 ) );

	int4 MinMax;
	CUDA_CALL( cudaEventRecord( evtStart[KERNEL_MEMCPY], 0 ) );
	CUDA_CALL( cudaMemcpy( &MinMax, data.g_MinMax, sizeof(MinMax), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaEventRecord( evtEnd[KERNEL_MEMCPY], 0 ) );
	cudaDeviceSynchronize();

	if( MinMax.x ) Imin = dp.iMin = max( dp.iMin-1, 2 );
	if( MinMax.y ) Imax = dp.iMax = min( dp.iMax+1, dp.nI-1 );
	if( MinMax.z ) Jmin = dp.jMin = max( dp.jMin-MEM_ALIGN, 2 );
	if( MinMax.w ) Jmax = dp.jMax = min( dp.jMax+1, dp.nJ-1 );

	float duration;
	for( int j = 0; j < NUM_TIMED_KERNELS; j++ ) {
		cudaEventElapsedTime( &duration, evtStart[j], evtEnd[j]);
		kernel_duration[j] += duration;
	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
