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

#include "hip/hip_runtime.h"
#include "ewGpuNode.hpp"
#include "ewKernels.cuda.cuh"

#define HIP_CALL(x) if( (x) != hipSuccess ) { throw std::runtime_error("Error in file " __FILE__ ":" + std::to_string(__LINE__) + ": " + hipGetErrorString( hipGetLastError() ) ); }

CGpuNode::CGpuNode() {

	pitch = 0;
	copied = true;

	int device;
	struct hipDeviceProp_t props;

	HIP_CALL(hipGetDevice(&device));
	HIP_CALL(hipGetDeviceProperties(&props, device));

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
	HIP_CALL( hipMallocPitch( (void**) &(data.d), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	HIP_CALL( hipMallocPitch( (void**) &(data.h), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	HIP_CALL( hipMallocPitch( (void**) &(data.hMax), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	HIP_CALL( hipMallocPitch( (void**) &(data.fM), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	HIP_CALL( hipMallocPitch( (void**) &(data.fN), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	HIP_CALL( hipMallocPitch( (void**) &(data.cR1), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	HIP_CALL( hipMallocPitch( (void**) &(data.cR2), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	HIP_CALL( hipMallocPitch( (void**) &(data.cR4), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	HIP_CALL( hipMallocPitch( (void**) &(data.tArr), &pitch, nJ_aligned * sizeof(float), dp.nI ) );
	/* TODO: cR3, cR5 for coriolis */

	/* 1-dim */
	HIP_CALL( hipMalloc( &(data.cR6), dp.nJ * sizeof(float) ) );
	HIP_CALL( hipMalloc( &(data.cB1), dp.nI * sizeof(float) ) );
	HIP_CALL( hipMalloc( &(data.cB2), dp.nJ * sizeof(float) ) );
	HIP_CALL( hipMalloc( &(data.cB3), dp.nI * sizeof(float) ) );
	HIP_CALL( hipMalloc( &(data.cB4), dp.nJ * sizeof(float) ) );

	HIP_CALL( hipMalloc( &(data.g_MinMax), sizeof(int4) ) );

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
	HIP_CALL( hipMemcpy2D( data.d + dp.lpad, pitch, d, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy2D( data.h + dp.lpad, pitch, h, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy2D( data.hMax + dp.lpad, pitch, hMax, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy2D( data.fM + dp.lpad, pitch, fM, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy2D( data.fN + dp.lpad, pitch, fN, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy2D( data.cR1 + dp.lpad, pitch, cR1, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy2D( data.cR2 + dp.lpad, pitch, cR2, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy2D( data.cR4 + dp.lpad, pitch, cR4, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy2D( data.tArr + dp.lpad, pitch, tArr, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, hipMemcpyHostToDevice ) );

	/* FIXME: move global variables into data structure */
	/* 1-dim */
	HIP_CALL( hipMemcpy( data.cR6, R6, dp.nJ * sizeof(float), hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy( data.cB1, C1, dp.nI * sizeof(float), hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy( data.cB2, C2, dp.nJ * sizeof(float), hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy( data.cB3, C3, dp.nI * sizeof(float), hipMemcpyHostToDevice ) );
	HIP_CALL( hipMemcpy( data.cB4, C4, dp.nJ * sizeof(float), hipMemcpyHostToDevice ) );

	return 0;
}
int CGpuNode::copyFromGPU() {

	Params& dp = data.params;

	HIP_CALL( hipMemcpy2D( hMax, dp.nJ * sizeof(float), data.hMax + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, hipMemcpyDeviceToHost ) );
	HIP_CALL( hipMemcpy2D( tArr, dp.nJ * sizeof(float), data.tArr + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, hipMemcpyDeviceToHost ) );

	return 0;
}

int CGpuNode::copyIntermediate() {

	/* ignore copy requests if data already present on CPU side */
	if( copied )
		return 0;

	Params& dp = data.params;

	HIP_CALL( hipMemcpy2D( h, dp.nJ * sizeof(float), data.h + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, hipMemcpyDeviceToHost ) );

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

		HIP_CALL( hipMemcpy( h + idxPOI[n], data.h + dp.lpad + id, sizeof(float), hipMemcpyDeviceToHost ) );
	}

	return 0;
}

int CGpuNode::freeMem() {

	/* 2-dim */
	HIP_CALL( hipFree( data.d ) );
	HIP_CALL( hipFree( data.h ) );
	HIP_CALL( hipFree( data.hMax ) );
	HIP_CALL( hipFree( data.fM ) );
	HIP_CALL( hipFree( data.fN ) );
	HIP_CALL( hipFree( data.cR1 ) );
	HIP_CALL( hipFree( data.cR2 ) );
	HIP_CALL( hipFree( data.cR4 ) );
	HIP_CALL( hipFree( data.tArr ) );

	/* 1-dim */
	HIP_CALL( hipFree( data.cR6 ) );
	HIP_CALL( hipFree( data.cB1 ) );
	HIP_CALL( hipFree( data.cB2 ) );
	HIP_CALL( hipFree( data.cB3 ) );
	HIP_CALL( hipFree( data.cB4 ) );

	HIP_CALL( hipFree( data.g_MinMax ) );

	CArrayNode::freeMem();

	return 0;
}

int CGpuNode::run() {

	Params& dp = data.params;

	static hipEvent_t evtStart[NUM_TIMED_KERNELS];
	static hipEvent_t evtEnd[NUM_TIMED_KERNELS];
	static bool events_initialized = false;

	if (!events_initialized) {
	        for (int i = 0; i < NUM_TIMED_KERNELS; i++) {
        	        HIP_CALL(hipEventCreate(&(evtStart[i])));
                	HIP_CALL(hipEventCreate(&(evtEnd[i])));
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

	int nBlocks = ceil( (float)std::max(dp.nI,dp.nJ) / (float)nThreads );

	dp.mTime = Par.time;

	HIP_CALL( hipEventRecord( evtStart[KERNEL_WAVE_UPDATE], 0 ) );
	waveUpdateKernel<<<blocks,threads>>>( data );
	HIP_CALL( hipEventRecord( evtEnd[KERNEL_WAVE_UPDATE], 0 ) );

	HIP_CALL( hipEventRecord( evtStart[KERNEL_WAVE_BOUND], 0 ) );
	waveBoundaryKernel<<<nBlocks,nThreads>>>( data );
	HIP_CALL( hipEventRecord( evtEnd[KERNEL_WAVE_BOUND], 0 ) );

	HIP_CALL( hipEventRecord( evtStart[KERNEL_FLUX_UPDATE], 0 ) );
	fluxUpdateKernel<<<blocks,threads>>>( data );
	HIP_CALL( hipEventRecord( evtEnd[KERNEL_FLUX_UPDATE], 0 ) );

	HIP_CALL( hipEventRecord( evtStart[KERNEL_FLUX_BOUND], 0 ) );
	fluxBoundaryKernel<<<nBlocks,nThreads>>>( data );
	HIP_CALL( hipEventRecord( evtEnd[KERNEL_FLUX_BOUND], 0 ) );

	HIP_CALL( hipEventRecord( evtStart[KERNEL_MEMSET], 0 ) );
	HIP_CALL( hipMemset( data.g_MinMax, 0, sizeof(*data.g_MinMax) ) );
	HIP_CALL( hipEventRecord( evtEnd[KERNEL_MEMSET], 0 ) );

	HIP_CALL( hipEventRecord( evtStart[KERNEL_EXTEND], 0 ) );
	gridExtendKernel<<<nBlocks,nThreads>>>( data );
	HIP_CALL( hipEventRecord( evtEnd[KERNEL_EXTEND], 0 ) );

	int4 MinMax;
	static_assert(sizeof(MinMax) == sizeof(*data.g_MinMax));
	HIP_CALL( hipEventRecord( evtStart[KERNEL_MEMCPY], 0 ) );
	HIP_CALL( hipMemcpy( &MinMax, data.g_MinMax, sizeof(MinMax), hipMemcpyDeviceToHost ) );
	HIP_CALL( hipEventRecord( evtEnd[KERNEL_MEMCPY], 0 ) );
	HIP_CALL( hipDeviceSynchronize() );

	if( MinMax.x ) Imin = dp.iMin = std::max( dp.iMin-1, 2 );
	if( MinMax.y ) Imax = dp.iMax = std::min( dp.iMax+1, dp.nI-1 );
	if( MinMax.z ) Jmin = dp.jMin = std::max( dp.jMin-MEM_ALIGN, 2 );
	if( MinMax.w ) Jmax = dp.jMax = std::min( dp.jMax+1, dp.nJ-1 );

	float duration;
	for( int j = 0; j < NUM_TIMED_KERNELS; j++ ) {
		HIP_CALL(hipEventElapsedTime( &duration, evtStart[j], evtEnd[j]));
		kernel_duration[j] += duration;
	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
