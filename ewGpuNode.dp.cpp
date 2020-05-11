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

#include <unistd.h>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ewGpuNode.dp.hpp"
//#include "ewCudaKernels.dp.hpp"
#include <cmath>
#include "utilits.h"

/* ugly as hell but a quickfix for runtime issue (undefined reference) when compiling for DPC++ */
#include "ewCudaKernels.dp.cpp"

CGpuNode::CGpuNode() {

    char* dev_spec = getenv("EW_CL_DEVICE");
    if (dev_spec) {
        unsigned int dev_id = atoi(dev_spec);
        dpct::get_device_manager().select_device(dev_id);
    }

    dpct::device_info di;
    dpct::get_device_manager().current_device().get_device_info(di);
    std::cout << "running in accellerated mode using " << di.get_name() << std::endl;
	dpct::get_device_manager().current_device().queues_wait_and_throw();

	pitch = 0;
	copied = true;

	for( int i = 0; i < 5; i++ ) {
		dur[i] = 0.0;
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
	dp.lpad = 0/*31*/;

	size_t nJ_aligned = dp.nJ + dp.lpad;

	/* 2-dim */
	//cudaMallocPitch( &data.d, &pitch, nJ_aligned * sizeof(float), dp.nI );
	//cudaMallocPitch( &data.h, &pitch, nJ_aligned * sizeof(float), dp.nI );
	//cudaMallocPitch( &(data.hMax), &pitch, nJ_aligned * sizeof(float), dp.nI );
	//cudaMallocPitch( &(data.fM), &pitch, nJ_aligned * sizeof(float), dp.nI );
	//cudaMallocPitch( &(data.fN), &pitch, nJ_aligned * sizeof(float), dp.nI );
	//cudaMallocPitch( &(data.cR1), &pitch, nJ_aligned * sizeof(float), dp.nI );
	//cudaMallocPitch( &(data.cR2), &pitch, nJ_aligned * sizeof(float), dp.nI );
	//cudaMallocPitch( &(data.cR4), &pitch, nJ_aligned * sizeof(float), dp.nI );
	//cudaMallocPitch( &(data.tArr), &pitch, nJ_aligned * sizeof(float), dp.nI );
    *((void **)&(data.d_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
    *((void **)&(data.h_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
    *((void **)&(data.hMax_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
    *((void **)&(data.fM_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
    *((void **)&(data.fN_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
    *((void **)&(data.cR1_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
    *((void **)&(data.cR2_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
    *((void **)&(data.cR4_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
    *((void **)&(data.tArr_ptr)) = cl::sycl::malloc_device(nJ_aligned * sizeof(float) * dp.nI, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());

	/* TODO: cR3, cR5 for coriolis */

	/* 1-dim */
	*((void **)&(data.cR6_ptr)) = cl::sycl::malloc_device(dp.nJ * sizeof(float), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
	*((void **)&(data.cB1_ptr)) = cl::sycl::malloc_device(dp.nI * sizeof(float), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
	*((void **)&(data.cB2_ptr)) = cl::sycl::malloc_device(dp.nJ * sizeof(float), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
	*((void **)&(data.cB3_ptr)) = cl::sycl::malloc_device(dp.nI * sizeof(float), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
	*((void **)&(data.cB4_ptr)) = cl::sycl::malloc_device(dp.nJ * sizeof(float), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());

    /* no malloc required for fixed-sized array */
	*((void **)&(data.g_MinMax)) = cl::sycl::malloc_device(4 * sizeof(*data.g_MinMax), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());

	/* TODO: make sure that pitch is a multiple of 4 and the same for each cudaMallocPitch() call */
    /* fake CUDA pitch */
    pitch = nJ_aligned * sizeof(float);
	dp.pI = pitch / sizeof(float);

	dpct::get_device_manager().current_device().queues_wait_and_throw();

	return 0;
}

int CGpuNode::copyToGPU() {

	Params& dp = data.params;

	/* align left grid boundary to a multiple of 32 with an offset 1 */
    Jmin -= (Jmin-2) % 32;

    /* fill in further fields here */
    dp.iMin = Imin;
    dp.iMax = Imax;
    dp.jMin = Jmin;
    dp.jMax = Jmax;

	/* add offset to data.d to guarantee alignment: data.d + LPAD */
	/* 2-dim */
	//cudaMemcpy2D( data.d + dp.lpad, pitch, d, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	//cudaMemcpy2D( data.h + dp.lpad, pitch, h, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	//cudaMemcpy2D( data.hMax + dp.lpad, pitch, hMax, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	//cudaMemcpy2D( data.fM + dp.lpad, pitch, fM, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	//cudaMemcpy2D( data.fN + dp.lpad, pitch, fN, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	//cudaMemcpy2D( data.cR1 + dp.lpad, pitch, cR1, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	//cudaMemcpy2D( data.cR2 + dp.lpad, pitch, cR2, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	//cudaMemcpy2D( data.cR4 + dp.lpad, pitch, cR4, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	//cudaMemcpy2D( data.tArr + dp.lpad, pitch, tArr, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, cudaMemcpyHostToDevice );
	dpct::get_default_queue_wait().memcpy( (void*)(data.d_ptr + dp.lpad), (void*)(d), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.h_ptr + dp.lpad), (void*)(h), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.hMax_ptr + dp.lpad), (void*)(hMax), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.fM_ptr + dp.lpad), (void*)(fM), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.fN_ptr + dp.lpad), (void*)(fN), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.cR1_ptr + dp.lpad), (void*)(cR1), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.cR2_ptr + dp.lpad), (void*)(cR2), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.cR4_ptr + dp.lpad), (void*)(cR4), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.tArr_ptr + dp.lpad), (void*)(tArr), dp.nJ * sizeof(float) * dp.nI ).wait();


	/* FIXME: move global variables into data structure */
	/* 1-dim */
	dpct::get_default_queue_wait().memcpy( (void*)(data.cR6_ptr), (void*)(R6), dp.nJ * sizeof(float) ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.cB1_ptr), (void*)(C1), dp.nI * sizeof(float) ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.cB2_ptr), (void*)(C2), dp.nJ * sizeof(float) ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.cB3_ptr), (void*)(C3), dp.nI * sizeof(float) ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(data.cB4_ptr), (void*)(C4), dp.nJ * sizeof(float) ).wait();

	return 0;
}
int CGpuNode::copyFromGPU() {

	Params& dp = data.params;

	//cudaMemcpy2D( hMax, dp.nJ * sizeof(float), data.hMax + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, cudaMemcpyDeviceToHost );
	//cudaMemcpy2D( tArr, dp.nJ * sizeof(float), data.tArr + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, cudaMemcpyDeviceToHost );
	dpct::get_default_queue_wait().memcpy( (void*)(hMax), (void*)(data.hMax_ptr + dp.lpad), dp.nJ * sizeof(float) * dp.nI ).wait();
	dpct::get_default_queue_wait().memcpy( (void*)(tArr), (void*)(data.tArr_ptr + dp.lpad), dp.nJ * sizeof(float) * dp.nI ).wait();


	return 0;
}

int CGpuNode::copyIntermediate() {

	/* ignore copy requests if data already present on CPU side */
	if( copied )
		return 0;

	Params& dp = data.params;

	//cudaMemcpy2D( h, dp.nJ * sizeof(float), data.h + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, cudaMemcpyDeviceToHost );
    memset((void*)h, 0, sizeof(float) * dp.nI * dp.nJ);
	dpct::get_default_queue_wait().memcpy( (void*)(h), (void*)(data.h_ptr + dp.lpad), dp.nJ * sizeof(float) * dp.nI ).wait();

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

		dpct::get_default_queue_wait().memcpy( (void*)(h + idxPOI[n]), (void*)(data.h_ptr + dp.lpad + id), sizeof(float) ).wait();
	}

	return 0;
}

int CGpuNode::freeMem() {

	/* 2-dim */
	cl::sycl::free(data.d_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.h_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.hMax_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.fM_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.fN_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.cR1_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.cR2_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.cR4_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.tArr_ptr, dpct::get_default_queue().get_context());

	/* 1-dim */
	cl::sycl::free(data.cR6_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.cB1_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.cB2_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.cB3_ptr, dpct::get_default_queue().get_context());
	cl::sycl::free(data.cB4_ptr, dpct::get_default_queue().get_context());

	cl::sycl::free(data.g_MinMax, dpct::get_default_queue().get_context());

	float total_dur = 0.f;
	for( int j = 0; j < 5; j++ ) {
		printf_v("Duration %u: %.3f\n", j, dur[j]);
		total_dur += dur[j];
	}
	printf_v("Duration total: %.3f\n",total_dur);

	CArrayNode::freeMem();

	return 0;
}

int CGpuNode::run() try {

	Params& dp = data.params;

	int nThreads = 128;
	int xThreads = 32;
	int yThreads = nThreads / xThreads;

/*
    int xThreads = 32;
    int yThreads = 32;
    int nThreads = xThreads * yThreads;
*/

	int NJ = dp.jMax - dp.jMin + 1;
	int NI = dp.iMax - dp.iMin + 1;
	int xBlocks = ceil((float)NJ / (float)xThreads);
	int yBlocks = ceil((float)NI / (float)yThreads);

	cl::sycl::range<3> threads(xThreads, yThreads, 1);
	cl::sycl::range<3> blocks(xBlocks, yBlocks, 1);

	cl::sycl::range<2> threads2(xThreads, yThreads);
	cl::sycl::range<2> blocks2(xBlocks, yBlocks);

	int nBlocks = ceil((float)std::max(dp.nI, dp.nJ) / (float)nThreads);

	dp.mTime = Par.time;
    KernelData kd = data;

    float* dev_h = data.h_ptr;
    float *dev_hMax = data.hMax_ptr;
    float* dev_d = data.d_ptr;
    float* dev_cR1 = data.cR1_ptr;
    float* dev_cR2 = data.cR2_ptr;
    float* dev_cR4 = data.cR4_ptr;
    float* dev_cR6 = data.cR6_ptr;
    float* dev_fM = data.fM_ptr;
    float* dev_fN = data.fN_ptr;
    float* dev_tArr = data.tArr_ptr;
    float* dev_cB1 = data.cB1_ptr;
    float* dev_cB2 = data.cB2_ptr;
    float* dev_cB3 = data.cB3_ptr;
    float* dev_cB4 = data.cB4_ptr;

	evtStart[0] = std::chrono::high_resolution_clock::now();
	{
	  dpct::get_default_queue_wait().submit(
	    [&](cl::sycl::handler &cgh) {
	      cgh.parallel_for(
	        cl::sycl::nd_range<3>((blocks * threads), threads),
	        [=](cl::sycl::nd_item<3> item_ct1) {
//	        cl::sycl::nd_range<2>((blocks2 * threads2), threads2),
//	        [=](cl::sycl::nd_item<2> item_ct1) {
	          runWaveUpdateKernel(kd, dev_h, dev_hMax, dev_d, dev_cR1, dev_fM, dev_fN, dev_cR6, dev_tArr, item_ct1);
	        });
	    });
	}
//	dpct::get_device_manager().current_device().queues_wait_and_throw();
	evtEnd[0] = std::chrono::high_resolution_clock::now();

	/*
	DPCT1012:24: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
	*/
	evtStart[1] = std::chrono::high_resolution_clock::now();
	{
	  dpct::get_default_queue_wait().submit(
	    [&](cl::sycl::handler &cgh) {
	      cgh.parallel_for(
	        cl::sycl::nd_range<1>((cl::sycl::range<1>(nBlocks) * cl::sycl::range<1>(nThreads)), cl::sycl::range<1>(nThreads)),
	        [=](cl::sycl::nd_item<1> item_ct1) {
	          runWaveBoundaryKernel(kd, dev_h, dev_fM, dev_fN, dev_cB1, dev_cB2, dev_cB3, dev_cB4, item_ct1);
	        });
	    });
	}
//	dpct::get_device_manager().current_device().queues_wait_and_throw();

	evtEnd[1] = std::chrono::high_resolution_clock::now();

	evtStart[2] = std::chrono::high_resolution_clock::now();
	{
	  dpct::get_default_queue_wait().submit(
	    [&](cl::sycl::handler &cgh) {
	      cgh.parallel_for(
	        cl::sycl::nd_range<3>((blocks * threads), threads),
	        [=](cl::sycl::nd_item<3> item_ct1) {
//	        cl::sycl::nd_range<2>((blocks2 * threads2), threads2),
//	        [=](cl::sycl::nd_item<2> item_ct1) {
	          runFluxUpdateKernel(kd, dev_h, dev_d, dev_fM, dev_fN, dev_cR2, dev_cR4, item_ct1);
	        });
	    });
	}

//	dpct::get_device_manager().current_device().queues_wait_and_throw();
	evtEnd[2] = std::chrono::high_resolution_clock::now();

	evtStart[3] = std::chrono::high_resolution_clock::now();
	{
	  dpct::get_default_queue_wait().submit(
	    [&](cl::sycl::handler &cgh) {
	      cgh.parallel_for(
	        cl::sycl::nd_range<1>((cl::sycl::range<1>(nBlocks) * cl::sycl::range<1>(nThreads)), cl::sycl::range<1>(nThreads)),
	        [=](cl::sycl::nd_item<1> item_ct1) {
              runFluxBoundaryKernel( kd, dev_h, dev_fM, dev_fN, dev_cR2, dev_cR4, item_ct1);
	        });
	    });
	}

//	dpct::get_device_manager().current_device().queues_wait_and_throw();
	evtEnd[3] = std::chrono::high_resolution_clock::now();

	/*
	DPCT1012:30: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
	*/
	evtStart[4] = std::chrono::high_resolution_clock::now();
	dpct::get_default_queue_wait().memset( (void*)(data.g_MinMax), 0, 4 * sizeof(*data.g_MinMax) ).wait();
	{
	  dpct::get_default_queue_wait().submit(
	    [&](cl::sycl::handler &cgh) {
	      cgh.parallel_for(
	        cl::sycl::nd_range<1>((cl::sycl::range<1>(nBlocks) * cl::sycl::range<1>(nThreads)), cl::sycl::range<1>(nThreads)),
	        [=](cl::sycl::nd_item<1> item_ct1) {
	          runGridExtendKernel(kd, dev_h, item_ct1);
	        });
	    });
	}

//	dpct::get_device_manager().current_device().queues_wait_and_throw();

	evtEnd[4] = std::chrono::high_resolution_clock::now();

	int MinMax[4] = { 0 };
	dpct::get_default_queue_wait().memcpy( (void*)(&MinMax), (void*)(data.g_MinMax), 4 * sizeof(*data.g_MinMax) ).wait();

	dpct::get_device_manager().current_device().queues_wait_and_throw();

    //std::cout << MinMax[0] << " " << MinMax[1] << " " << MinMax[2] << " " << MinMax[3] << std::endl;
	if( MinMax[0] ) {
	    Imin = dp.iMin = std::max(dp.iMin-1, 2);
    }

	if( MinMax[1] ) {
	    Imax = dp.iMax = std::min(dp.iMax+1, dp.nI-1);
    }

	if( MinMax[2] ) {
	    Jmin = dp.jMin = std::max(dp.jMin-32, 2);
    }

	if( MinMax[3] ) {
	    Jmax = dp.jMax = std::min(dp.jMax+1, dp.nJ-1);
    }

    //std::cout << Imin << " " << Imax << " " << Jmin << " " << Jmax << std::endl;
	for( int j = 0; j < 5; j++ ) {
		dur[j] += std::chrono::duration<float, std::milli>(evtEnd[j] - evtStart[j]).count();
	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
  std::exit(1);
}
