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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ewGpuNode.hpp"
#include "ewCudaKernels.hpp"
#include <cmath>

#include <algorithm>

#define dpct_malloc(ptr, pitch, size_x, size_y)  dpct_malloc((void**) (ptr), pitch, size_x, size_y)

CGpuNode::CGpuNode() {

	pitch = 0;
	copied = true;

    char* dev_spec = getenv("EW_CL_DEVICE");
    if (dev_spec) {
        dpct::dev_mgr::instance().select_device(atoi(dev_spec));
    }

    dpct::device_info di;
    dpct::dev_mgr::instance().current_device().get_device_info(di);
    std::cout << "running in accellerated mode using " << di.get_name() << std::endl;
    dpct::dev_mgr::instance().current_device().queues_wait_and_throw();

    for( int i = 0; i < 5; i++ ) { dur[i] = 0.0; }
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
	/* FIXME: move global variables into data structure */
	dpct::dpct_malloc(&data.d, &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&data.d, &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&data.h, &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&(data.hMax), &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&(data.fM), &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&(data.fN), &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&(data.cR1), &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&(data.cR2), &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&(data.cR4), &pitch, nJ_aligned * sizeof(float), dp.nI);
	dpct::dpct_malloc(&(data.tArr), &pitch, nJ_aligned * sizeof(float), dp.nI);
    /* TODO: cR3, cR5 for coriolis */

	/* 1-dim */
    (data.cR6) = (float *)sycl::malloc_device(dp.nJ * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
	(data.cB1) = (float *)sycl::malloc_device(dp.nI * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
	(data.cB2) = (float *)sycl::malloc_device(dp.nJ * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
	(data.cB3) = (float *)sycl::malloc_device(dp.nI * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
	(data.cB4) = (float *)sycl::malloc_device(dp.nJ * sizeof(float), dpct::get_current_device(), dpct::get_default_context());

	(data.g_MinMax) = (sycl::int4 *)sycl::malloc_device( sizeof(sycl::int4), dpct::get_current_device(), dpct::get_default_context());

	/* TODO: make sure that pitch is a multiple of 4 and the same for each cudaMallocPitch() call */
	dp.pI = pitch / sizeof(float);

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
	dpct::dpct_memcpy(data.d + dp.lpad, pitch, d, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
	dpct::dpct_memcpy(data.h + dp.lpad, pitch, h, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
	dpct::dpct_memcpy(data.hMax + dp.lpad, pitch, hMax, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
	dpct::dpct_memcpy(data.fM + dp.lpad, pitch, fM, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
	dpct::dpct_memcpy(data.fN + dp.lpad, pitch, fN, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
	dpct::dpct_memcpy(data.cR1 + dp.lpad, pitch, cR1, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
	dpct::dpct_memcpy(data.cR2 + dp.lpad, pitch, cR2, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
	dpct::dpct_memcpy(data.cR4 + dp.lpad, pitch, cR4, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
	dpct::dpct_memcpy(data.tArr + dp.lpad, pitch, tArr, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);

	/* FIXME: move global variables into data structure */
	/* 1-dim */
	sycl::queue &q_ct0 = dpct::get_default_queue();
	q_ct0.wait();
	q_ct0.memcpy(data.cR6, R6, dp.nJ * sizeof(float)).wait();
	q_ct0.memcpy(data.cB1, C1, dp.nI * sizeof(float)).wait();
	q_ct0.memcpy(data.cB2, C2, dp.nJ * sizeof(float)).wait();
	q_ct0.memcpy(data.cB3, C3, dp.nI * sizeof(float)).wait();
	q_ct0.memcpy(data.cB4, C4, dp.nJ * sizeof(float)).wait();

	return 0;
}

int CGpuNode::copyFromGPU() {

	Params& dp = data.params;

	dpct::dpct_memcpy(hMax, dp.nJ * sizeof(float), data.hMax + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, dpct::device_to_host);
	dpct::dpct_memcpy(tArr, dp.nJ * sizeof(float), data.tArr + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, dpct::device_to_host);

	return 0;
}

int CGpuNode::copyIntermediate() {

	/* ignore copy requests if data already present on CPU side */
	if( copied )
		return 0;

	Params& dp = data.params;

	dpct::dpct_memcpy(h, dp.nJ * sizeof(float), data.h + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI, dpct::device_to_host);

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

		dpct::get_default_queue().memcpy(h + idxPOI[n], data.h + dp.lpad + id, sizeof(float)) .wait();
	}

	return 0;
}

int CGpuNode::freeMem() {

	/* 2-dim */
	sycl::free(data.d, dpct::get_default_context());
	sycl::free(data.h, dpct::get_default_context());
	sycl::free(data.hMax, dpct::get_default_context());
	sycl::free(data.fM, dpct::get_default_context());
	sycl::free(data.fN, dpct::get_default_context());
	sycl::free(data.cR1, dpct::get_default_context());
	sycl::free(data.cR2, dpct::get_default_context());
	sycl::free(data.cR4, dpct::get_default_context());
	sycl::free(data.tArr, dpct::get_default_context());

	/* 1-dim */
	sycl::free(data.cR6, dpct::get_default_context());
	sycl::free(data.cB1, dpct::get_default_context());
	sycl::free(data.cB2, dpct::get_default_context());
	sycl::free(data.cB3, dpct::get_default_context());
	sycl::free(data.cB4, dpct::get_default_context());

	sycl::free(data.g_MinMax, dpct::get_default_context());

	float total_dur = 0.f;
	for( int j = 0; j < 5; j++ ) {
		printf_v("Duration %u: %.3f\n", j, dur[j]);
		total_dur += dur[j];
	}
	printf_v("Duration total: %.3f\n",total_dur);

	CArrayNode::freeMem();

	return 0;
}

#ifdef TIMING
#define SYNC      do { dpct::get_current_device().queues_wait_and_throw() } while (0)
#else
#define SYNC
#endif

int CGpuNode::run() {

	Params& dp = data.params;

	int nThreads = 128;
	int xThreads = 32;
	int yThreads = nThreads / xThreads;

	int NJ = dp.jMax - dp.jMin + 1;
	int NI = dp.iMax - dp.iMin + 1;
	int xBlocks = ceil((float)NJ / (float)xThreads);
	int yBlocks = ceil((float)NI / (float)yThreads);

	sycl::range<2> threads(xThreads, yThreads);
	sycl::range<2> blocks(xBlocks, yBlocks);

	int nBlocks = ceil((float)std::max(dp.nI, dp.nJ) / (float)nThreads);

	dp.mTime = Par.time;

	auto data_ct0 = data;
	auto dpct_global_range = blocks * threads;

	evtStart[0] = std::chrono::high_resolution_clock::now();
	dpct::get_default_queue().submit([&](sycl::handler &cgh) {

		cgh.parallel_for(
			sycl::nd_range<2>(
				sycl::range<2>(dpct_global_range.get(0), dpct_global_range.get(1)),
				sycl::range<2>(threads.get(0), threads.get(1))),
			[=](sycl::nd_item<2> item_ct1) {
				runWaveUpdateKernel(data_ct0, item_ct1);
			});
	});
	SYNC;
	evtEnd[0] = std::chrono::high_resolution_clock::now();

	evtStart[1] = std::chrono::high_resolution_clock::now();
	dpct::get_default_queue().submit([&](sycl::handler &cgh) {

		cgh.parallel_for(
			sycl::nd_range<1>(
				sycl::range<1>(nBlocks) * sycl::range<1>(nThreads),
				sycl::range<1>(nThreads)),
			[=](sycl::nd_item<1> item_ct1) {
				runWaveBoundaryKernel(data_ct0, item_ct1);
			});
	});
	SYNC;
	evtEnd[1] = std::chrono::high_resolution_clock::now();

	evtStart[2] = std::chrono::high_resolution_clock::now();
	dpct::get_default_queue().submit([&](sycl::handler &cgh) {

		cgh.parallel_for(
			sycl::nd_range<2>(
				sycl::range<2>(dpct_global_range.get(0), dpct_global_range.get(1)),
				sycl::range<2>(threads.get(0), threads.get(1))),
			[=](sycl::nd_item<2> item_ct1) {
				runFluxUpdateKernel(data_ct0, item_ct1);
			});
	});
	SYNC;
	evtEnd[2] = std::chrono::high_resolution_clock::now();

	evtStart[3] = std::chrono::high_resolution_clock::now();
	dpct::get_default_queue().submit([&](sycl::handler &cgh) {

		cgh.parallel_for(sycl::nd_range<1>(
				sycl::range<1>(nBlocks) * sycl::range<1>(nThreads),
				sycl::range<1>(nThreads)),
			[=](sycl::nd_item<1> item_ct1) {
				runFluxBoundaryKernel(data_ct0, item_ct1);
			});
	});
	SYNC;
	evtEnd[3] = std::chrono::high_resolution_clock::now();

	evtStart[4] = std::chrono::high_resolution_clock::now();
	dpct::get_default_queue().memset(data.g_MinMax, 0, sizeof(sycl::int4)).wait();
	dpct::get_default_queue().submit([&](sycl::handler &cgh) {

		cgh.parallel_for(sycl::nd_range<1>(
				sycl::range<1>(nBlocks) * sycl::range<1>(nThreads),
				sycl::range<1>(nThreads)),
			[=](sycl::nd_item<1> item_ct1) {
				runGridExtendKernel(data_ct0, item_ct1);
			});
	});
	SYNC;
	evtEnd[4] = std::chrono::high_resolution_clock::now();

	sycl::int4 MinMax;
	dpct::get_default_queue().memcpy(&MinMax, data.g_MinMax, sizeof(sycl::int4)).wait();
	dpct::get_current_device().queues_wait_and_throw();

	if (static_cast<int>(MinMax.x())) Imin = dp.iMin = std::max(dp.iMin - 1, 2);
	if (static_cast<int>(MinMax.y())) Imax = dp.iMax = std::min(dp.iMax + 1, dp.nI - 1);
	if (static_cast<int>(MinMax.z())) Jmin = dp.jMin = std::max(dp.jMin - 32, 2);
	if (static_cast<int>(MinMax.w())) Jmax = dp.jMax = std::min(dp.jMax + 1, dp.nJ - 1);

	for( int j = 0; j < 5; j++ ) {
		dur[j] += std::chrono::duration<float, std::milli>(evtEnd[j] - evtStart[j]).count();
	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
