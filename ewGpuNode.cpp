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

CGpuNode::CGpuNode() {

	pitch = 0;
	copied = true;

	char* dev_spec = getenv("EW_CL_DEVICE");
	if (dev_spec) {
		dpct::dev_mgr::instance().select_device(atoi(dev_spec));
	}

	dpct::device_info di;
	auto &dev = dpct::dev_mgr::instance().current_device();
	dev.get_device_info(di);
	std::cout << "running in accellerated mode using " << di.get_name() << std::endl;
	std::cout << "Profling supported: " << dev.get_info<cl::sycl::info::device::queue_profiling>() << std::endl;

	auto kernels = dev.get_info<cl::sycl::info::device::built_in_kernels>();
	std::cout << "Built-in kernels: " << (kernels.size() ? std::to_string(kernels.size()) : "None.") << std::endl;
	for (const auto &k: kernels) {
		std::cout << " - " << k << std::endl;
	}

	/* use new here, otherwise you earn a crash in glibc memory allocation */
	kernel_events = new std::vector<cl::sycl::event>(NUM_KERNELS);

	for( int i = 0; i < NUM_TIMED_KERNELS; i++ ) { dur[i] = 0.0; }

	have_profiling = dev.get_info<cl::sycl::info::device::queue_profiling>() && Par.verbose;
	if (have_profiling) {
		auto enable_list = cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
		queue = new cl::sycl::queue(dev, enable_list);
	} else {
		queue = &dpct::get_default_queue();
	}
}

CGpuNode::~CGpuNode()
{
	/* not really safe, but try to free a possibly manually created queue */
	if (*queue != dpct::get_default_queue()) {
		delete queue;
	}
	delete kernel_events;
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
	data.d = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
	data.h = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
	data.hMax = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
	data.fM = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
	data.fN = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
	data.cR1 = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
	data.cR2 = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
	data.cR4 = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
	data.tArr = (float*) dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
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

	CArrayNode::freeMem();

	return 0;
}

#define INT_CEIL(x, n) ((((x) + (n) - 1) / (n)) * (n))

int CGpuNode::run() {

	Params& dp = data.params;

	int NJ = dp.jMax - dp.jMin + 1;
	int NI = dp.iMax - dp.iMin + 1;

	const cl::sycl::device dev = queue->get_device();
	size_t max_wg_size = dev.get_info<cl::sycl::info::device::max_work_group_size>();

	/* Using max wg size or a preferred wg_size would be better here, but causes runtime errors (CL_OUT_OF_RESOURCES) */
	sycl::range<1> boundary_workgroup_size(max_wg_size); /* div by 7 works as well for CPU, but not by 6 */
	sycl::range<1> boundary_size(INT_CEIL(std::max(dp.nI, dp.nJ), boundary_workgroup_size[0]));

	/* Originally we had n = 128 threads, 32 for x and 128/x = 4 threads, hardcoded. So do this again here, to avoid runtime errors */
	sycl::range<2> compute_wnd_workgroup_size(32, 4);
	sycl::range<2> compute_wnd_size(
		INT_CEIL(NJ, compute_wnd_workgroup_size[0]),
		INT_CEIL(NI, compute_wnd_workgroup_size[1])
	);

	dp.mTime = Par.time;

	auto kernel_data = data;

	kernel_events->at(KERNEL_WAVE_UPDATE) = queue->submit([&](sycl::handler &cgh) {
		cgh.parallel_for(
			sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
			[=](sycl::nd_item<2> item) {
				runWaveUpdateKernel(kernel_data, item);
			});
	});

	kernel_events->at(KERNEL_WAVE_BOUND) = queue->submit([&](sycl::handler &cgh) {
		cgh.depends_on({ kernel_events->at(KERNEL_WAVE_UPDATE) });
		cgh.parallel_for(
			sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](sycl::nd_item<1> item) {
				runWaveBoundaryKernel(kernel_data, item);
			});
	});

	kernel_events->at(KERNEL_FLUX_UPDATE) = queue->submit([&](sycl::handler &cgh) {
		cgh.depends_on({ kernel_events->at(KERNEL_WAVE_BOUND) });
		cgh.parallel_for(
			sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
			[=](sycl::nd_item<2> item) {
				runFluxUpdateKernel(kernel_data, item);
			});
	});

	kernel_events->at(KERNEL_FLUX_BOUND) = queue->submit([&](sycl::handler &cgh) {
		cgh.depends_on({ kernel_events->at(KERNEL_FLUX_UPDATE) });
		cgh.parallel_for(sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](sycl::nd_item<1> item) {
				runFluxBoundaryKernel(kernel_data, item);
			});
	});

	kernel_events->at(KERNEL_MEMSET) = queue->memset(data.g_MinMax, 0, sizeof(sycl::int4));

	kernel_events->at(KERNEL_EXTEND) = queue->submit([&](sycl::handler &cgh) {
		cgh.depends_on({ kernel_events->at(KERNEL_FLUX_BOUND), kernel_events->at(KERNEL_MEMSET) });
		cgh.parallel_for(sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](sycl::nd_item<1> item) {
				runGridExtendKernel(kernel_data, item);
			});
	});

	sycl::int4 MinMax;
	kernel_events->at(KERNEL_EXTEND).wait();
	queue->memcpy(&MinMax, data.g_MinMax, sizeof(sycl::int4)).wait();

	/* TODO: respect alignments from device in window expansion (Preferred work group size multiple ?!) */
	if (static_cast<int>(MinMax.x())) Imin = dp.iMin = std::max(dp.iMin - 1, 2);
	if (static_cast<int>(MinMax.y())) Imax = dp.iMax = std::min(dp.iMax + 1, dp.nI - 1);
	if (static_cast<int>(MinMax.z())) Jmin = dp.jMin = std::max(dp.jMin - 32, 2);
	if (static_cast<int>(MinMax.w())) Jmax = dp.jMax = std::min(dp.jMax + 1, dp.nJ - 1);

	for( int j = 0; have_profiling && j < NUM_TIMED_KERNELS; j++ ) {
		dur[j] += (kernel_events->at(j).get_profiling_info<cl::sycl::info::event_profiling::command_end>()
			- kernel_events->at(j).get_profiling_info<cl::sycl::info::event_profiling::command_submit>()) / 1.0E+6;
	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
