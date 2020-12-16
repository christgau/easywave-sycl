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
#include <iomanip>
#include <numeric>

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
	std::cout << "Profiling supported: " << dev.get_info<cl::sycl::info::device::queue_profiling>() << std::endl;

	auto kernels = dev.get_info<cl::sycl::info::device::built_in_kernels>();
	std::cout << "Built-in kernels: " << (kernels.size() ? std::to_string(kernels.size()) : "None.") << std::endl;
	for (const auto &k: kernels) {
		std::cout << " - " << k << std::endl;
	}

	for( int i = 0; i < NUM_TIMED_KERNELS; i++ ) { dur[i] = 0.0; }

	default_queue = &dev.default_queue();
	have_profiling = dev.get_info<cl::sycl::info::device::queue_profiling>() && Par.verbose;

	if (have_profiling) {
		auto enable_list = cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
		queue = new cl::sycl::queue(default_queue->get_context(), dev, enable_list);
		std::cout << "per-kernel profiling activated" << std::endl;
	} else {
		queue = &dev.default_queue();
	}

	for (auto &kd: kernel_duration) {
		kd = 0;
	}
}

CGpuNode::~CGpuNode()
{
	if (have_profiling) {
		delete queue;
	}

    if (have_profiling) {
        /* all kernel timings */
        auto total = std::accumulate(kernel_duration.begin(), kernel_duration.end(), 0.0);

        for (int i = 0; i < kernel_duration.size(); i++) {
            std::cout << "timing " << i << " (" << kernel_names[i] << "): "
                << std::fixed << std::setprecision(3) << kernel_duration[i] << " ms ("
                << std::fixed << std::setprecision(3) << (kernel_duration[i] / total) << ")" << std::endl;
        }
        std::cout << "timing_total: " << total << std::endl;

        /* backwards compatibility */
        for (int i = 0; i < NUM_DURATIONS; i++) {
            dur[i] = kernel_duration[i];
        }
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
	data.cR6 = sycl::malloc_device<float>(dp.nJ, *queue);
	data.cB1 = sycl::malloc_device<float>(dp.nI, *queue);
	data.cB2 = sycl::malloc_device<float>(dp.nJ, *queue);
	data.cB3 = sycl::malloc_device<float>(dp.nI, *queue);
	data.cB4 = sycl::malloc_device<float>(dp.nJ, *queue);

	data.g_MinMax = sycl::malloc_device<sycl::int4>(1, *queue);

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
	queue->memcpy(data.cR6, R6, dp.nJ * sizeof(float)).wait();
	queue->memcpy(data.cB1, C1, dp.nI * sizeof(float)).wait();
	queue->memcpy(data.cB2, C2, dp.nJ * sizeof(float)).wait();
	queue->memcpy(data.cB3, C3, dp.nI * sizeof(float)).wait();
	queue->memcpy(data.cB4, C4, dp.nJ * sizeof(float)).wait();

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

		queue->memcpy(h + idxPOI[n], data.h + dp.lpad + id, sizeof(float)).wait();
	}

	return 0;
}

int CGpuNode::freeMem() {
	dpct::device_ext &dev = dpct::get_current_device();

	/* 2-dim */
	sycl::free(data.d, *default_queue);
	sycl::free(data.h, *default_queue);
	sycl::free(data.hMax, *default_queue);
	sycl::free(data.fM, *default_queue);
	sycl::free(data.fN, *default_queue);
	sycl::free(data.cR1, *default_queue);
	sycl::free(data.cR2, *default_queue);
	sycl::free(data.cR4, *default_queue);
	sycl::free(data.tArr, *default_queue);

	/* 1-dim */
	sycl::free(data.cR6, *default_queue);
	sycl::free(data.cB1, *default_queue);
	sycl::free(data.cB2, *default_queue);
	sycl::free(data.cB3, *default_queue);
	sycl::free(data.cB4, *default_queue);

	sycl::free(data.g_MinMax, *default_queue);

	CArrayNode::freeMem();

	return 0;
}

#define INT_CEIL(x, n) ((((x) + (n) - 1) / (n)) * (n))

int CGpuNode::run() {
	dpct::device_ext &dev = dpct::get_current_device();
	sycl::queue &q = *queue;

	Params& dp = data.params;

	int NJ = dp.jMax - dp.jMin + 1;
	int NI = dp.iMax - dp.iMin + 1;

	size_t max_wg_size = dev.get_info<cl::sycl::info::device::max_work_group_size>();

	/* Using max wg size or a preferred wg_size would be better here, but causes runtime errors (CL_OUT_OF_RESOURCES) */
	sycl::range<1> boundary_workgroup_size(max_wg_size); /* div by 7 works as well for CPU, but not by 6 */
	sycl::range<1> boundary_size(INT_CEIL(std::max(dp.nI, dp.nJ), boundary_workgroup_size[0]));

	/* Originally we had n = 128 threads, 32 for x and 128/x = 4 threads, hardcoded. So do this again here, to avoid runtime errors */
	sycl::range<2> compute_wnd_workgroup_size(4, 32);
	sycl::range<2> compute_wnd_size(
		INT_CEIL(NI, compute_wnd_workgroup_size[0]),
		INT_CEIL(NJ, compute_wnd_workgroup_size[1])
	);

	dp.mTime = Par.time;

	std::array<cl::sycl::event, NUM_KERNELS> kernel_events;

	kernel_events[KERNEL_WAVE_UPDATE] = q.submit([&](sycl::handler &cgh) {
	    auto kernel_data = data;

		cgh.parallel_for(
			sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
			[=](sycl::nd_item<2> item) {
				runWaveUpdateKernel(kernel_data, item);
			});
	});

	kernel_events[KERNEL_WAVE_BOUND] = q.submit([&](sycl::handler &cgh) {
		cgh.depends_on({ kernel_events[KERNEL_WAVE_UPDATE] });
	    auto kernel_data = data;
		cgh.parallel_for(
			sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](sycl::nd_item<1> item) {
				runWaveBoundaryKernel(kernel_data, item);
			});
	});

	kernel_events[KERNEL_FLUX_UPDATE] = q.submit([&](sycl::handler &cgh) {
		cgh.depends_on({ kernel_events[KERNEL_WAVE_BOUND] });
	    auto kernel_data = data;
		cgh.parallel_for(
			sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
			[=](sycl::nd_item<2> item) {
				runFluxUpdateKernel(kernel_data, item);
			});
	});

	kernel_events[KERNEL_FLUX_BOUND] = q.submit([&](sycl::handler &cgh) {
		cgh.depends_on({ kernel_events[KERNEL_FLUX_UPDATE] });
	    auto kernel_data = data;
		cgh.parallel_for(sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](sycl::nd_item<1> item) {
				runFluxBoundaryKernel(kernel_data, item);
			});
	});

	kernel_events[KERNEL_MEMSET] = q.memset(data.g_MinMax, 0, sizeof(sycl::int4));

	kernel_events[KERNEL_EXTEND] = q.submit([&](sycl::handler &cgh) {
		cgh.depends_on({ kernel_events[KERNEL_FLUX_BOUND], kernel_events[KERNEL_MEMSET] });
	    auto kernel_data = data;
		cgh.parallel_for(sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](sycl::nd_item<1> item) {
				runGridExtendKernel(kernel_data, item);
			});
	});
	kernel_events[KERNEL_EXTEND].wait();

	sycl::int4 MinMax;
	kernel_events[KERNEL_MEMCPY] = q.memcpy(&MinMax, data.g_MinMax, sizeof(sycl::int4));
	kernel_events[KERNEL_MEMCPY].wait();
	dev.queues_wait_and_throw();

	/* TODO: respect alignments from device in window expansion (Preferred work group size multiple ?!) */
	if (MinMax.x()) Imin = dp.iMin = std::max(dp.iMin - 1, 2);
	if (MinMax.y()) Imax = dp.iMax = std::min(dp.iMax + 1, dp.nI - 1);
	if (MinMax.z()) Jmin = dp.jMin = std::max(dp.jMin - 32, 2);
	if (MinMax.w()) Jmax = dp.jMax = std::min(dp.jMax + 1, dp.nJ - 1);

	for( int j = 0; have_profiling && j < NUM_TIMED_KERNELS; j++ ) {
		kernel_duration[j] += (kernel_events[j].get_profiling_info<cl::sycl::info::event_profiling::command_end>()
			- kernel_events[j].get_profiling_info<cl::sycl::info::event_profiling::command_start>()) / 1.0E+6;
	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
