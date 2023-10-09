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

#include <sycl/sycl.hpp>
#include "ewGpuNode.hpp"
#include "ewKernels.sycl.hpp"
#include <cmath>

#include <vector>

#ifdef EW_KERNEL_DURATION_CHECK
#include <limits>
#endif

/* memory helpers, inspired by dpct headers */
#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))

namespace zib {
	namespace sycl {
		static inline void *malloc_pitch(size_t &pitch, size_t x, size_t y, cl::sycl::queue &q)
		{
			pitch = PITCH_DEFAULT_ALIGN(x);
			void *retval = cl::sycl::malloc_device(pitch * y, q.get_device(), q.get_context());

			if (!retval) {
				throw std::runtime_error("Could not allocate 2D device memory");
			}

			return retval;
		}

		static inline std::vector<cl::sycl::event> memcpy(
				cl::sycl::queue &q, void *to_ptr, const void *from_ptr,
				cl::sycl::range<3> to_range, cl::sycl::range<3> from_range,
				cl::sycl::id<3> to_id, cl::sycl::id<3> from_id,
				cl::sycl::range<3> size) {
			std::vector<cl::sycl::event> event_list;

			size_t to_slice = to_range.get(1) * to_range.get(0),
				from_slice = from_range.get(1) * from_range.get(0);
			unsigned char *to_surface = (unsigned char *)to_ptr +
				to_id.get(2) * to_slice + to_id.get(1) * to_range.get(0) + to_id.get(0);
			const unsigned char *from_surface = (const unsigned char *)from_ptr + from_id.get(2) * from_slice +
				from_id.get(1) * from_range.get(0) + from_id.get(0);

			for (size_t z = 0; z < size.get(2); ++z) {
				unsigned char *to_ptr = to_surface;
				const unsigned char *from_ptr = from_surface;
				for (size_t y = 0; y < size.get(1); ++y) {
					event_list.push_back(q.memcpy(to_ptr, from_ptr, size.get(0)));
					to_ptr += to_range.get(0);
					from_ptr += from_range.get(0);
				}

				to_surface += to_slice;
				from_surface += from_slice;
			}
			return event_list;
		}

		static inline std::vector<cl::sycl::event> memcpy(
				cl::sycl::queue &q,
				void *to_ptr, size_t to_pitch,
				const void *from_ptr, size_t from_pitch,
				size_t x, size_t y)
		{
			return memcpy(q, to_ptr, from_ptr, cl::sycl::range<3>(to_pitch, y, 1),
				cl::sycl::range<3>(from_pitch, y, 1),
				cl::sycl::id<3>(0, 0, 0), cl::sycl::id<3>(0, 0, 0),
				cl::sycl::range<3>(x, y, 1));
		}
	}
}

CGpuNode::CGpuNode() {

	pitch = 0;
	copied = true;

	default_queue = new cl::sycl::queue();
	const auto &dev = default_queue->get_device();

	std::cout << "Selected device: " << dev.get_info<cl::sycl::info::device::name>() << std::endl;
	std::cout << "Profiling supported: " << dev.get_info<cl::sycl::info::device::queue_profiling>() << std::endl;
	std::cout << "Maximum Work group size: " << dev.get_info<cl::sycl::info::device::max_work_group_size>() << std::endl;
	std::cout << "USM explicit allocations supported: " << dev.has(cl::sycl::aspect::usm_device_allocations) << std::endl;

	if (!dev.has(cl::sycl::aspect::usm_device_allocations)) {
		throw std::runtime_error("Device does not support USM explicit allications.");
	}

	auto kernels = dev.get_info<cl::sycl::info::device::built_in_kernels>();
	std::cout << "Built-in kernels: " << (kernels.size() ? std::to_string(kernels.size()) : "None.") << std::endl;
	for (const auto &k: kernels) {
		std::cout << " - " << k << std::endl;
	}

	have_profiling = dev.get_info<cl::sycl::info::device::queue_profiling>() && Par.verbose;

	if (have_profiling) {
#ifndef __HIPSYCL__
		auto enable_list = cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
		queue = new cl::sycl::queue(default_queue->get_context(), dev, enable_list);
		std::cout << "per-kernel profiling activated" << std::endl;
#else
		std::cout << "warning: profiling requested, but not supported by hipSYCL" << std::endl;
		have_profiling = false;
#endif
	} else {
		queue = default_queue;
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

    dumpProfilingData();
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
	data.d = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	data.h = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	data.hMax = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	data.fM = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	data.fN = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	data.cR1 = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	data.cR2 = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	data.cR4 = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	data.tArr = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, *queue);
	/* TODO: cR3, cR5 for coriolis */

	/* 1-dim */
	data.cR6 = cl::sycl::malloc_device<float>(dp.nJ, *queue);
	data.cB1 = cl::sycl::malloc_device<float>(dp.nI, *queue);
	data.cB2 = cl::sycl::malloc_device<float>(dp.nJ, *queue);
	data.cB3 = cl::sycl::malloc_device<float>(dp.nI, *queue);
	data.cB4 = cl::sycl::malloc_device<float>(dp.nJ, *queue);

	data.g_MinMax = cl::sycl::malloc_device<cl::sycl::int4>(1, *queue);

	queue->wait_and_throw();

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
	zib::sycl::memcpy(*queue, data.d + dp.lpad, pitch, d, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, data.h + dp.lpad, pitch, h, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, data.hMax + dp.lpad, pitch, hMax, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, data.fM + dp.lpad, pitch, fM, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, data.fN + dp.lpad, pitch, fN, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, data.cR1 + dp.lpad, pitch, cR1, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, data.cR2 + dp.lpad, pitch, cR2, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, data.cR4 + dp.lpad, pitch, cR4, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, data.tArr + dp.lpad, pitch, tArr, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);

	/* FIXME: move global variables into data structure */
	/* 1-dim */
	queue->memcpy(data.cR6, R6, dp.nJ * sizeof(float));
	queue->memcpy(data.cB1, C1, dp.nI * sizeof(float));
	queue->memcpy(data.cB2, C2, dp.nJ * sizeof(float));
	queue->memcpy(data.cB3, C3, dp.nI * sizeof(float));
	queue->memcpy(data.cB4, C4, dp.nJ * sizeof(float));

	queue->wait_and_throw();

	return 0;
}

int CGpuNode::copyFromGPU() {

	Params& dp = data.params;

	zib::sycl::memcpy(*queue, hMax, dp.nJ * sizeof(float), data.hMax + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(*queue, tArr, dp.nJ * sizeof(float), data.tArr + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI);

	queue->wait_and_throw();

	return 0;
}

int CGpuNode::copyIntermediate() {

	/* ignore copy requests if data already present on CPU side */
	if( copied )
		return 0;

	Params& dp = data.params;

	zib::sycl::memcpy(*queue, h, dp.nJ * sizeof(float), data.h + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI);
	queue->wait_and_throw();

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
	/* 2-dim */
	cl::sycl::free(data.d, *default_queue);
	cl::sycl::free(data.h, *default_queue);
	cl::sycl::free(data.hMax, *default_queue);
	cl::sycl::free(data.fM, *default_queue);
	cl::sycl::free(data.fN, *default_queue);
	cl::sycl::free(data.cR1, *default_queue);
	cl::sycl::free(data.cR2, *default_queue);
	cl::sycl::free(data.cR4, *default_queue);
	cl::sycl::free(data.tArr, *default_queue);

	/* 1-dim */
	cl::sycl::free(data.cR6, *default_queue);
	cl::sycl::free(data.cB1, *default_queue);
	cl::sycl::free(data.cB2, *default_queue);
	cl::sycl::free(data.cB3, *default_queue);
	cl::sycl::free(data.cB4, *default_queue);

	cl::sycl::free(data.g_MinMax, *default_queue);

	CArrayNode::freeMem();

	return 0;
}

#define INT_CEIL(x, n) ((((x) + (n) - 1) / (n)) * (n))

int CGpuNode::run() {
	Params& dp = data.params;

	int NJ = dp.jMax - dp.jMin + 1;
	int NI = dp.iMax - dp.iMin + 1;

	size_t max_wg_size = queue->get_device().get_info<cl::sycl::info::device::max_work_group_size>();

	cl::sycl::range<1> boundary_workgroup_size(max_wg_size);
	cl::sycl::range<1> boundary_size(INT_CEIL(std::max(dp.nI, dp.nJ), boundary_workgroup_size[0]));

#if 0
#if defined(SYCL_LANGUAGE_VERSION) && defined (__INTEL_LLVM_COMPILER)
	/* For Intel, prevent the nd_range_error: "Non-uniform work-groups are not supported by the target device -54 (CL_INVALID_WORK_GROUP_SIZE))". */
	/* Originally we had n = 128 threads, 32 for x and 128/x = 4 threads, hardcoded in the CUDA code. */
	cl::sycl::range<2> compute_wnd_workgroup_size(4, 32);
#else
	cl::sycl::range<2> compute_wnd_workgroup_size(32, 32);
#endif
#else
	cl::sycl::range<2> compute_wnd_workgroup_size(Par.threads_x, Par.threads_y);
#endif

	cl::sycl::range<2> compute_wnd_size(
		INT_CEIL(NI, compute_wnd_workgroup_size[0]),
		INT_CEIL(NJ, compute_wnd_workgroup_size[1])
	);

	dp.mTime = Par.time;

	std::array<cl::sycl::event, NUM_KERNELS> kernel_events;

	kernel_events[KERNEL_WAVE_UPDATE] = queue->submit([&](cl::sycl::handler &cgh) {
	    auto kernel_data = data;

		cgh.parallel_for(
			cl::sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
			[=](cl::sycl::nd_item<2> item) {
				waveUpdate(kernel_data, item);
			});
	});

	kernel_events[KERNEL_WAVE_BOUND] = queue->submit([&](cl::sycl::handler &cgh) {
		cgh.depends_on({ kernel_events[KERNEL_WAVE_UPDATE] });
	    auto kernel_data = data;
		cgh.parallel_for(
			cl::sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](cl::sycl::nd_item<1> item) {
				waveBoundary(kernel_data, item);
			});
	});

	kernel_events[KERNEL_FLUX_UPDATE] = queue->submit([&](cl::sycl::handler &cgh) {
		cgh.depends_on({ kernel_events[KERNEL_WAVE_BOUND] });
	    auto kernel_data = data;
		cgh.parallel_for(
			cl::sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
			[=](cl::sycl::nd_item<2> item) {
				fluxUpdate(kernel_data, item);
			});
	});

	kernel_events[KERNEL_FLUX_BOUND] = queue->submit([&](cl::sycl::handler &cgh) {
		cgh.depends_on({ kernel_events[KERNEL_FLUX_UPDATE] });
	    auto kernel_data = data;
		cgh.parallel_for(cl::sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](cl::sycl::nd_item<1> item) {
				fluxBoundary(kernel_data, item);
			});
	});

	kernel_events[KERNEL_MEMSET] = queue->memset(data.g_MinMax, 0, sizeof(cl::sycl::int4));

	kernel_events[KERNEL_EXTEND] = queue->submit([&](cl::sycl::handler &cgh) {
		cgh.depends_on({ kernel_events[KERNEL_FLUX_BOUND], kernel_events[KERNEL_MEMSET] });
	    auto kernel_data = data;
		cgh.parallel_for(cl::sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
			[=](cl::sycl::nd_item<1> item) {
				gridExtend(kernel_data, item);
			});
	});
	kernel_events[KERNEL_EXTEND].wait();

	cl::sycl::int4 MinMax;
	kernel_events[KERNEL_MEMCPY] = queue->memcpy(&MinMax, data.g_MinMax, sizeof(cl::sycl::int4));
	kernel_events[KERNEL_MEMCPY].wait();

	/* TODO: respect alignments from device in window expansion (Preferred work group size multiple ?!) */
	if (MinMax.x()) Imin = dp.iMin = std::max(dp.iMin - 1, 2);
	if (MinMax.y()) Imax = dp.iMax = std::min(dp.iMax + 1, dp.nI - 1);
	if (MinMax.z()) Jmin = dp.jMin = std::max(dp.jMin - MEM_ALIGN, 2);
	if (MinMax.w()) Jmax = dp.jMax = std::min(dp.jMax + 1, dp.nJ - 1);

	for( int j = 0; have_profiling && j < NUM_TIMED_KERNELS; j++ ) {
		float duration = (kernel_events[j].get_profiling_info<cl::sycl::info::event_profiling::command_end>()
			- kernel_events[j].get_profiling_info<cl::sycl::info::event_profiling::command_start>()) / 1.0E+6;
#ifdef EW_KERNEL_DURATION_CHECK
		// Intel/Codeplay's LLVM runtime for AMD appear to have problem with short lived kernels (?), so do
		// a tiny check here if requested (limit is chosen arbitrary, but if absent, numbers become really large)
		if (duration < std::numeric_limits<int>::max()) {
			kernel_duration[j] += duration;
		}
#else
		kernel_duration[j] += duration;
#endif

	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
