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
#include "ewGpuNode.hpp"
#include "ewCudaKernels.hpp"
#include <cmath>

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <vector>

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

    template<typename T>
    bool in_vec(const T x, const std::vector<T> v)
    {
        return std::find(v.begin(), v.end(), x) != v.end();
    }
}

CGpuNode::CGpuNode() :
    pitch(0),
    copied(true),
    root_device({ sycl::default_selector() }),
    kernel_duration({0})
{
    /* print some information about the device */
	std::cout << "Selected device: " << root_device.get_info<cl::sycl::info::device::name>() << std::endl
        << "Max Subdevice Count: " << root_device.get_info<sycl::info::device::partition_max_sub_devices>() << std::endl
	    << "Profiling supported: " << root_device.get_info<cl::sycl::info::device::queue_profiling>() << std::endl
    	<< "Maximum Work group size: " << root_device.get_info<cl::sycl::info::device::max_work_group_size>() << std::endl
	    << "USM explicit allocations supported: " << root_device.has(cl::sycl::aspect::usm_device_allocations) << std::endl;

	if (!root_device.has(cl::sycl::aspect::usm_device_allocations)) {
		throw std::runtime_error("Device does not support USM explicit allications.");
	}


	auto kernels = root_device.get_info<cl::sycl::info::device::built_in_kernels>();
	std::cout << "Built-in kernels: " << (kernels.size() ? std::to_string(kernels.size()) : "None.") << std::endl;
	for (const auto &k: kernels) {
		std::cout << " - " << k << std::endl;
	}

    /* create queues with profiling information enabled if required */
	is_profiling_enabled = root_device.get_info<cl::sycl::info::device::queue_profiling>() && Par.verbose;
#ifdef __HIPSYCL__
    if (is_profiling_enabled) {
		std::cout << "warning: profiling requested, but not supported by hipSYCL" << std::endl;
		is_profiling_enabled = false;
    }
#endif

    cl::sycl::property_list prop_list = is_profiling_enabled ? cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()} : cl::sycl::property_list{};
    if (is_profiling_enabled) {
		std::cout << "per-kernel profiling activated" << std::endl;
    }

    root_queue = sycl::queue(root_device, prop_list);

    use_subdevices = getenv("EW_USE_SUBDEVS") != nullptr;
    if (getenv("EW_MAX_SUBDEVS") != nullptr) {
        max_subdev_count = std::stoi(getenv("EW_MAX_SUBDEVS"));
        if (max_subdev_count < 1) {
            throw std::runtime_error("invalid subdevice count " + std::string(getenv("EW_MAX_SUBDEVS")));
        }
    } else {
        max_subdev_count = 0;
    }

    /* create queues for subdevices or use queue of root device if not subdevices are available */
    size_t num_sub_devs = root_device.get_info<sycl::info::device::partition_max_sub_devices>();
    if (num_sub_devs > 0 && use_subdevices) {
        std::vector<sycl::info::partition_property> part_props = root_device.get_info<sycl::info::device::partition_properties>();
        std::vector<sycl::device> subdevs;

        /* prefer equal partitioning over affinity/numa partitioning */
        if (zib::in_vec(sycl::info::partition_property::partition_equally, part_props)) {
            size_t max_compute_units = root_device.get_info<sycl::info::device::max_compute_units>();
            size_t part_count = std::max(max_compute_units / num_sub_devs, 1UL);

            std::cout << "partition_equally to " << part_count << std::endl;
            subdevs = root_device.create_sub_devices<sycl::info::partition_property::partition_equally>(part_count);
        } else if (zib::in_vec(sycl::info::partition_property::partition_by_affinity_domain, part_props)) {
            std::cout << "partition by affinity domain" << std::endl;
            subdevs = root_device.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::next_partitionable);
        } else {
            throw std::runtime_error("Unsupported partitioning type.");
        }

        if (max_subdev_count > 0 && subdevs.size() > max_subdev_count) {
            subdevs.resize(max_subdev_count);
        }

        common_ctx = cl::sycl::context(subdevs);

        for (const auto &subdev: subdevs) {
            queues.push_back(sycl::queue(common_ctx, subdev, prop_list));
        }
    } else {
        common_ctx = root_queue.get_context();
        queues.push_back(sycl::queue(root_device, prop_list));
    }

    std::cout << "using " << queues.size() << " queues" << std::endl;

	for( int i = 0; i < NUM_TIMED_KERNELS; i++ ) { dur[i] = 0.0; }
}

CGpuNode::~CGpuNode()
{
    if (is_profiling_enabled) {
        /* all kernel timings */
        auto total = std::accumulate(kernel_duration.begin(), kernel_duration.end(), 0.0);

        for (int i = 0; i < kernel_duration.size(); i++) {
            std::cout << "runtime kernel " << i << " (" << kernel_names[i] << "): "
                << std::fixed << std::setprecision(3) << kernel_duration[i] << " ms ("
                << std::fixed << std::setprecision(3) << (kernel_duration[i] / total) << ")" << std::endl;
        }
        std::cout << "kernels total: " << total << std::endl;
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
	data.d = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	data.h = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	data.hMax = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	data.fM = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	data.fN = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	data.cR1 = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	data.cR2 = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	data.cR4 = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	data.tArr = (float*) zib::sycl::malloc_pitch(pitch, nJ_aligned * sizeof(float), dp.nI, root_queue);
	/* TODO: cR3, cR5 for coriolis */

	/* 1-dim */
	data.cR6 = cl::sycl::malloc_device<float>(dp.nJ, root_queue);
	data.cB1 = cl::sycl::malloc_device<float>(dp.nI, root_queue);
	data.cB2 = cl::sycl::malloc_device<float>(dp.nJ, root_queue);
	data.cB3 = cl::sycl::malloc_device<float>(dp.nI, root_queue);
	data.cB4 = cl::sycl::malloc_device<float>(dp.nJ, root_queue);

	data.g_MinMax = cl::sycl::malloc_device<cl::sycl::int4>(1, root_queue);

	root_queue.wait_and_throw();

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
	zib::sycl::memcpy(root_queue, data.d + dp.lpad, pitch, d, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, data.h + dp.lpad, pitch, h, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, data.hMax + dp.lpad, pitch, hMax, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, data.fM + dp.lpad, pitch, fM, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, data.fN + dp.lpad, pitch, fN, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, data.cR1 + dp.lpad, pitch, cR1, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, data.cR2 + dp.lpad, pitch, cR2, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, data.cR4 + dp.lpad, pitch, cR4, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, data.tArr + dp.lpad, pitch, tArr, dp.nJ * sizeof(float), dp.nJ * sizeof(float), dp.nI);

	/* FIXME: move global variables into data structure */
	/* 1-dim */
	root_queue.memcpy(data.cR6, R6, dp.nJ * sizeof(float));
	root_queue.memcpy(data.cB1, C1, dp.nI * sizeof(float));
	root_queue.memcpy(data.cB2, C2, dp.nJ * sizeof(float));
	root_queue.memcpy(data.cB3, C3, dp.nI * sizeof(float));
	root_queue.memcpy(data.cB4, C4, dp.nJ * sizeof(float));

	root_queue.wait_and_throw();

	return 0;
}

int CGpuNode::copyFromGPU() {

	Params& dp = data.params;

	zib::sycl::memcpy(root_queue, hMax, dp.nJ * sizeof(float), data.hMax + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI);
	zib::sycl::memcpy(root_queue, tArr, dp.nJ * sizeof(float), data.tArr + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI);

	root_queue.wait_and_throw();

	return 0;
}

int CGpuNode::copyIntermediate() {

	/* ignore copy requests if data already present on CPU side */
	if( copied )
		return 0;

	Params& dp = data.params;

	zib::sycl::memcpy(root_queue, h, dp.nJ * sizeof(float), data.h + dp.lpad, pitch, dp.nJ * sizeof(float), dp.nI);
	root_queue.wait_and_throw();

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

		root_queue.memcpy(h + idxPOI[n], data.h + dp.lpad + id, sizeof(float)).wait();
	}

	return 0;
}

int CGpuNode::freeMem() {
	/* 2-dim */
	cl::sycl::free(data.d, root_queue);
	cl::sycl::free(data.h, root_queue);
	cl::sycl::free(data.hMax, root_queue);
	cl::sycl::free(data.fM, root_queue);
	cl::sycl::free(data.fN, root_queue);
	cl::sycl::free(data.cR1, root_queue);
	cl::sycl::free(data.cR2, root_queue);
	cl::sycl::free(data.cR4, root_queue);
	cl::sycl::free(data.tArr, root_queue);

	/* 1-dim */
	cl::sycl::free(data.cR6, root_queue);
	cl::sycl::free(data.cB1, root_queue);
	cl::sycl::free(data.cB2, root_queue);
	cl::sycl::free(data.cB3, root_queue);
	cl::sycl::free(data.cB4, root_queue);

	cl::sycl::free(data.g_MinMax, root_queue);

	CArrayNode::freeMem();

	return 0;
}

#define INT_CEIL(x, n) ((((x) + (n) - 1) / (n)) * (n))

int CGpuNode::run() {
	Params& dp = data.params;

	int NJ = dp.jMax - dp.jMin + 1;
	int NI = dp.iMax - dp.iMin + 1;

	size_t max_wg_size = root_queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();

	cl::sycl::range<1> boundary_workgroup_size(max_wg_size);
	cl::sycl::range<1> boundary_size(INT_CEIL(std::max(dp.nI, dp.nJ), boundary_workgroup_size[0]));

#if defined(SYCL_LANGUAGE_VERSION) && defined (__INTEL_LLVM_COMPILER)
	/* For Intel, prevent the nd_range_error: "Non-uniform work-groups are not supported by the target device -54 (CL_INVALID_WORK_GROUP_SIZE))". */
	/* Originally we had n = 128 threads, 32 for x and 128/x = 4 threads, hardcoded in the CUDA code. */
	cl::sycl::range<2> compute_wnd_workgroup_size(4, 32);
#else
	cl::sycl::range<2> compute_wnd_workgroup_size(32, 32);
#endif
	cl::sycl::range<2> compute_wnd_size(
		INT_CEIL(NI, compute_wnd_workgroup_size[0]), /* y */
		INT_CEIL(NJ, compute_wnd_workgroup_size[1])  /* x */
	);

	dp.mTime = Par.time;

	std::array<std::vector<cl::sycl::event>, NUM_KERNELS> kernel_events;
    std::vector<cl::sycl::range<2>> cw_subsizes;
    std::vector<cl::sycl::id<2>> cw_offsets;

    /* split work */
    size_t num_used_queues = queues.size();
    for (size_t i = 0; i < num_used_queues; i++) {
        cw_offsets.push_back(cl::sycl::id<2>(compute_wnd_size[0] / num_used_queues * i, 0));
        cw_subsizes.push_back(cl::sycl::range<2>(compute_wnd_size[0] / num_used_queues * (i + 1), compute_wnd_size[1]));
    }

    /* launch kernels per (sub)device */
    for (size_t queue_idx = 0; queue_idx < num_used_queues; queue_idx++) {
    	kernel_events[KERNEL_WAVE_UPDATE].push_back(queues[queue_idx].submit([&](cl::sycl::handler &cgh) {
            auto kernel_data = data;

            cl::sycl::id<2> offset = cw_offsets[queue_idx];
            cl::sycl::range<2> subsize = cw_subsizes[queue_idx];

	    	cgh.parallel_for(
		    	cl::sycl::nd_range<2>(subsize, compute_wnd_workgroup_size),
    			[=](cl::sycl::nd_item<2> item) {
	    			waveUpdate(kernel_data, item, offset);
		    	});
    	}));
    }

    /* the lightweight boundary kernels are executed by a single subdevice only */
    kernel_events[KERNEL_WAVE_BOUND].push_back(queues.front().submit([&](cl::sycl::handler &cgh) {
        auto kernel_data = data;

        cgh.depends_on(kernel_events[KERNEL_WAVE_UPDATE]); // wait for all wave updates to finish
        cgh.parallel_for(
            cl::sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
            [=](cl::sycl::nd_item<1> item) {
                waveBoundary(kernel_data, item);
            });
    }));


    for (size_t queue_idx = 0; queue_idx < num_used_queues; queue_idx++) {
    	kernel_events[KERNEL_FLUX_UPDATE].push_back(queues[queue_idx].submit([&](cl::sycl::handler &cgh) {
            auto kernel_data = data;
            cl::sycl::id<2> offset = cw_offsets[queue_idx];

	    	cgh.depends_on(kernel_events[KERNEL_WAVE_BOUND]); /* wait for all/the single boundary kernel */
    		cgh.parallel_for(
	    		cl::sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
		    	[=](cl::sycl::nd_item<2> item) {
			    	fluxUpdate(kernel_data, item, offset);
    			});
	    }));
    }

    /* single device only */
    kernel_events[KERNEL_FLUX_BOUND].push_back(queues.front().submit([&](cl::sycl::handler &cgh) {
        auto kernel_data = data;

        cgh.depends_on(kernel_events[KERNEL_FLUX_UPDATE]);
        cgh.parallel_for(cl::sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
            [=](cl::sycl::nd_item<1> item) {
                fluxBoundary(kernel_data, item);
            });
    }));

	kernel_events[KERNEL_MEMSET].push_back(root_queue.memset(data.g_MinMax, 0, sizeof(cl::sycl::int4)));
    std::vector<cl::sycl::event> extend_deps{ kernel_events[KERNEL_FLUX_BOUND] };
    extend_deps.push_back(kernel_events[KERNEL_MEMSET].front());

    kernel_events[KERNEL_EXTEND].push_back(root_queue.submit([&](cl::sycl::handler &cgh) {
        auto kernel_data = data;

        cgh.depends_on(extend_deps);
        cgh.parallel_for(cl::sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
            [=](cl::sycl::nd_item<1> item) {
                gridExtend(kernel_data, item);
            });
    }));

	cl::sycl::int4 MinMax;
	kernel_events[KERNEL_MEMCPY].push_back(root_queue.memcpy(&MinMax, data.g_MinMax, sizeof(cl::sycl::int4), kernel_events[KERNEL_EXTEND]));
	kernel_events[KERNEL_MEMCPY].front().wait();

	/* TODO: respect alignments from device in window expansion (Preferred work group size multiple ?!) */
	if (MinMax.x()) Imin = dp.iMin = std::max(dp.iMin - 1, 2);
	if (MinMax.y()) Imax = dp.iMax = std::min(dp.iMax + 1, dp.nI - 1);
	if (MinMax.z()) Jmin = dp.jMin = std::max(dp.jMin - MEM_ALIGN, 2);
	if (MinMax.w()) Jmax = dp.jMax = std::min(dp.jMax + 1, dp.nJ - 1);

	kernel_events[KERNEL_WAVE_UPDATE].front().wait();

	for( int i = 0; is_profiling_enabled && i < /*NUM_TIMED_KERNELS*/ KERNEL_WAVE_UPDATE; i++ ) {
        std::vector<float> durations;
        for(const auto& e: kernel_events[i]) {
            durations.push_back(
                (e.get_profiling_info<cl::sycl::info::event_profiling::command_end>()
	    		- e.get_profiling_info<cl::sycl::info::event_profiling::command_start>()) / 1.0E+6
            );
        }

		kernel_duration[i] += *std::max_element(durations.begin(), durations.end());
	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
