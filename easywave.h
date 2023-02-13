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

#ifndef EASYWAVE_H
#define EASYWAVE_H

#define Re 6384.e+3          // Earth radius
#define Gravity 9.81         // gravity acceleration
#define Omega 7.29e-5        // Earth rotation period [1/sec]

namespace easywave {
	using gpu_api_type_t = enum class GpuApiType { UNKNOWN, CUDA, HIP, SYCL };
}

#ifdef SYCL_LANGUAGE_VERSION

// Needs to be included before ewdefs to avoid nasty clash with idx preprocessor define.
#include <CL/sycl.hpp>

#define EW_GPU_ENABLED 1

#define DEVICE_FUNCTION /* empty */
#define HOST_FUNCTION /* empty */

constexpr easywave::gpu_api_type_t api = easywave::GpuApiType::SYCL;
namespace easywave { using quad_int_t = cl::sycl::int4; }

#elif defined(__CUDACC__)

#include <cuda.h>

#define DEVICE_FUNCTION __device__
#define HOST_FUNCTION __host__

#define EW_GPU_ENABLED 1

constexpr easywave::gpu_api_type_t api = easywave::GpuApiType::CUDA;
namespace easywave { using quad_int_t = int4; }

#elif defined(__HIPCC__)

#define DEVICE_FUNCTION __device__
#define HOST_FUNCTION __host__

#include "hip/hip_runtime.h"

#define EW_GPU_ENABLED 1

constexpr easywave::gpu_api_type_t api = easywave::GpuApiType::HIP;
namespace easywave { using quad_int_t = int4; }

#else

#include <array>

#define DEVICE_FUNCTION /* empty */
#define HOST_FUNCTION /* empty */

#undef EW_GPU_ENABLED

constexpr easywave::gpu_api_type_t api = easywave::GpuApiType::UNKNOWN;
namespace easywave { using quad_int_t = std::array<int, 4>; }

#endif /* GPU programming model */

#include "ewdefs.h"

int ewLoadBathymetry();
int ewParam( int argc, char **argv );
void ewLogParams(void);
int ewReset();
int ewSource();
int ewStep();
int ewStepCor();

int ewStart2DOutput();
int ewOut2D();
int ewDump2D();
int ewLoadPOIs();
int ewSavePOIs();
int ewDumpPOIs();
int ewDumpPOIsCompact( int istage );

extern int NPOIs;
extern long *idxPOI;

#define NUM_DURATIONS 5
extern float dur[NUM_DURATIONS];

#include "ewNode.h"

extern CNode *gNode;

/* verbose printf: only executed if -verbose was set */
#define printf_v( Args, ... )	if( Par.verbose ) printf( Args, ##__VA_ARGS__);

#endif /* EASYWAVE_H */
