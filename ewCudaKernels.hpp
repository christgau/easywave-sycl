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

#ifndef EW_KERNELS_H
#define EW_KERNELS_H

#ifdef USE_INLINE_KERNELS
/* see below for rationale */
#undef SYCL_EXTERNAL
#define SYCL_EXTERNAL static
#endif /* USE_INLINE_KERNELS */

SYCL_EXTERNAL void waveUpdate(KernelData data, const cl::sycl::nd_item<2> item_ct1, const cl::sycl::id<2> offset);
SYCL_EXTERNAL void waveBoundary(KernelData data, const cl::sycl::nd_item<1> item_ct1);
SYCL_EXTERNAL void fluxUpdate(KernelData data, const cl::sycl::nd_item<2> item_ct1, const cl::sycl::id<2> offset);
SYCL_EXTERNAL void fluxBoundary(KernelData data, const cl::sycl::nd_item<1> item_ct1);
SYCL_EXTERNAL void gridExtend(KernelData data, const cl::sycl::nd_item<1> item_ct1);

#ifdef USE_INLINE_KERNELS
/* hipSYCL does not support SYCL_EXTERNAL so kernels must be contained in the same
 * compilation unit where they are called. Thus, we do a dirty include of the source
 * file here (see also https://github.com/illuhad/hipSYCL/issues/604) */
#include "ewCudaKernels.cpp"
#endif /* USE_INLINE_KERNELS */

#endif /* EW_KERNELS_H */
