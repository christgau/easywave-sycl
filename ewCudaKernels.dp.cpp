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
#include "ewGpuNode.dp.hpp"
#include "ewCudaKernels.dp.hpp"
#include <cmath>


void runWaveUpdateKernel( KernelData data, float* h, float *hMax, float* d, float* cR1, float* fM, float* fN, float *cR6, float *tArr, cl::sycl::nd_item<3> item_ct1)
{

  Params& dp = data.params;

  int i = /*item_ct1[1]*/ item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1) + dp.iMin;
  int j = /*item_ct1[0]*/ item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + dp.jMin;
  int ij = data.idx(i,j);
  float absH;

  /* maybe unnecessary if controlled from outside */
  if( i <= dp.iMax && j <= dp.jMax && d[ij] != 0 ) {
	  float hh = h[ij] - cR1[ij] * ( fM[ij] - fM[data.le(ij)] + fN[ij] * cR6[j] - fN[data.dn(ij)]*cR6[j-1] );

	  absH = cl::sycl::fabs(hh);

	  if( absH < dp.sshZeroThreshold ) {
		  hh = 0.f;
	  } else if( hh > hMax[ij] ) {
		  hMax[ij] = hh;
		  //hMax[ij] = fmaxf(hMax[ij],h[ij]);
	  }

	  if( dp.sshArrivalThreshold && tArr[ij] < 0 && absH > dp.sshArrivalThreshold )
	  	  tArr[ij] = dp.mTime;

	  h[ij] = hh;
  }
}

void runFluxUpdateKernel( KernelData data, float *h, float *d, float *fM, float *fN, float *cR2, float *cR4, cl::sycl::nd_item<3> item_ct1) {

	Params& dp = data.params;

	int i = /*item_ct1[1];*/ item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1) + dp.iMin;
	int j = /*item_ct1[0];*/ item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + dp.jMin;
	int ij = data.idx(i,j);

	if( i <= dp.iMax && j <= dp.jMax && d[ij] != 0 ) {

	  float hh = h[ij];

	  if( d[data.ri(ij)] != 0 ) {
		  fM[ij] = fM[ij] - cR2[ij]*(h[data.ri(ij)] - hh);
	  }

	  if( d[data.up(ij)] != 0 )
		  fN[ij] = fN[ij] - cR4[ij]*(h[data.up(ij)] - hh);

	}

}


void runWaveBoundaryKernel( KernelData data, float *h, float *fM, float *fN, float *cB1, float *cB2, float *cB3, float *cB4, cl::sycl::nd_item<3> item_ct1) {

	KernelData& dt = data;
	Params& dp = data.params;

	int id = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + 2;
	int ij;

	if( id <= dp.nI-1 ) {
	  ij = dt.idx(id,1);
	  h[ij] = cl::sycl::sqrt(cl::sycl::pow(fN[ij], 2.0f) + 0.25f*cl::sycl::pow((fM[ij] + fM[dt.le(ij)]), 2.0f))*cB1[id-1];
	  if( fN[ij] > 0 ) h[ij] = -h[ij];
	}

	if( id <= dp.nI-1 ) {
	  ij = dt.idx(id,dp.nJ);
	  h[ij] = cl::sycl::sqrt(cl::sycl::pow(fN[dt.dn(ij)], 2.0f) + 0.25f*cl::sycl::pow((fM[ij] + fM[dt.dn(ij)]), 2.0f))*cB3[id-1];
	  if( fN[dt.dn(ij)] < 0 ) h[ij] = -h[ij];
	}

	if( id <= dp.nJ-1 ) {
	  ij = dt.idx(1,id);
	  h[ij] = cl::sycl::sqrt(cl::sycl::pow(fM[ij], 2.0f) + 0.25f*cl::sycl::pow((fN[ij] + fN[dt.dn(ij)]), 2.0f))*cB2[id-1];
	  if( fM[ij] > 0 ) h[ij] = -h[ij];
	}

	if( id <= dp.nJ-1 ) {
	  ij = dt.idx(dp.nI,id);
	  h[ij] = cl::sycl::sqrt(cl::sycl::pow(fM[dt.le(ij)], 2.0f) + 0.25f*cl::sycl::pow((fN[ij] + fN[dt.dn(ij)]), 2.0f))*cB4[id-1];
	  if( fM[dt.le(ij)] < 0 ) h[ij] = -h[ij];
	}

	if( id == 2 ) {
	  ij = dt.idx(1,1);
	  h[ij] = cl::sycl::sqrt(cl::sycl::pow(fM[ij], 2.0f) + cl::sycl::pow(fN[ij], 2.0f))*cB1[0];
	  if( fN[ij] > 0 ) h[ij] = -h[ij];

	  ij = dt.idx(dp.nI,1);
	  h[ij] = cl::sycl::sqrt(cl::sycl::pow(fM[dt.le(ij)], 2.0f) + cl::sycl::pow(fN[ij], 2.0f))*cB1[dp.nI-1];
	  if( fN[ij] > 0 ) h[ij] = -h[ij];

	  ij = dt.idx(1,dp.nJ);
	  h[ij] = cl::sycl::sqrt(cl::sycl::pow(fM[ij], 2.0f) + cl::sycl::pow(fN[dt.dn(ij)], 2.0f))*cB3[0];
	  if( fN[dt.dn(ij)] < 0 ) h[ij] = -h[ij];

	  ij = dt.idx(dp.nI,dp.nJ);
	  h[ij] = cl::sycl::sqrt(cl::sycl::pow(fM[dt.le(ij)], 2.0f) + cl::sycl::pow(fN[dt.dn(ij)], 2.0f))*cB3[dp.nI-1];
	  if( fN[dt.dn(ij)] < 0 ) h[ij] = -h[ij];
	}

}

 void runFluxBoundaryKernel( KernelData data, float *h, float *fM, float *fN, float *cR2, float *cR4, cl::sycl::nd_item<3> item_ct1) {

	KernelData& dt = data;
	Params& dp = data.params;

	int id = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + 1;
	int ij;

	if( id <= dp.nI-1 ) {
	  ij = dt.idx(id,1);
	  fM[ij] = fM[ij] - cR2[ij]*(h[dt.ri(ij)] - h[ij]);
	}

	if( id <= dp.nJ ) {
	  ij = dt.idx(1,id);
	  fM[ij] = fM[ij] - cR2[ij]*(h[dt.ri(ij)] - h[ij]);
	}

	if( id <= dp.nI-1 ) {
	  ij = dt.idx(id,dp.nJ);
	  fM[ij] = fM[ij] - cR2[ij]*(h[dt.ri(ij)] - h[ij]);
	}

	if( id <= dp.nJ-1 ) {
	  ij = dt.idx(1,id);
	  fN[ij] = fN[ij] - cR4[ij]*(h[dt.up(ij)] - h[ij]);
	}

	if( id <= dp.nI ) {
	  ij = dt.idx(id,1);
	  fN[ij] = fN[ij] - cR4[ij]*(h[dt.up(ij)] - h[ij]);
	}

	if( id <= dp.nJ-1 ) {
	  ij = dt.idx(dp.nI,id);
	  fN[ij] = fN[ij] - cR4[ij]*(h[dt.up(ij)] - h[ij]);
	}

}

void runGridExtendKernel( KernelData data, float *h, cl::sycl::nd_item<3> item_ct1) {

	Params& dp = data.params;

	int id = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + 1;

#if 0
    cl::sycl::atomic<int> m[4] = {
        cl::sycl::atomic<int> { cl::sycl::global_ptr<int>{ &(data.g_MinMax[0]) } },
        cl::sycl::atomic<int> { cl::sycl::global_ptr<int>{ &(data.g_MinMax[1]) } },
        cl::sycl::atomic<int> { cl::sycl::global_ptr<int>{ &(data.g_MinMax[2]) } },
        cl::sycl::atomic<int> { cl::sycl::global_ptr<int>{ &(data.g_MinMax[3]) } }
    };

//    (DPCPP_COMPATIBILITY_TEMP >= 130)
    /* for devices with support for atomics (was CUDA CC >= 2.0) */

	if( id >= dp.jMin && id <= dp.jMax ) {

	  if( fabsf(data.h[data.idx(dp.iMin+2,id)]) > dp.sshClipThreshold )
          m[0].fetch_add(1);
//          data.g_MinMax[0]++;
//          data.g_MinMax[0].fetch_add(1);
//		  atomicAdd( &(data.g_MinMax[0]), 1 );

	  if( fabsf(data.h[data.idx(dp.iMax-2,id)]) > dp.sshClipThreshold )
          m[1].fetch_add(1);
//          data.g_MinMax[1].fetch_add(1);
//		  atomicAdd( &(data.g_MinMax[1]), 1 );
	}

	if( id >= dp.iMin && id <= dp.iMax ) {

	  if( fabsf(data.h[data.idx(id,dp.jMin+2)]) > dp.sshClipThreshold )
          m[2].fetch_add(1);
//          data.g_MinMax[2].fetch_add(1);
//		  atomicAdd( &(data.g_MinMax[2]), 1 );

	  if( fabsf(data.h[data.idx(id,dp.jMax-2)]) > dp.sshClipThreshold )
          m[3].fetch_add(1);
//          data.g_MinMax[3].fetch_add(1);
//		  atomicAdd( &(data.g_MinMax[3]), 1 );
	}

#else

        if( id == 1 ) {

          for( int j = dp.jMin; j <= dp.jMax; j++ ) {

            if( cl::sycl::fabs(h[data.idx(dp.iMin+2,j)]) > dp.sshClipThreshold ) {
                data.g_MinMax[0] = 1;
                break;
            }

          }

          for( int j = dp.jMin; j <= dp.jMax; j++ ) {

            if( cl::sycl::fabs(h[data.idx(dp.iMax-2,j)]) > dp.sshClipThreshold ) {
               data.g_MinMax[1] = 1;
               break;
            }

          }

          for( int i = dp.iMin; i <= dp.iMax; i++ ) {

            if( cl::sycl::fabs(h[data.idx(i,dp.jMin+2)]) > dp.sshClipThreshold ) {
              data.g_MinMax[2] = 1;
              break;
            }

          }

          for( int i = dp.iMin; i <= dp.iMax; i++ ) {

            if( cl::sycl::fabs(h[data.idx(i,dp.jMax-2)]) > dp.sshClipThreshold ) {
              data.g_MinMax[3] = 1;
              break;
            }

          }

        }

#endif

}
