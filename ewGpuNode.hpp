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

#ifndef EW_GPUNODE_H
#define EW_GPUNODE_H

/* FIXME: check header dependencies */
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "easywave.h"
#include "ewNode.h"
#include <stdio.h>

#include <chrono>

#define CUDA_CALL(x) if( x != cudaSuccess ) { fprintf( stderr, "Error in file %s on line %u: %s\n", __FILE__, __LINE__, cudaGetErrorString( cudaGetLastError() ) ); return 1; }

#undef idx

class Params {

public:
	int mTime;
	int nI;
	int nJ;
	int iMin;
	int iMax;
	int jMin;
	int jMax;
	float sshArrivalThreshold;
	float sshZeroThreshold;
	float sshClipThreshold;

	/* pitch / sizeof(float) */
	size_t pI;
	size_t lpad;
};

class KernelData {

public:
	/* 2-dim */
	float *d;
	float *h;
	float *hMax;
	float *fM;
	float *fN;
	float *cR1;
	float *cR2;
	float *cR4;
	float *tArr;

	/* 1-dim */
	float *cR6;
	float *cB1;
	float *cB2;
	float *cB3;
	float *cB4;

	Params params;

	sycl::int4 *g_MinMax;

        int le( int ij ) { return ij - params.pI; }
	int ri( int ij ) { return ij + params.pI; }
	int up( int ij ) { return ij + 1; }
	int dn( int ij ) { return ij - 1; }
	int idx( int i, int j ) { return (j-1) + (i-1) * params.pI + params.lpad; }
};

/* GPU dependent */
class CGpuNode : public CArrayNode {

protected:
	KernelData data;

	/* line size in bytes */
	size_t pitch;

	/* specifies if data was already copied in the current calculation step */
	bool copied;

	std::chrono::high_resolution_clock::time_point evtStart[5];
	std::chrono::high_resolution_clock::time_point evtEnd[5];
        float dur[5];

public:
	CGpuNode();
	int mallocMem();
	int copyToGPU();
	int copyFromGPU();
	int copyIntermediate();
	int copyPOIs();
	int freeMem();
	int run();
};

#endif /* EW_GPUNODE_H */
