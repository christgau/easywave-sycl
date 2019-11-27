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

#define MAX_VARS_PER_NODE 12

#define iD    0
#define iH    1
#define iHmax 2
#define iM    3
#define iN    4
#define iR1   5
#define iR2   6
#define iR3   7
#define iR4   8
#define iR5   9
#define iTime 10
#define iTopo 11

// Global data
struct EWPARAMS {
  char *modelName;
  char *modelSubset;
  char *fileBathymetry;
  char *fileSource;
  char *filePOIs;
  int dt;
  int time;
  int timeMax;
  int poiDt;
  int poiReport;
  int outDump;
  int outProgress;
  int outPropagation;
  int coriolis;
  float dmin;
  float poiDistMax;
  float poiDepthMin;
  float poiDepthMax;
  float ssh0ThresholdRel;
  float ssh0ThresholdAbs;
  float sshClipThreshold;
  float sshZeroThreshold;
  float sshTransparencyThreshold;
  float sshArrivalThreshold;
  bool gpu;
  bool adjustZtop;
  bool verbose;
};

extern struct EWPARAMS Par;
extern int NLon,NLat;
extern double LonMin,LonMax,LatMin,LatMax;
extern double DLon,DLat;                 // steps in grad
extern double Dx,Dy;                     // steps in m, dx must be multiplied by cos(y) before use
extern float *R6;
extern float *C1;
extern float *C2;
extern float *C3;
extern float *C4;
extern int Imin;
extern int Imax;
extern int Jmin;
extern int Jmax;

#define idx(j,i) ((i-1)*NLat+j-1)
#define getLon(i) (LonMin+(i-1)*DLon)
#define getLat(j) (LatMin+(j-1)*DLat)

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

/* verbose printf: only executed if -verbose was set */
#define printf_v( Args, ... )	if( Par.verbose ) printf( Args, ##__VA_ARGS__);

#include "ewNode.h"

extern CNode *gNode;

#endif /* EASYWAVE_H */
