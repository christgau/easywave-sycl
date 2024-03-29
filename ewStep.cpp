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

// Time stepping
#include <stdio.h>
#include <stdlib.h>

#include <chrono>

#include "utilits.h"
#include "easywave.h"
#include <cmath>

/* TODO: still not perfect */
//#define Node(idx1, idx2) Node.node[idx1][idx2]
//#define CNode CStructNode
//#define gNode ((CStructNode*)gNode)
#define CNode CArrayNode
#define gNode ((CArrayNode*)gNode)

#define SQR(x) ((x) * (x))

float dur[5] = { 0 };

int ewStep( void )
{
  int i, j, enlarge;

  CNode& Node = *gNode;

  // sea floor topography (mass conservation)
  float* D = (float*) Node.getBuf(iD);
  float* H = (float*) Node.getBuf(iH);
  float* M = (float*) Node.getBuf(iM);
  float* N = (float*) Node.getBuf(iN);
  float* R1 = (float*) Node.getBuf(iR1);
  float* Hmax = (float*) Node.getBuf(iHmax);
  float* Time = (float*) Node.getBuf(iTime);

  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared) private(j)
  //if (Imax - Imin > 500)
  for( i=Imin; i<=Imax; i++ ) {
    #pragma omp simd
    for( j=Jmin; j<=Jmax; j++ ) {

      int m = idx(j,i);

//      if( Node(m, iD) != 0 ) {
      if (D[m] != 0) {

//        Node(m, iH) = Node(m, iH) - Node(m, iR1)*( Node(m, iM) - Node(m-NLat, iM) + Node(m, iN)*R6[j] - Node(m-1, iN)*R6[j-1] );
        H[m] = H[m] - R1[m] * (M[m] - M[m - NLat] + N[m] * R6[j] - N[m - 1]*R6[j-1]);

//        float absH = fabs(Node(m, iH));
        float absH = fabs(H[m]);

//        if( absH < Par.sshZeroThreshold ) Node(m, iH) = 0.;
        if( absH < Par.sshZeroThreshold ) H[m] = 0.;

//        if( Node(m, iH) > Node(m, iHmax) ) Node(m, iHmax) = Node(m, iH);
        if( H[m] > Hmax[m] ) Hmax[m] = H[m];

//        if( Par.sshArrivalThreshold && Node(m, iTime) < 0 && absH > Par.sshArrivalThreshold ) Node(m, iTime) = (float)Par.time;
        if( Par.sshArrivalThreshold && Time[m] < 0 && absH > Par.sshArrivalThreshold ) Time[m] = (float)Par.time;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  dur[0] += std::chrono::duration<float, std::milli>(end - start).count();

  // open bondary conditions
  start = std::chrono::high_resolution_clock::now();
  if( Jmin <= 2 ) {
    #pragma omp simd
    for( i=2; i<=(NLon-1); i++ ) {
      int m = idx(1,i);
//      Node(m, iH) = sqrt( pow(Node(m, iN),2.) + 0.25*pow((Node(m, iM)+Node(m-NLat, iM)),2.) )*C1[i];
//      if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
      H[m] = sqrt( SQR(N[m]) + 0.25 * SQR((M[m] + M[m-NLat]))) * C1[i];
      if ( N[m] > 0 ) H[m] = -H[m];
    }
  }
  if( Imin <= 2 ) {
    #pragma omp simd
    for( j=2; j<=(NLat-1); j++ ) {
      int m = idx(j,1);
//      Node(m, iH) = sqrt( pow(Node(m, iM),2.) + 0.25*pow((Node(m, iN)+Node(m-1, iN)),2.) )*C2[j];
//      if( Node(m, iM) > 0 ) Node(m, iH) = - Node(m, iH);
      H[m] = sqrt( SQR(M[m]) + 0.25 * SQR(N[m] + N[m-1]) ) * C2[j];
      if (M[m] > 0) H[m] = -H[m];
    }
  }
  if( Jmax >= (NLat-1) ) {
    #pragma omp simd
    for( i=2; i<=(NLon-1); i++ ) {
      int m = idx(NLat,i);
//      Node(m, iH) = sqrt( pow(Node(m-1, iN),2.) + 0.25*pow((Node(m, iM)+Node(m-1, iM)),2.) )*C3[i];
//      if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
      H[m] = sqrt( SQR(N[m-1]) + 0.25 * SQR(M[m] + M[m-1]) ) * C3[i];
      if (N[m-1] < 0) H[m] = -H[m];
    }
  }
  if( Imax >= (NLon-1) ) {
    #pragma omp simd
    for( j=2; j<=(NLat-1); j++ ) {
      int m = idx(j,NLon);
//      Node(m, iH) = sqrt( pow(Node(m-NLat, iM),2.) + 0.25*pow((Node(m, iN)+Node(m-1, iN)),2.) )*C4[j];
//      if( Node(m-NLat, iM) < 0 ) Node(m, iH) = - Node(m, iH);
      H[m] = sqrt( SQR(M[m-NLat]) + 0.25 * SQR(N[m] + N[m-1]) ) * C4[j];
      if (M[m-NLat] < 0) H[m] = -H[m];
    }
  }

  if( Jmin <= 2 ) {
    int m = idx(1,1);
    Node(m, iH) = sqrt( SQR(Node(m, iM)) + SQR(Node(m, iN)) )*C1[1];
    if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
    m = idx(1,NLon);
    Node(m, iH) = sqrt( SQR(Node(m-NLat, iM)) + SQR(Node(m, iN)) )*C1[NLon];
    if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
  }
  if( Jmin >= (NLat-1) ) {
    int m = idx(NLat,1);
    Node(m, iH) = sqrt( SQR(Node(m, iM)) + SQR(Node(m-1, iN)) )*C3[1];
    if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
    m = idx(NLat,NLon);
    Node(m, iH) = sqrt( SQR(Node(m-NLat, iM)) + SQR(Node(m-1, iN)) )*C3[NLon];
    if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
  }
  end = std::chrono::high_resolution_clock::now();
  dur[1] += std::chrono::duration<float, std::milli>(end - start).count();

  // moment conservation
  start = std::chrono::high_resolution_clock::now();
  float* R2 = (float*) Node.getBuf(iR2);
  float* R4 = (float*) Node.getBuf(iR4);
  #pragma omp parallel for private(j)
  //if (Imax - Imin > 1000)
  for( i=Imin; i<=Imax; i++ ) {
    #pragma omp simd
    for( j=Jmin; j<=Jmax; j++ ) {

      int m = idx(j,i);

//      if( (Node(m, iD)*Node(m+NLat, iD)) != 0 )
//        Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH)-Node(m, iH));
      if ( D[m] * D[m + NLat] != 0 )
          M[m] = M[m] - R2[m] * (H[m + NLat] - H[m]);

//      if( (Node(m, iD)*Node(m+1, iD)) != 0 )
//        Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH)-Node(m, iH));
      if ( D[m] * D[m + 1] != 0 )
          N[m] = N[m] - R4[m] * (H[m + 1] - H[m]);

    }
  }
  end = std::chrono::high_resolution_clock::now();
  dur[2] += std::chrono::duration<float, std::milli>(end - start).count();

  start = std::chrono::high_resolution_clock::now();
  // open boundaries
  if( Jmin <= 2 ) {
    #pragma omp simd
    for( i=1; i<=(NLon-1); i++ ) {
      int m = idx(1,i);
//    Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
      M[m] = M[m] - R2[m] * (H[m + NLat] - H[m]);
    }
  }
  if( Imin <= 2 ) {
    #pragma omp simd
    for( j=1; j<=NLat; j++ ) {
      int m = idx(j,1);
//      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
      M[m] = M[m] - R2[m] * (H[m + NLat] - H[m]);
    }
  }
  if( Jmax >= (NLat-1) ) {
    #pragma omp simd
    for( i=1; i<=(NLon-1); i++ ) {
      int m = idx(NLat,i);
//      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
      M[m] = M[m] - R2[m] * (H[m + NLat] - H[m]);
    }
  }
  if( Imin <= 2 ) {
    #pragma omp simd
    for( j=1; j<=(NLat-1); j++ ) {
      int m = idx(j,1);
//      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
      N[m] = N[m] - R4[m] * (H[m+1] - H[m]);
    }
  }
  if( Jmin <= 2 ) {
    #pragma omp simd
    for( i=1; i<=NLon; i++ ) {
      int m = idx(1,i);
//      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
      N[m] = N[m] - R4[m] * (H[m+1] - H[m]);
    }
  }
  if( Imax >= (NLon-1) ) {
    #pragma omp simd
    for( j=1; j<=(NLat-1); j++ ) {
      int m = idx(j,NLon);
//      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
      N[m] = N[m] - R4[m] * (H[m+1] - H[m]);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  dur[3] += std::chrono::duration<float, std::milli>(end - start).count();

  start = std::chrono::high_resolution_clock::now();
  // calculation area for the next step
  if( Imin > 2 ) {
    for( enlarge=0, j=Jmin; j<=Jmax; j++ ) {
      if( fabs(Node(idx(j,Imin+2), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Imin--; if( Imin < 2 ) Imin = 2; }
  }
  if( Imax < (NLon-1) ) {
    for( enlarge=0, j=Jmin; j<=Jmax; j++ ) {
      if( fabs(Node(idx(j,Imax-2), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Imax++; if( Imax > (NLon-1) ) Imax = NLon-1; }
  }
  if( Jmin > 2 ) {
    for( enlarge=0, i=Imin; i<=Imax; i++ ) {
      if( fabs(Node(idx(Jmin+2,i), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Jmin--; if( Jmin < 2 ) Jmin = 2; }
  }
  if( Jmax < (NLat-1) ) {
    for( enlarge=0, i=Imin; i<=Imax; i++ ) {
      if( fabs(Node(idx(Jmax-2,i), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Jmax++; if( Jmax > (NLat-1) ) Jmax = NLat-1; }
  }
  end = std::chrono::high_resolution_clock::now();
  dur[4] += std::chrono::duration<float, std::milli>(end - start).count();

return 0;
}



int ewStepCor( void )
{
  int i,j,enlarge;
  float absH,v1,v2;
  int m;

  CNode& Node = *gNode;

  // sea floor topography (mass conservation)
  #pragma omp parallel for default(shared) private(i,j,absH)
  for( i=Imin; i<=Imax; i++ ) {
    for( j=Jmin; j<=Jmax; j++ ) {

      m = idx(j,i);

      if( Node(m, iD) == 0 ) continue;

      Node(m, iH) = Node(m, iH) - Node(m, iR1)*( Node(m, iM) - Node(m-NLat, iM) + Node(m, iN)*R6[j] - Node(m-1, iN)*R6[j-1] );

      absH = fabs(Node(m, iH));

      if( absH < Par.sshZeroThreshold ) Node(m, iH) = 0.;

      if( Node(m, iH) > Node(m, iHmax) ) Node(m, iHmax) = Node(m, iH);

      if( Par.sshArrivalThreshold && Node(m, iTime) < 0 && absH > Par.sshArrivalThreshold ) Node(m, iTime) = (float)Par.time;

    }
  }

  // open bondary conditions
  if( Jmin <= 2 ) {
    for( i=2; i<=(NLon-1); i++ ) {
      m = idx(1,i);
      Node(m, iH) = sqrt(pow(Node(m, iN), 2.) +
                         0.25 * pow((Node(m, iM) + Node(m - NLat, iM)), 2.)) *
                    C1[i];
      if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
    }
  }
  if( Imin <= 2 ) {
    for( j=2; j<=(NLat-1); j++ ) {
      m = idx(j,1);
      Node(m, iH) = sqrt(pow(Node(m, iM), 2.) +
                         0.25 * pow((Node(m, iN) + Node(m - 1, iN)), 2.)) *
                    C2[j];
      if( Node(m, iM) > 0 ) Node(m, iH) = - Node(m, iH);
    }
  }
  if( Jmax >= (NLat-1) ) {
    for( i=2; i<=(NLon-1); i++ ) {
      m = idx(NLat,i);
      Node(m, iH) = sqrt(pow(Node(m - 1, iN), 2.) +
                         0.25 * pow((Node(m, iM) + Node(m - 1, iM)), 2.)) *
                    C3[i];
      if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
    }
  }
  if( Imax >= (NLon-1) ) {
    for( j=2; j<=(NLat-1); j++ ) {
      m = idx(j,NLon);
      Node(m, iH) = sqrt(pow(Node(m - NLat, iM), 2.) +
                         0.25 * pow((Node(m, iN) + Node(m - 1, iN)), 2.)) *
                    C4[j];
      if( Node(m-NLat, iM) < 0 ) Node(m, iH) = - Node(m, iH);
    }
  }
  if( Jmin <= 2 ) {
    m = idx(1,1);
    Node(m, iH) = sqrt(pow(Node(m, iM), 2.) + pow(Node(m, iN), 2.)) * C1[1];
    if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
    m = idx(1,NLon);
    Node(m, iH) =
        sqrt(pow(Node(m - NLat, iM), 2.) + pow(Node(m, iN), 2.)) * C1[NLon];
    if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
  }
  if( Jmin >= (NLat-1) ) {
    m = idx(NLat,1);
    Node(m, iH) = sqrt(pow(Node(m, iM), 2.) + pow(Node(m - 1, iN), 2.)) * C3[1];
    if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
    m = idx(NLat,NLon);
    Node(m, iH) =
        sqrt(pow(Node(m - NLat, iM), 2.) + pow(Node(m - 1, iN), 2.)) * C3[NLon];
    if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
  }

  // moment conservation
  // longitudial flux update
  #pragma omp parallel for default(shared) private(i,j,v1,v2)
  for( i=Imin; i<=Imax; i++ ) {
    for( j=Jmin; j<=Jmax; j++ ) {

      m = idx(j,i);

      if( (Node(m, iD)*Node(m+NLat, iD)) == 0 ) continue;

      v1 = Node(m+NLat, iH) - Node(m, iH);
      v2 = Node(m-1, iN) + Node(m, iN) + Node(m+NLat, iN) + Node(m+NLat-1, iN);
      Node(m, iM) = Node(m, iM) - Node(m, iR2)*v1 + Node(m, iR3)*v2;
    }
  }
  // open boundaries
  if( Jmin <= 2 ) {
    for( i=1; i<=(NLon-1); i++ ) {
      m = idx(1,i);
      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
    }
  }
  if( Imin <= 2 ) {
    for( j=1; j<=NLat; j++ ) {
      m = idx(j,1);
      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
    }
  }
  if( Jmax >= (NLat-1) ) {
    for( i=1; i<=(NLon-1); i++ ) {
      m = idx(NLat,i);
      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
    }
  }

  // lattitudial flux update
  #pragma omp parallel for default(shared) private(i,j,v1,v2)
  for( i=Imin; i<=Imax; i++ ) {
    for( j=Jmin; j<=Jmax; j++ ) {

      m = idx(j,i);

      if( (Node(m, iD)*Node(m+1, iD)) == 0 ) continue;

      v1 = Node(m+1, iH) - Node(m, iH);
      v2 = Node(m-NLat, iM) + Node(m, iM) + Node(m-NLat+1, iM) + Node(m+1, iM);
      Node(m, iN) = Node(m, iN) - Node(m, iR4)*v1 - Node(m, iR5)*v2;
    }
  }
  // open boundaries
  if( Imin <= 2 ) {
    for( j=1; j<=(NLat-1); j++ ) {
      m = idx(j,1);
      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
    }
  }
  if( Jmin <= 2 ) {
    for( i=1; i<=NLon; i++ ) {
      m = idx(1,i);
      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
    }
  }
  if( Imax >= (NLon-1) ) {
    for( j=1; j<=(NLat-1); j++ ) {
      m = idx(j,NLon);
      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
    }
  }

  // calculation area for the next step
  if( Imin > 2 ) {
    for( enlarge=0, j=Jmin; j<=Jmax; j++ ) {
      if( fabs(Node(idx(j,Imin+2), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Imin--; if( Imin < 2 ) Imin = 2; }
  }
  if( Imax < (NLon-1) ) {
    for( enlarge=0, j=Jmin; j<=Jmax; j++ ) {
      if( fabs(Node(idx(j,Imax-2), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Imax++; if( Imax > (NLon-1) ) Imax = NLon-1; }
  }
  if( Jmin > 2 ) {
    for( enlarge=0, i=Imin; i<=Imax; i++ ) {
      if( fabs(Node(idx(Jmin+2,i), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Jmin--; if( Jmin < 2 ) Jmin = 2; }
  }
  if( Jmax < (NLat-1) ) {
    for( enlarge=0, i=Imin; i<=Imax; i++ ) {
      if( fabs(Node(idx(Jmax-2,i), iH)) > Par.sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Jmax++; if( Jmax > (NLat-1) ) Jmax = NLat-1; }
  }

return 0;
}
