__global__ void orcu_kernel7002(const int n, float* a, float* b, float* c) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  __shared__ float shared_c[32];
  __shared__ float shared_a[32];
  __shared__ float shared_b[32];
  for (int i=tid; i<=n-1; i+=gsize) {
    shared_a[threadIdx.x]=a[i];
    shared_b[threadIdx.x]=b[i];
    {
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=0;
      shared_c[threadIdx.x]=0;
      shared_c[threadIdx.x]=0;
      shared_c[threadIdx.x]=0;
      shared_c[threadIdx.x]=0;
      shared_c[threadIdx.x]=0;
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x]+shared_b[threadIdx.x]+shared_a[threadIdx.x]*shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
      shared_c[threadIdx.x]=shared_b[threadIdx.x];
    }
    c[i]=shared_c[threadIdx.x];
  }
}
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <chrono>
#include <iomanip>

#define FIXED_RSEED

#ifndef nevts
#define nevts 100
#endif

#ifndef bsize
#define bsize 128
#endif
#ifndef ntrks
#define ntrks 9600
#endif

#define nb    ntrks/bsize
#define smear 0.1

#ifndef NITER
#define NITER 5
#endif
#ifndef nlayer
#define nlayer 20
#endif

#ifndef nthreads
#define nthreads 64
#endif

size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

size_t SymOffsets33(size_t i) {
  const size_t offs[9] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
  return offs[i];
}

size_t SymOffsets66(size_t i) {
  const size_t offs[36] = {0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};
  return offs[i];
}

/struct ATRK {
  float par[6];
  float cov[21];
  int q;
  //  int hitidx[22];
};

struct AHIT {
  float pos[3];
  float cov[6];
};

struct MP1I {
  int data[1*bsize];
};

struct MP22I {
  int data[22*bsize];
};

struct MP3F {
  float data[3*bsize];
};

struct MP6F {
  float data[6*bsize];
};

struct MP3x3 {
  float data[9*bsize];
};
struct MP3x6 {
  float data[18*bsize];
};
struct MP3x3SF {
  float data[6*bsize];
};

struct MP6x6SF {
  float data[21*bsize];
};

struct MP6x6F {
  float data[36*bsize];
};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;
  //  MP22I   hitidx;
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
};

#define N bsize
void MultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C) {
  // First setp of error propagation = A*B
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = [32];
          param BC[] = [14];
          param SC[] = [1];
          param CB[] = [True];
          param PL[] = [16];
          param CFLAGS[] = [''];
        }
        def build {
          arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
        }
        def input_params {
          param _N[] = [128];
        }
        def input_vars {
          decl dynamic float a[_N] = random;
          decl dynamic float b[_N] = random;
          decl dynamic float c[_N] = random;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 1;
        }
  ) @*/
/**-- (Generated by Orio) 
Best performance cost: 
  [0.021248] 
Tuned for specific problem sizes: 
  _N = 128 
Best performance parameters: 
  BC = 14 
  CB = True 
  CFLAGS =  
  PL = 16 
  SC = 1 
  TC = 32 
--**/

  
  int n = _N;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, streamCount=SC, cacheBlocks=CB, preferL1Size=PL)

  for (i=0; i<=n-1; i++)
  {
    c[ 0*n+i] = b[ 0*n+i] + a[ 2*n+i]*b[ 3*n+i] + a[ 3*n+i]*b[ 6*n+i] + a[ 4*n+i]*b[10*n+i] + a[ 5*n+i]*b[15*n+i];
    c[ 1*n+i] = b[ 1*n+i] + a[ 2*n+i]*b[ 4*n+i] + a[ 3*n+i]*b[ 7*n+i] + a[ 4*n+i]*b[11*n+i] + a[ 5*n+i]*b[16*n+i];
    c[ 2*n+i] = b[ 3*n+i] + a[ 2*n+i]*b[ 5*n+i] + a[ 3*n+i]*b[ 8*n+i] + a[ 4*n+i]*b[12*n+i] + a[ 5*n+i]*b[17*n+i];
    c[ 3*n+i] = b[ 6*n+i] + a[ 2*n+i]*b[ 8*n+i] + a[ 3*n+i]*b[ 9*n+i] + a[ 4*n+i]*b[13*n+i] + a[ 5*n+i]*b[18*n+i];
    c[ 4*n+i] = b[10*n+i] + a[ 2*n+i]*b[12*n+i] + a[ 3*n+i]*b[13*n+i] + a[ 4*n+i]*b[14*n+i] + a[ 5*n+i]*b[19*n+i];
    c[ 5*n+i] = b[15*n+i] + a[ 2*n+i]*b[17*n+i] + a[ 3*n+i]*b[18*n+i] + a[ 4*n+i]*b[19*n+i] + a[ 5*n+i]*b[20*n+i];
    c[ 6*n+i] = b[ 1*n+i] + a[ 8*n+i]*b[ 3*n+i] + a[ 9*n+i]*b[ 6*n+i] + a[10*n+i]*b[10*n+i] + a[11*n+i]*b[15*n+i];
    c[ 7*n+i] = b[ 2*n+i] + a[ 8*n+i]*b[ 4*n+i] + a[ 9*n+i]*b[ 7*n+i] + a[10*n+i]*b[11*n+i] + a[11*n+i]*b[16*n+i];
    c[ 8*n+i] = b[ 4*n+i] + a[ 8*n+i]*b[ 5*n+i] + a[ 9*n+i]*b[ 8*n+i] + a[10*n+i]*b[12*n+i] + a[11*n+i]*b[17*n+i];
    c[ 9*n+i] = b[ 7*n+i] + a[ 8*n+i]*b[ 8*n+i] + a[ 9*n+i]*b[ 9*n+i] + a[10*n+i]*b[13*n+i] + a[11*n+i]*b[18*n+i];
    c[10*n+i] = b[11*n+i] + a[ 8*n+i]*b[12*n+i] + a[ 9*n+i]*b[13*n+i] + a[10*n+i]*b[14*n+i] + a[11*n+i]*b[19*n+i];
    c[11*n+i] = b[16*n+i] + a[ 8*n+i]*b[17*n+i] + a[ 9*n+i]*b[18*n+i] + a[10*n+i]*b[19*n+i] + a[11*n+i]*b[20*n+i];
    c[12*n+i] = 0;
    c[13*n+i] = 0;
    c[14*n+i] = 0;
    c[15*n+i] = 0;
    c[16*n+i] = 0;
    c[17*n+i] = 0;
    c[18*n+i] = b[ 6*n+i];
    c[19*n+i] = b[ 7*n+i];
    c[20*n+i] = b[ 8*n+i];
    c[21*n+i] = b[ 9*n+i];
    c[22*n+i] = b[13*n+i];
    c[23*n+i] = b[18*n+i];
    c[24*n+i] = a[26*n+i]*b[ 3*n+i] + a[27*n+i]*b[ 6*n+i] + b[10*n+i] + a[29*n+i]*b[15*n+i];
    c[25*n+i] = a[26*n+i]*b[ 4*n+i] + a[27*n+i]*b[ 7*n+i] + b[11*n+i] + a[29*n+i]*b[16*n+i];
    c[26*n+i] = a[26*n+i]*b[ 5*n+i] + a[27*n+i]*b[ 8*n+i] + b[12*n+i] + a[29*n+i]*b[17*n+i];
    c[27*n+i] = a[26*n+i]*b[ 8*n+i] + a[27*n+i]*b[ 9*n+i] + b[13*n+i] + a[29*n+i]*b[18*n+i];
    c[28*n+i] = a[26*n+i]*b[12*n+i] + a[27*n+i]*b[13*n+i] + b[14*n+i] + a[29*n+i]*b[19*n+i];
    c[29*n+i] = a[26*n+i]*b[17*n+i] + a[27*n+i]*b[18*n+i] + b[19*n+i] + a[29*n+i]*b[20*n+i];
    c[30*n+i] = b[15*n+i];
    c[31*n+i] = b[16*n+i];
    c[32*n+i] = b[17*n+i];
    c[33*n+i] = b[18*n+i];
    c[34*n+i] = b[19*n+i];
    c[35*n+i] = b[20*n+i];
  }

  ) @*/
  {
    cudaDeviceSynchronize();
    /*declare variables*/
    float* dev_a;
    float* dev_b;
    float* dev_c;
    int nthreads=32;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=14;
    /*allocate device memory*/
    cudaMalloc(&dev_c,_N*sizeof(float));
    cudaMalloc(&dev_a,_N*sizeof(float));
    cudaMalloc(&dev_b,_N*sizeof(float));
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
    cudaEventRecord(tstart,0);
    cudaMemcpy(dev_a,a,_N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,_N*sizeof(float),cudaMemcpyHostToDevice);
    cudaEventRecord(tstop,0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&orcu_transfer,tstart,tstop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    orcu_kernel7002<<<dimGrid,dimBlock>>>(n,dev_a,dev_b,dev_c);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    /*copy data from device to host*/
    cudaMemcpy(c,dev_c,_N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    /*free allocated memory*/
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaError_t err=cudaGetLastError();
    if (cudaSuccess!=err) 
      printf("CUDA runtime error: %s@",cudaGetErrorString(err));
  }
  /*@ end @*/
    /*@ end @*/
}
