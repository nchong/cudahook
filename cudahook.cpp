#include <stdio.h>
#include <dlfcn.h>
#include <cassert>
#include <map>
#include <list>

#include <cuda.h>
#include <vector_types.h>

typedef struct {
  dim3 gridDim;
  dim3 blockDim;
  int counter;
  std::list<void *> args;
} kernelInfo_t;

kernelInfo_t &kernelInfo() {
  static kernelInfo_t _kernelInfo;
  return _kernelInfo;
}

std::map<const char *, char *> &kernels() {
  static std::map<const char*, char*> _kernels;
  return _kernels;
}

void print_kernel_invocation(const char *entry) {
  dim3 gridDim = kernelInfo().gridDim;
  dim3 blockDim = kernelInfo().blockDim;
  printf("SENTINEL %s ", kernels()[entry]);
  if (gridDim.y == 1 && gridDim.z == 1) {
    printf("--gridDim=%d ", gridDim.x);
  } else if (gridDim.z == 1) {
    printf("--gridDim=[%d,%d] ", gridDim.x, gridDim.y);
  } else {
    printf("--gridDim=[%d,%d,%d] ", gridDim.x, gridDim.y, gridDim.z);
  }
  if (blockDim.y == 1 && blockDim.z == 1) {
    printf("--blockDim=%d ", blockDim.x);
  } else if (blockDim.z == 1) {
    printf("--blockDim=[%d,%d] ", blockDim.x, blockDim.y);
  } else {
    printf("--blockDim=[%d,%d,%d] ", blockDim.x, blockDim.y, blockDim.z);
  }
  for (std::list<void *>::iterator it = kernelInfo().args.begin(),
       end = kernelInfo().args.end();
       it != end;
       ++it) {
    unsigned i = std::distance(kernelInfo().args.begin(), it);
    printf("%d:%d ", i, *(static_cast<int *>(*it)));
  }
  printf("\n");
}

typedef cudaError_t (*cudaConfigureCall_t)(dim3,dim3,size_t,cudaStream_t);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

extern "C"
cudaError_t cudaConfigureCall(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem=0,
  cudaStream_t stream=0) {
  assert(kernelInfo().counter == 0 && "Multiple cudaConfigureCalls before cudaLaunch?");
  kernelInfo().gridDim = gridDim;
  kernelInfo().blockDim = blockDim;
  kernelInfo().counter++;
  if (realCudaConfigureCall == NULL)
    realCudaConfigureCall = (cudaConfigureCall_t)dlsym(RTLD_NEXT,"cudaConfigureCall");
  assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
  return realCudaConfigureCall(gridDim,blockDim,sharedMem,stream);
}

typedef cudaError_t (*cudaLaunch_t)(const char *);
static cudaLaunch_t realCudaLaunch = NULL;

extern "C"
cudaError_t cudaLaunch(const char *entry) {
  assert(kernelInfo().counter == 1 && "Multiple cudaConfigureCalls before cudaLaunch?");
  print_kernel_invocation(entry);
  kernelInfo().counter--; kernelInfo().args.clear();
  if (realCudaLaunch == NULL) {
    realCudaLaunch = (cudaLaunch_t)dlsym(RTLD_NEXT,"cudaLaunch");
  }
  assert(realCudaLaunch != NULL && "cudaLaunch is null");
  return realCudaLaunch(entry);
}

typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
                                         const char *, int, uint3 *,
                                         uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C"
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid,
                            uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
  kernels()[hostFun] = deviceFun;
  if (realCudaRegisterFunction == NULL) {
    realCudaRegisterFunction = (cudaRegisterFunction_t)dlsym(RTLD_NEXT,"__cudaRegisterFunction");
  }
  assert(realCudaRegisterFunction != NULL && "cudaRegisterFunction is null");
  realCudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                           deviceName, thread_limit, tid,
                           bid, bDim, gDim, wSize);
}

typedef cudaError_t (*cudaSetupArgument_t)(const void *, size_t, size_t);
static cudaSetupArgument_t realCudaSetupArgument = NULL;

extern "C"
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  kernelInfo().args.push_back(const_cast<void *>(arg));
  if (realCudaSetupArgument == NULL) {
    realCudaSetupArgument = (cudaSetupArgument_t)dlsym(RTLD_NEXT,"cudaSetupArgument");
  }
  assert(realCudaSetupArgument != NULL);
  return realCudaSetupArgument(arg, size, offset);
}
