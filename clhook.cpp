#include <stdio.h>
#include <dlfcn.h>
#include <cassert>
#include <map>
#include <vector>
#include <iostream>

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

typedef struct {
  const char *name;
  unsigned work_dim;
  unsigned global_work_size[3];
  unsigned local_work_size[3];
  std::vector<void *> args;
} kernelInfo_t;

std::map<cl_kernel, kernelInfo_t> &kernels() {
  static std::map<cl_kernel, kernelInfo_t> _kernels;
  return _kernels;
}

void print_kernel_invocation(cl_kernel entry) {
  assert(kernels().count(entry) == 1 && "Kernel not found(2)");
  kernelInfo_t *info = &kernels()[entry];
  unsigned work_dim = info->work_dim;
  unsigned *global_work_size = info->global_work_size;
  unsigned *local_work_size = info->local_work_size;
  printf("\nSENTINEL %s ", info->name);
  if (work_dim == 1) {
    printf("--global_size=%d ", global_work_size[0]);
    printf("--local_size=%d ", local_work_size[0]);
  } else if (work_dim == 2) {
    printf("--global_size=[%d,%d] ", global_work_size[0], global_work_size[1]);
    printf("--local_size=[%d,%d] ", local_work_size[0], local_work_size[1]);
  } else if (work_dim == 3) {
    printf("--global_size=[%d,%d,%d] ", global_work_size[0], global_work_size[1], global_work_size[2]);
    printf("--local_size=[%d,%d,%d] ", local_work_size[0], local_work_size[1], local_work_size[2]);
  }
  for (std::vector<void *>::iterator it = info->args.begin(),
       end = info->args.end();
       it != end;
       ++it) {
    unsigned i = std::distance(info->args.begin(), it);
    int *x = static_cast<int *>(*it);
    if (x) printf("%d:%d ", i, *x);
    else printf("%d:- ", i);
  }
  printf("\n");
}

typedef cl_kernel (*clCreateKernel_t)(cl_program,const char *,cl_int *);
static clCreateKernel_t realClCreateKernel = NULL;
extern "C"
cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
  if (realClCreateKernel == NULL)
    realClCreateKernel = (clCreateKernel_t)dlsym(RTLD_NEXT,"clCreateKernel");
  assert(realClCreateKernel != NULL && "clCreateKernel is null");
  cl_kernel k = realClCreateKernel(program, kernel_name, errcode_ret);
  kernels()[k].name = kernel_name;
  return k;
}

typedef cl_int (*clSetKernelArg_t)(cl_kernel,cl_uint,size_t,const void *);
static clSetKernelArg_t realClSetKernelArg = NULL;
extern "C"
cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index,
                      size_t arg_size, const void *arg_value) {
  assert(kernels().count(kernel) == 1 && "Kernel not found(0)");
  if (kernels()[kernel].args.size() < arg_index+1) {
		kernels()[kernel].args.resize(arg_index + 1);
  }
  if (arg_value) {
    kernels()[kernel].args[arg_index] = const_cast<void *>(arg_value);
  } else {
    kernels()[kernel].args[arg_index] = NULL;
  }
  if (realClSetKernelArg == NULL)
    realClSetKernelArg = (clSetKernelArg_t)dlsym(RTLD_NEXT,"clSetKernelArg");
  assert(realClSetKernelArg != NULL && "clSetKernelArg is null");
  return realClSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

typedef cl_int (*clEnqueueNDRangeKernel_t)(
  cl_command_queue,cl_kernel,cl_uint,
  const size_t *,const size_t *,const size_t *,
  cl_uint,const cl_event *,cl_event *);
static clEnqueueNDRangeKernel_t realClEnqueueNDRangeKernel = NULL;
extern "C"
cl_int clEnqueueNDRangeKernel(
  cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
  const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size,
  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  assert(kernels().count(kernel) == 1 && "Kernel not found(1)");
  assert(work_dim <= 3 && "work_dim is too large");
  kernels()[kernel].work_dim = work_dim;
  for (unsigned int i=0; i<work_dim; i++) {
    kernels()[kernel].global_work_size[i] = global_work_size[i];
    kernels()[kernel].local_work_size[i] = local_work_size == NULL ? 0 : local_work_size[i];
  }
  print_kernel_invocation(kernel);
  if (realClEnqueueNDRangeKernel == NULL)
    realClEnqueueNDRangeKernel = (clEnqueueNDRangeKernel_t)dlsym(RTLD_NEXT,"clEnqueueNDRangeKernel");
  assert(realClEnqueueNDRangeKernel != NULL && "clEnqueueNDRangeKernel is null");
  return realClEnqueueNDRangeKernel(
    command_queue, kernel, work_dim,
    global_work_offset, global_work_size, local_work_size,
    num_events_in_wait_list, event_wait_list, event);
}

typedef cl_program (*clCreateProgramWithSource_t)(
  cl_context,cl_uint,const char **, const size_t *, cl_int *);
static clCreateProgramWithSource_t realClCreateProgramWithSource = NULL;
extern "C"
cl_program clCreateProgramWithSource(
  cl_context context,
  cl_uint count,
  const char **strings,
  const size_t *lengths,
  cl_int *errcode_ret) {
  if (realClCreateProgramWithSource == NULL)
    realClCreateProgramWithSource = (clCreateProgramWithSource_t)dlsym(RTLD_NEXT,"clCreateProgramWithSource");
  assert(realClCreateProgramWithSource != NULL && "realClCreateProgramWithSource is null");
	std::vector<std::string> code(count);
	if (lengths == NULL) { // all strings are null-terminated
		for (unsigned i = 0; i < count; i++) {
			std::string line (strings[i]);
			code.push_back(line);
		}
  } else { // lengths contains length of each entry in strings
		for (unsigned i = 0; i < count; i++) {
			if (lengths[i] == 0) { // entry is null-terminated
				std::string line (strings[i]);
				code.push_back(line);
			} else { // entry has specified length
				std::string line (strings[i], lengths[i]);
				code.push_back(line);
			}
		}
  }
  std::cerr << "\nSENTINEL (program intercepted)" << std::endl;
  for (unsigned i = 0; i < code.size(); i++) {
    std::cerr << code[i];
  }

  return realClCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}
