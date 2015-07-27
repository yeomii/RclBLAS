#ifndef __R_CLBLAS__
#define __R_CLBLAS__

#include <CL/opencl.h>
#include <clBLAS.h>
#include <R.h>
#include <Rinternals.h>

#define CHECK(error) { \
  if (error != CL_SUCCESS) { \
    Rf_error("Error in clutil.c line :%d, error code :%d\n", __LINE__, error); \
  } \
} \

typedef struct cl_env {
  cl_platform_id *platform;
  cl_int num_devices;
  cl_device_type device_type; 
  cl_device_id *devices;
  cl_context *context;
  cl_int num_queues;
  cl_command_queue *queues;
  cl_event *events;
} cl_env;


cl_mem create_buffer(cl_env *env, int size);
void read_buffer(cl_env *env, cl_mem mem, void *ptr, size_t size);
void write_buffer(cl_env *env, cl_mem mem, void *ptr, size_t size);

#endif
