#ifndef __R_CLBLAS__
#define __R_CLBLAS__

#include <CL/opencl.h>
#include <clBLAS.h>
#include <R.h>
#include <Rinternals.h>

#define CHECK(error) { \
  if (error != CL_SUCCESS) { \
    Rf_error("Error in RclBLAS/src/%s, line :%d, error code :%d\n", __FILE__, __LINE__, error); \
  } \
} \

#define TYPECHECK(exp, type) { \
  if (TYPEOF(exp) != type) { \
    Rf_error("Error in RclBLAS/src/%s, line :%d, invalid type %d, expected type is %d\n", \
             __FILE__, __LINE__, TYPEOF(exp), type); \
    return R_NilValue; \
  } \
} \

#define MATRIXCHECK(exp, type) { \
  TYPECHECK(exp, type) \
  if (!isMatrix(exp)) {\
    Rf_error("Error in RclBLAS/src/%s, line :%d, invalid exp, expected type is matrix\n", \
             __FILE__, __LINE__); \
    return R_NilValue; \
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
  cl_uint num_events;
  cl_event *events;
} cl_env;


cl_mem create_buffer(cl_env *env, int size);
void read_buffer(cl_env *env, cl_mem mem, void *ptr, size_t size);
void write_buffer(cl_env *env, cl_mem mem, void *ptr, size_t size);
cl_env* get_env(SEXP env_exp);
clblasUplo getUplo(SEXP UPLO);
clblasDiag getDiag(SEXP DIAG);
clblasSide getSide(SEXP SIDE);
#endif
