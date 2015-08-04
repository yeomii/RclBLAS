#include "clutil.h"

void env_finalizer(SEXP ref)
{
  cl_env *env = (cl_env *)R_ExternalPtrAddr(ref);
  for (int i = 0; i < env->num_queues; i++)
    clReleaseCommandQueue(env->queues[i]);
  clReleaseContext(*(env->context));
  if (env->platform != NULL) free(env->platform);
  if (env->devices != NULL) free(env->devices);
  if (env->queues != NULL) free(env->queues);
  //if (env->events != NULL) free(env->events);
  free(env);
}

cl_device_type get_device_type(SEXP dev_type)
{
  if (TYPEOF(dev_type) != STRSXP || LENGTH(dev_type) != 1)
    Rf_error("device type must be one character - c for cpu, g for gpu, otherwise all type");
  char *d = CHAR(STRING_ELT(dev_type, 0));
  if (d[0] == 'c')
    return CL_DEVICE_TYPE_CPU;
  else if (d[0] == 'g')
    return CL_DEVICE_TYPE_GPU;
  else
    return CL_DEVICE_TYPE_ALL;
}

int get_num_devices(SEXP dev_num)
{
  if (TYPEOF(dev_num) != REALSXP && TYPEOF(dev_num) != INTSXP)
    Rf_error("number of device must be numeric vector");
  if (LENGTH(dev_num) < 1)
    Rf_error("dev_num contains nothing");
  if (TYPEOF(dev_num) == REALSXP)
    return (int)(REAL(dev_num)[0]);
  else
    return (int)(INTEGER(dev_num)[0]);
}

SEXP create_env(SEXP dev_type, SEXP dev_num)
{
  cl_int err;
  cl_uint np;
  
  CHECK(clGetPlatformIDs(0, NULL, &np));
  if (np <= 0)
  {
    Rf_error("No platform found\n");
    return R_NilValue;
  }
  cl_env *env = (cl_env *)malloc(sizeof(cl_env));
  SEXP ext = PROTECT(R_MakeExternalPtr(env, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ext, env_finalizer, TRUE);
  Rf_setAttrib(ext, R_ClassSymbol, mkString("cl_env"));
  env->platform = (cl_platform_id *)malloc(sizeof(cl_platform_id)*np);
  // TODO : enable platform selection
  CHECK(clGetPlatformIDs(1, env->platform, NULL));
  
  env->device_type = get_device_type(dev_type);
  env->num_devices = get_num_devices(dev_num);
  env->devices = (cl_device_id *)malloc(sizeof(cl_device_id) * env->num_devices);
  CHECK(clGetDeviceIDs(env->platform[0], env->device_type, env->num_devices, env->devices, NULL));
  env->context = (cl_context *)malloc(sizeof(cl_context));
  *(env->context) = clCreateContext(NULL, env->num_devices, env->devices, NULL, NULL, &err);
  CHECK(err);
  env->num_queues = env->num_devices;
  env->queues = (cl_command_queue *)malloc(sizeof(cl_command_queue) * env->num_queues);
  for (int i = 0; i < env->num_queues; i++)
  {
    env->queues[i] = clCreateCommandQueue(*(env->context), env->devices[i], 0, &err);
    CHECK(err);
  }
  UNPROTECT(1);
  return ext;
}

cl_mem create_buffer(cl_env *env, int size)
{
  cl_int err;
  cl_mem mem = clCreateBuffer( *(env->context), CL_MEM_READ_WRITE, size, NULL, &err);
  CHECK(err);
  return mem;
}

void read_buffer(cl_env *env, cl_mem mem, void *ptr, size_t size)
{
  cl_int err = clEnqueueReadBuffer( env->queues[0], mem, CL_TRUE, 0, size, ptr, 0, NULL, NULL );
  CHECK(err);
}

void write_buffer(cl_env *env, cl_mem mem, void *ptr, size_t size)
{
  cl_int err = clEnqueueWriteBuffer( env->queues[0], mem, CL_TRUE, 0, size, ptr, 0, NULL, NULL );
  CHECK(err);
}

cl_env* get_env(SEXP env_exp)
{
  if (TYPEOF(env_exp) != EXTPTRSXP || !inherits(env_exp, "cl_env"))
    Rf_error("invalid environment argument");
  return (cl_env *)R_ExternalPtrAddr(env_exp);
}

SEXP get_type(SEXP x)
{
  printf("type is %d\n", TYPEOF(x));
  return R_NilValue;
}

clblasUplo getUplo(SEXP UPLO)
{
  char c;
  if (!IS_SCALAR(UPLO, STRSXP) || ((c = *CHAR(STRING_ELT(UPLO, 0))) != 'u' && c != 'l') )
    Rf_error("uplo must be character \'u\' or \'l\'\n");
  return c == 'u' ? clblasUpper : clblasLower;
}

clblasDiag getDiag(SEXP DIAG)
{
  char c;
  if (!IS_SCALAR(DIAG, STRSXP) || ((c = *CHAR(STRING_ELT(DIAG, 0))) != 'u' && c != 'n') )
    Rf_error("diag must be character \'u\' or \'n\'\n");
  return c == 'u' ? clblasUnit : clblasNonUnit;
}

clblasSide getSide(SEXP SIDE)
{
  char c;
  if (!IS_SCALAR(SIDE, STRSXP) || ((c = *CHAR(STRING_ELT(SIDE, 0))) != 'l' && c != 'r') )
    Rf_error("side must be character \'l\' or \'r\'\n");
  return c == 'l' ? clblasLeft : clblasRight;
}
