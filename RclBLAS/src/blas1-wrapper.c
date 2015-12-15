#include "clutil.h"
#include "routines.h"

SEXP Dswap(
  SEXP ENV, SEXP X, SEXP Y, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(OFFY, INTSXP);
  SCALARCHECK(INCY, INTSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int offy = SCALARINT(OFFY), incy = SCALARINT(INCY);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);
  if (n < 0) n = (LENGTH(X) - offx) / incx;

  double *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Dswap_internal(env, xp, yp, 
    size_x, size_y, n, offx, incx, offy, incy);
  CHECK(err);
  
  return X;
}

cl_int Dswap_internal(
  cl_env *env, double *x, double *y, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_WRITE, &(events[0]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_WRITE, &(events[1]));

  cl_int err = clblasDswap(n, mem_x, offx, incx, mem_y, offy, incy,
                           1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_x, x, size_x, 1, &(events[2]));
  events[4] = *read_mem(env, mem_y, y, size_y, 1, &(events[2]));
  CHECK(clWaitForEvents(2, &(events[3])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dscal(
  SEXP ENV, SEXP X, SEXP ALPHA, SEXP N, SEXP OFFX, SEXP INCX)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  double alpha = SCALARREAL(ALPHA);
  if (n < 0) n = (LENGTH(X) - offx) / incx;
  int size_x = LENGTH(X) * sizeof(double);
  double *xp = REAL(X);
  
  cl_int err = Dscal_internal(env, xp, alpha, size_x, n, offx, incx);
  CHECK(err);
  return X;
}

cl_int Dscal_internal(
  cl_env *env, double* x, double alpha, int size_x,
  int n, int offx, int incx)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_WRITE, &(events[0]));

  cl_int err = clblasDscal(n, alpha, mem_x, offx, incx, 
                           1, &(env->queues[0]), 1, events, &(events[1]));
  CHECK(err);
  events[2] = *read_mem(env, mem_x, x, size_x, 1, &(events[1]));
  CHECK(clWaitForEvents(1, &(events[2])));
  CHECK(clReleaseMemObject(mem_x));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dcopy(
  SEXP ENV, SEXP X, SEXP Y, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY)
{
  cl_env *env = get_env(ENV);
  int allocated = 0;
  TYPECHECK(X, REALSXP);
  if (isNull(Y) || LENGTH(Y) == 0)
  {
    Y = PROTECT(allocVector(REALSXP, LENGTH(X)));
    allocated++;
  }
  else
    TYPECHECK(Y, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(OFFY, INTSXP);
  SCALARCHECK(INCY, INTSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int offy = SCALARINT(OFFY), incy = SCALARINT(INCY);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);
  if (n < 0) n = (LENGTH(X) - offx) / incx;

  double *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Dcopy_internal(
    env, xp, yp, size_x, size_y, n, offx, incx, offy, incy);
  CHECK(err);
  
  UNPROTECT(allocated);
  return X;
}

cl_int Dcopy_internal(
  cl_env *env, double *x, double *y, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_WRITE_ONLY, &(events[1]));

  cl_int err = clblasDcopy(n, mem_x, offx, incx, mem_y, offy, incy,
                           1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_y, y, size_y, 1, &(events[2]));
  CHECK(clWaitForEvents(1, &(events[3])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}


SEXP Daxpy(
  SEXP ENV, SEXP X, SEXP Y, SEXP ALPHA, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(OFFY, INTSXP);
  SCALARCHECK(INCY, INTSXP);
  SCALARCHECK(ALPHA, REALSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int offy = SCALARINT(OFFY), incy = SCALARINT(INCY);
  double alpha = SCALARREAL(ALPHA);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);
  if (n < 0) n = (LENGTH(X) - offx) / incx;

  double *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Daxpy_internal(env, xp, yp, alpha,
    size_x, size_y, n, offx, incx, offy, incy);
  CHECK(err);
  
  return Y;
}

cl_int Daxpy_internal(
  cl_env *env, double *x, double *y, double alpha, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_WRITE, &(events[1]));

  cl_int err = clblasDaxpy(n, alpha, mem_x, offx, incx, mem_y, offy, incy,
                           1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_y, y, size_y, 1, &(events[2]));
  CHECK(clWaitForEvents(1, &(events[3])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Ddot(
  SEXP ENV, SEXP X, SEXP Y, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY, SEXP SCRATCH)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(OFFY, INTSXP);
  SCALARCHECK(INCY, INTSXP);
  SCALARCHECK(SCRATCH, INTSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int offy = SCALARINT(OFFY), incy = SCALARINT(INCY);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);
  int size_scratch = SCALARINT(SCRATCH) * sizeof(double);
  if (n < 0) n = (LENGTH(X) - offx) / incx;
  if (size_scratch < 0) size_scratch = n * sizeof(double);

  SEXP DOT = PROTECT(allocVector(REALSXP, 1));
  double *xp = REAL(X), *yp = REAL(Y), *dot = REAL(DOT);
  cl_int err = Ddot_internal(env, xp, yp, dot,
    size_x, size_y, n, offx, incx, offy, incy, size_scratch);
  CHECK(err);
  
  UNPROTECT(1);
  return DOT;
}

cl_int Ddot_internal(
  cl_env *env, double *x, double *y, double *res, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy, int size_scratch)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_ONLY, &(events[1]));
  cl_mem mem_res = create_mem(env, NULL, sizeof(double), CL_MEM_WRITE_ONLY, NULL);
  cl_mem mem_scratch = create_mem(env, NULL, size_scratch, CL_MEM_READ_WRITE, NULL);

  cl_int err = clblasDdot(n, mem_res, 0, mem_x, offx, incx, mem_y, offy, incy, mem_scratch,
                          1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_res, res, sizeof(double), 1, &(events[2]));
  CHECK(clWaitForEvents(1, &(events[3])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  CHECK(clReleaseMemObject(mem_res));
  CHECK(clReleaseMemObject(mem_scratch));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Drotg(
  SEXP ENV, SEXP A, SEXP B, SEXP C, SEXP S)
{
  cl_env *env = get_env(ENV);
  SCALARCHECK(A, REALSXP);
  SCALARCHECK(B, REALSXP);
  SCALARCHECK(C, REALSXP);
  SCALARCHECK(S, REALSXP);

  double *ap = REAL(A), *bp = REAL(B);
  double *cp = REAL(C), *sp = REAL(S);

  cl_int err = Drotg_internal(env, ap, bp, cp, sp);
  CHECK(err);
  
  return R_NilValue;
}

cl_int Drotg_internal(
  cl_env *env, double *a, double *b, double *c, double *s)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, sizeof(double), CL_MEM_READ_WRITE, &(events[0]));
  cl_mem mem_b = create_mem(env, b, sizeof(double), CL_MEM_READ_WRITE, &(events[1]));
  cl_mem mem_c = create_mem(env, c, sizeof(double), CL_MEM_READ_WRITE, &(events[2]));
  cl_mem mem_s = create_mem(env, s, sizeof(double), CL_MEM_READ_WRITE, &(events[3]));

  cl_int err = clblasDrotg(mem_a, 0, mem_b, 0, mem_c, 0, mem_s, 0, 
                           1, &(env->queues[0]), 4, events, &(events[4]));
  CHECK(err);
  events[5] = *read_mem(env, mem_a, a, sizeof(double), 1, &(events[4]));
  events[6] = *read_mem(env, mem_b, b, sizeof(double), 1, &(events[4]));
  events[7] = *read_mem(env, mem_c, c, sizeof(double), 1, &(events[4]));
  events[8] = *read_mem(env, mem_s, s, sizeof(double), 1, &(events[4]));
  CHECK(clWaitForEvents(4, &(events[5])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_b));
  CHECK(clReleaseMemObject(mem_c));
  CHECK(clReleaseMemObject(mem_s));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Drot(
  SEXP ENV, SEXP X, SEXP Y, SEXP C, SEXP S, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(OFFY, INTSXP);
  SCALARCHECK(INCY, INTSXP);
  SCALARCHECK(C, REALSXP);
  SCALARCHECK(S, REALSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int offy = SCALARINT(OFFY), incy = SCALARINT(INCY);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);
  double c = SCALARREAL(C), s = SCALARREAL(S);
  if (n < 0) n = (LENGTH(X) - offx) / incx;

  double *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Drot_internal(env, xp, yp, c, s,
    size_x, size_y, n, offx, incx, offy, incy);
  CHECK(err);
  
  return R_NilValue;
}
cl_int Drot_internal(
  cl_env *env, double *x, double *y, double c, double s, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_WRITE, &(events[0]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_WRITE, &(events[1]));

  cl_int err = clblasDrot(n, mem_x, offx, incx, mem_y, offy, incy, c, s,
                          1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_x, x, size_x, 1, &(events[2]));
  events[4] = *read_mem(env, mem_y, y, size_y, 1, &(events[2]));
  CHECK(clWaitForEvents(2, &(events[3])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Drotmg(
  SEXP ENV, SEXP D1, SEXP D2, SEXP X, SEXP Y, SEXP PARAM)
{
  cl_env *env = get_env(ENV);
  SCALARCHECK(D1, REALSXP);
  SCALARCHECK(D2, REALSXP);
  SCALARCHECK(X, REALSXP);
  SCALARCHECK(Y, REALSXP);
  TYPECHECK(PARAM, REALSXP);

  double *d1 = REAL(D1), *d2 = REAL(D2);
  double *xp = REAL(X), *yp = REAL(Y);
  double *param = REAL(PARAM);
  int size_param = LENGTH(PARAM) * sizeof(double);

  cl_int err = Drotmg_internal(env, d1, d2, xp, yp, param, size_param);
  CHECK(err);
  
  return R_NilValue;
}
cl_int Drotmg_internal(
  cl_env *env, double *d1, double *d2, double *x, double *y, 
  double *param, int size_param)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_d1 = create_mem(env, d1, sizeof(double), CL_MEM_READ_WRITE, &(events[0]));
  cl_mem mem_d2 = create_mem(env, d2, sizeof(double), CL_MEM_READ_WRITE, &(events[1]));
  cl_mem mem_x = create_mem(env, x, sizeof(double), CL_MEM_READ_WRITE, &(events[2]));
  cl_mem mem_y = create_mem(env, y, sizeof(double), CL_MEM_READ_ONLY, &(events[3]));
  cl_mem mem_p = create_mem(env, param, size_param, CL_MEM_READ_WRITE, &(events[4]));

  cl_int err = clblasDrotmg(mem_d1, 0, mem_d2, 0, mem_x, 0, mem_y, 0, mem_p, 0,
                           1, &(env->queues[0]), 5, events, &(events[5]));
  CHECK(err);
  events[6] = *read_mem(env, mem_d1, d1, sizeof(double), 1, &(events[5]));
  events[7] = *read_mem(env, mem_d2, d2, sizeof(double), 1, &(events[5]));
  events[8] = *read_mem(env, mem_x, x, sizeof(double), 1, &(events[5]));
  events[9] = *read_mem(env, mem_p, param, size_param, 1, &(events[5]));
  CHECK(clWaitForEvents(4, &(events[5])));
  CHECK(clReleaseMemObject(mem_d1));
  CHECK(clReleaseMemObject(mem_d2));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  CHECK(clReleaseMemObject(mem_p));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Drotm(
  SEXP ENV, SEXP X, SEXP Y, SEXP PARAM, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY, SEXP OFFPARAM)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  TYPECHECK(PARAM, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(OFFY, INTSXP);
  SCALARCHECK(INCY, INTSXP);
  SCALARCHECK(OFFPARAM, INTSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int offy = SCALARINT(OFFY), incy = SCALARINT(INCY);
  int offparam = SCALARINT(OFFPARAM);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);
  int size_param = LENGTH(PARAM) * sizeof(double);
  if (n < 0) n = (LENGTH(X) - offx) / incx;

  double *xp = REAL(X), *yp = REAL(Y), *param = REAL(PARAM);
  cl_int err = Drotm_internal(env, xp, yp, param, 
    size_x, size_y, size_param, n, offx, incx, offy, incy, offparam);
  CHECK(err);
  
  return R_NilValue;
}

cl_int Drotm_internal(
  cl_env *env, double *x, double *y, double *param, int size_x, int size_y, int size_param,
  int n, int offx, int incx, int offy, int incy, int offparam)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_WRITE, &(events[0]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_WRITE, &(events[1]));
  cl_mem mem_p = create_mem(env, param, size_param, CL_MEM_READ_ONLY, &(events[2]));
  
  cl_int err = clblasDrotm(n, mem_x, offx, incx, mem_y, offy, incy, mem_p, offparam,
                           1, &(env->queues[0]), 3, events, &(events[3]));
  CHECK(err);
  events[4] = *read_mem(env, mem_x, x, size_x, 1, &(events[3]));
  events[5] = *read_mem(env, mem_y, y, size_x, 1, &(events[3]));
  CHECK(clWaitForEvents(2, &(events[4])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  CHECK(clReleaseMemObject(mem_p));
  clblasTeardown();
  return CL_SUCCESS;
}


SEXP Dnrm2(
  SEXP ENV, SEXP X, SEXP N, SEXP OFFX, SEXP INCX, SEXP SCRATCH)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(SCRATCH, INTSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int size_x = LENGTH(X) * sizeof(double);
  int size_scratch = SCALARINT(SCRATCH) * sizeof(double);
  if (n < 0) n = (LENGTH(X) - offx) / incx;
  if (size_scratch < 0) size_scratch = 2 * n * sizeof(double);

  SEXP NRM = PROTECT(allocVector(REALSXP, 1));
  double *xp = REAL(X), *nrm = REAL(NRM);
  cl_int err = Dnrm2_internal(env, xp, nrm, size_x, 
    n, offx, incx, size_scratch);
  CHECK(err);
  UNPROTECT(1);
  return NRM;
}

cl_int Dnrm2_internal(
  cl_env *env, double *x, double *nrm, int size_x, 
  int n, int offx, int incx, int size_scratch)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_res = create_mem(env, NULL, sizeof(double), CL_MEM_WRITE_ONLY, NULL);
  cl_mem mem_scratch = create_mem(env, NULL, size_scratch, CL_MEM_READ_WRITE, NULL);

  cl_int err = clblasDnrm2(n, mem_res, 0, mem_x, offx, incx, mem_scratch,
                           1, &(env->queues[0]), 1, events, &(events[1]));
  CHECK(err);
  events[2] = *read_mem(env, mem_res, nrm, sizeof(double), 1, &(events[1]));
  CHECK(clWaitForEvents(1, &(events[2])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_res));
  CHECK(clReleaseMemObject(mem_scratch));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP iDamax(
  SEXP ENV, SEXP X,SEXP N, SEXP OFFX, SEXP INCX, SEXP SCRATCH)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(SCRATCH, INTSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int size_x = LENGTH(X) * sizeof(double);
  int size_scratch = SCALARINT(SCRATCH) * sizeof(double);
  if (n < 0) n = (LENGTH(X) - offx) / incx;
  if (size_scratch < 0) size_scratch = 2 * n * sizeof(double);

  cl_uint idx;
  SEXP IDX = PROTECT(allocVector(INTSXP, 1));
  double *xp = REAL(X); int *ip = INTEGER(IDX);
  cl_int err = iDamax_internal(env, xp, &idx, size_x, 
    n, offx, incx, size_scratch);
  CHECK(err);
  *ip = (int)idx;
  UNPROTECT(1);
  return IDX;
}
cl_int iDamax_internal(
  cl_env *env, double *x, cl_uint *idx, int size_x,
  int n, int offx, int incx, int size_scratch)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_res = create_mem(env, NULL, sizeof(cl_uint), CL_MEM_WRITE_ONLY, NULL);
  cl_mem mem_scratch = create_mem(env, NULL, size_scratch, CL_MEM_READ_WRITE, NULL);

  cl_int err = clblasiDamax(n, mem_res, 0, mem_x, offx, incx, mem_scratch,
                           1, &(env->queues[0]), 1, events, &(events[1]));
  CHECK(err);
  events[2] = *read_mem(env, mem_res, idx, sizeof(cl_uint), 1, &(events[1]));
  CHECK(clWaitForEvents(1, &(events[2])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_res));
  CHECK(clReleaseMemObject(mem_scratch));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dasum(
  SEXP ENV, SEXP X, SEXP N, SEXP OFFX, SEXP INCX, SEXP SCRATCH)
{
  cl_env *env = get_env(ENV);
  TYPECHECK(X, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(OFFX, INTSXP);
  SCALARCHECK(INCX, INTSXP);
  SCALARCHECK(SCRATCH, INTSXP);
  int n = SCALARINT(N);
  int offx = SCALARINT(OFFX), incx = SCALARINT(INCX);
  int size_x = LENGTH(X) * sizeof(double);
  int size_scratch = SCALARINT(SCRATCH) * sizeof(double);
  if (n < 0) n = (LENGTH(X) - offx) / incx;
  if (size_scratch < 0) size_scratch = 2 * n * sizeof(double);

  SEXP ASUM = PROTECT(allocVector(REALSXP, 1));
  double *xp = REAL(X), *asum = REAL(ASUM);
  cl_int err = Dasum_internal(env, xp, asum, size_x, 
    n, offx, incx, size_scratch);
  CHECK(err);
  UNPROTECT(1);
  return ASUM;
}

cl_int Dasum_internal(
  cl_env *env, double *x, double *asum, int size_x,
  int n, int offx, int incx, int size_scratch)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_res = create_mem(env, NULL, sizeof(double), CL_MEM_WRITE_ONLY, NULL);
  cl_mem mem_scratch = create_mem(env, NULL, size_scratch, CL_MEM_READ_WRITE, NULL);

  cl_int err = clblasDasum(n, mem_res, 0, mem_x, offx, incx, mem_scratch,
                           1, &(env->queues[0]), 1, events, &(events[1]));
  CHECK(err);
  events[2] = *read_mem(env, mem_res, asum, sizeof(double), 1, &(events[1]));
  CHECK(clWaitForEvents(1, &(events[2])));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_res));
  CHECK(clReleaseMemObject(mem_scratch));
  clblasTeardown();
  return CL_SUCCESS;
}
