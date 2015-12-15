#include "clutil.h"
#include "routines.h"

#define DEBUG() {fprintf(stderr, "%d\n", __LINE__);}

SEXP Dgemv(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP ALPHA, SEXP BETA, SEXP TRANSA)
{
  cl_env *env = get_env(ENV);
  int allocated = 0;
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(BETA, REALSXP);
  clblasTranspose transA = getTrans(TRANSA);
  double alpha = SCALARREAL(ALPHA), beta = SCALARREAL(BETA);
  int ar = nrows(A), ac = ncols(A);

  if (isNull(Y) || LENGTH(Y) == 0)
  {
    int ny = (transA == clblasNoTrans ? ar : ac);
    Y = PROTECT(allocVector(REALSXP, ny));
    allocated++; beta = 0;
  }
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);
  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Dgemv_internal(
    env, ap, xp, yp, alpha, beta, transA, ar, ac, size_a, size_x, size_y);
  CHECK(err);

  UNPROTECT(allocated);
  return Y;
}

cl_int Dgemv_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, double beta,
  clblasTranspose transA, int ar, int ac, int size_a, int size_x, int size_y)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[1]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_WRITE, &(events[2]));

  cl_int err = clblasDgemv(clblasColumnMajor, transA,
    ar, ac, alpha, mem_a, 0, ar, mem_x, 0, 1, beta, mem_y, 0, 1,
    1, &(env->queues[0]), 3, events, &(events[3]));
  CHECK(err);
  events[4] = *read_mem(env, mem_y, y, size_y, 1, &(events[3]));
  CHECK(clWaitForEvents(1, &(events[4])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dsymv(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP ALPHA, SEXP BETA, SEXP UPLO)
{
  cl_env *env = get_env(ENV);
  int allocated = 0;
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(BETA, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  double alpha = SCALARREAL(ALPHA), beta = SCALARREAL(BETA);
  int ar = nrows(A), ac = ncols(A);

  if (isNull(Y) || LENGTH(Y) == 0)
  {
    Y = PROTECT(allocVector(REALSXP, ar));
    allocated++; beta = 0;
  }
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);
  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Dsymv_internal(
    env, ap, xp, yp, alpha, beta, uplo, ar, ac, size_a, size_x, size_y);
  CHECK(err);

  UNPROTECT(allocated);
  return Y;
}

cl_int Dsymv_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, double beta,
  clblasUplo uplo, int ar, int ac, int size_a, int size_x, int size_y)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[1]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_WRITE, &(events[2]));

  cl_int err = clblasDsymv(clblasColumnMajor, uplo,
    ar, alpha, mem_a, 0, ar, mem_x, 0, 1, beta, mem_y, 0, 1,
    1, &(env->queues[0]), 3, events, &(events[3]));
  CHECK(err);
  events[4] = *read_mem(env, mem_y, y, size_y, 1, &(events[3]));
  CHECK(clWaitForEvents(1, &(events[4])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dtrmv(
  SEXP ENV, SEXP A, SEXP X, SEXP TRANSA, SEXP UPLO, SEXP DIAG, SEXP SCRATCH)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  SCALARCHECK(SCRATCH, INTSXP);
  clblasTranspose transA = getTrans(TRANSA);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  int n = (nrows(A) < ncols(A)) ? nrows(A) : ncols(A), ar = nrows(A);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);
  int size_scratch = SCALARINT(SCRATCH) * sizeof(double);
  if (size_scratch <= 0) size_scratch = n * sizeof(double);
  //printf("%d %d %d\n", size_a, size_x, size_scratch);

  double *ap = REAL(A), *xp = REAL(X);
  cl_int err = Dtrmv_internal(
    env, ap, xp, transA, uplo, diag, n, ar, size_a, size_x, size_scratch);
  CHECK(err);
  return X;
}

cl_int Dtrmv_internal(
  cl_env *env, double *a, double *x, clblasTranspose transA, clblasUplo uplo,
  clblasDiag diag, int n, int ar, int size_a, int size_x, int size_scratch)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_WRITE, &(events[1]));
  cl_mem mem_scratch = create_mem(env, NULL, size_scratch, CL_MEM_WRITE_ONLY, NULL);

  cl_int err = clblasDtrmv(clblasColumnMajor, uplo, transA, diag,
    n, mem_a, 0, ar, mem_x, 0, 1, mem_scratch,
    1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_x, x, size_x, 1, &(events[2]));
  CHECK(clWaitForEvents(1, &(events[3])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_scratch));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dtrsv(
  SEXP ENV, SEXP A, SEXP X, SEXP TRANSA, SEXP UPLO, SEXP DIAG)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  clblasTranspose transA = getTrans(TRANSA);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  int n = (nrows(A) < ncols(A)) ? nrows(A) : ncols(A), ar = nrows(A);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);

  double *ap = REAL(A), *xp = REAL(X);
  cl_int err = Dtrsv_internal(
    env, ap, xp, transA, uplo, diag, n, ar, size_a, size_x);
  CHECK(err);
  return X;
}

cl_int Dtrsv_internal(
  cl_env *env, double *a, double *x, clblasTranspose transA, clblasUplo uplo,
  clblasDiag diag, int n, int ar, int size_a, int size_x)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_WRITE, &(events[1]));

  cl_int err = clblasDtrsv(clblasColumnMajor, uplo, transA, diag, 
                           n, mem_a, 0, ar, mem_x, 0, 1,
                           1, &(env->queues[0]), 2, events, &(events[2]));
  
  CHECK(err);
  events[3] = *read_mem(env, mem_x, x, size_x, 1, &(events[2]));
  CHECK(clWaitForEvents(1, &(events[3])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dger(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP ALPHA)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  double alpha = SCALARREAL(ALPHA);
  int ar = nrows(A), ac = ncols(A);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);

  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Dger_internal(env, ap, xp, yp, alpha, ar, ac, size_a, size_x, size_y);
  CHECK(err);
  return A;
}

cl_int Dger_internal(
  cl_env *env, double *a, double *x, double *y, double alpha,
  int ar, int ac, int size_a, int size_x, int size_y)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_WRITE, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[1]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_ONLY, &(events[2]));

  cl_int err = clblasDger(clblasColumnMajor,
    ar, ac, alpha, mem_x, 0, 1, mem_y, 0, 1, mem_a, 0, ar,
    1, &(env->queues[0]), 3, events, &(events[3]));
  CHECK(err);
  events[4] = *read_mem(env, mem_a, a, size_a, 1, &(events[3]));
  CHECK(clWaitForEvents(1, &(events[4])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dsyr(
  SEXP ENV, SEXP A, SEXP X, SEXP ALPHA, SEXP UPLO)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  double alpha = SCALARREAL(ALPHA);
  int ar = nrows(A), ac = ncols(A);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);

  double *ap = REAL(A), *xp = REAL(X);
  cl_int err = Dsyr_internal(env, ap, xp, alpha, uplo, ar, ac, size_a, size_x);
  CHECK(err);
  return A;
}

cl_int Dsyr_internal(
  cl_env *env, double *a, double *x, double alpha, clblasUplo uplo,
  int ar, int ac, int size_a, int size_x)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_WRITE, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[1]));

  cl_int err = clblasDsyr(clblasColumnMajor, uplo,
    ac, alpha, mem_x, 0, 1, mem_a, 0, ar, 
    1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_a, a, size_a, 1, &(events[2]));
  CHECK(clWaitForEvents(1, &(events[3])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dsyr2(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP ALPHA, SEXP UPLO)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  double alpha = SCALARREAL(ALPHA);
  int ar = nrows(A), ac = ncols(A);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);

  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Dsyr2_internal(env, ap, xp, yp, alpha, uplo, ar, ac, size_a, size_x, size_y);
  CHECK(err);
  return A;
}

cl_int Dsyr2_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, clblasUplo uplo,
  int ar, int ac, int size_a, int size_x, int size_y)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_WRITE, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[1]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_ONLY, &(events[2]));

  cl_int err = clblasDsyr2(clblasColumnMajor, uplo, 
    ac, alpha, mem_x, 0, 1, mem_y, 0, 1, mem_a, 0, ar,
    1, &(env->queues[0]), 3, events, &(events[3]));
  CHECK(err);
  events[4] = *read_mem(env, mem_a, a, size_a, 1, &(events[3]));
  CHECK(clWaitForEvents(1, &(events[4])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dgbmv(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP M, SEXP N, SEXP KL, SEXP KU, SEXP ALPHA, SEXP BETA, SEXP TRANSA)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(BETA, REALSXP);
  SCALARCHECK(KU, INTSXP);
  SCALARCHECK(KL, INTSXP);
  clblasTranspose transA = getTrans(TRANSA);
  double alpha = SCALARREAL(ALPHA);
  double beta = SCALARREAL(BETA);
  int m = SCALARINT(M);
  int n = SCALARINT(N);
  int ku = SCALARINT(KU);
  int kl = SCALARINT(KL);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);

  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Dgbmv_internal(env, ap, xp, yp, alpha, beta, m, n, kl, ku, transA, size_a, size_x, size_y);
  CHECK(err);
  return Y;
}

cl_int Dgbmv_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, double beta, int m, int n, int kl, int ku,
  clblasTranspose transA, int size_a, int size_x, int size_y)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[1]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_WRITE, &(events[2]));

  // lda is set to kl + ku + 1
  cl_int err = clblasDgbmv(clblasColumnMajor, transA, 
    m, n, kl, ku, alpha, mem_a, 0, kl+ku+1, mem_x, 0, 1, beta, mem_y, 0, 1,
    1, &(env->queues[0]), 3, events, &(events[3]));
  CHECK(err);
  events[4] = *read_mem(env, mem_y, y, size_y, 1, &(events[3]));
  CHECK(clWaitForEvents(1, &(events[4])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

// test시 주의

SEXP Dtbmv(
  SEXP ENV, SEXP A, SEXP X, SEXP N, SEXP K, SEXP TRANSA, SEXP UPLO, SEXP DIAG, SEXP SCRATCH)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(K, INTSXP);
  SCALARCHECK(SCRATCH, INTSXP);
  clblasTranspose transA = getTrans(TRANSA);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  int n = SCALARINT(N);
  int k = SCALARINT(K);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);
  int size_scratch = SCALARINT(SCRATCH) * sizeof(double);
  if (size_scratch <= 0) size_scratch = n * sizeof(double);

  double *ap = REAL(A), *xp = REAL(X);
  cl_int err = Dtbmv_internal(env, ap, xp, n, k, transA, uplo, diag, size_a, size_x, size_scratch);
  CHECK(err);
  return X;
}

cl_int Dtbmv_internal(
  cl_env *env, double *a, double *x, int n, int k, clblasTranspose transA, clblasUplo uplo, clblasDiag diag,
  int size_a, int size_x, int size_scratch)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_WRITE, &(events[1]));
  cl_mem mem_scratch = create_mem(env, NULL, size_scratch, CL_MEM_WRITE_ONLY, NULL);

  // lda is set to k + 1
  cl_int err = clblasDtbmv(clblasColumnMajor, uplo, transA, diag, 
    n, k, mem_a, 0, k+1, mem_x, 0, 1, mem_scratch,
    1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_x, x, size_x, 1, &(events[2]));
  CHECK(clWaitForEvents(1, &(events[3])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_scratch));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dsbmv(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP N, SEXP K, SEXP ALPHA, SEXP BETA, SEXP UPLO)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(BETA, REALSXP);
  SCALARCHECK(K, INTSXP);
  SCALARCHECK(N, INTSXP);
  clblasUplo uplo = getUplo(UPLO);
  double alpha = SCALARREAL(ALPHA);
  double beta = SCALARREAL(BETA);
  int n = SCALARINT(N);
  int k = SCALARINT(K);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);
  int size_y = LENGTH(Y) * sizeof(double);

  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  cl_int err = Dsbmv_internal(env, ap, xp, yp, alpha, beta, n, k, uplo, size_a, size_x, size_y);
  CHECK(err);
  return Y;
}

cl_int Dsbmv_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, double beta, int n, int k,
  clblasUplo uplo, int size_a, int size_x, int size_y)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_ONLY, &(events[1]));
  cl_mem mem_y = create_mem(env, y, size_y, CL_MEM_READ_WRITE, &(events[2]));

  // lda is set to k + 1
  cl_int err = clblasDsbmv(clblasColumnMajor, uplo, 
    n, k, alpha, mem_a, 0, k+1, mem_x, 0, 1, beta, mem_y, 0, 1,
    1, &(env->queues[0]), 3, events, &(events[3]));
  CHECK(err);
  events[4] = *read_mem(env, mem_y, y, size_y, 1, &(events[3]));
  CHECK(clWaitForEvents(1, &(events[4])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  CHECK(clReleaseMemObject(mem_y));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dtbsv(
  SEXP ENV, SEXP A, SEXP X, SEXP N, SEXP K, SEXP TRANSA, SEXP UPLO, SEXP DIAG)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  SCALARCHECK(N, INTSXP);
  SCALARCHECK(K, INTSXP);
  clblasTranspose transA = getTrans(TRANSA);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  int n = SCALARINT(N);
  int k = SCALARINT(K);
  int size_a = LENGTH(A) * sizeof(double);
  int size_x = LENGTH(X) * sizeof(double);

  double *ap = REAL(A), *xp = REAL(X);
  cl_int err = Dtbsv_internal(env, ap, xp, n, k, transA, uplo, diag, size_a, size_x);
  CHECK(err);
  return X;
}

cl_int Dtbsv_internal(cl_env *env, double *a, double *x, int n, int k, 
  clblasTranspose transA, clblasUplo uplo, clblasDiag diag, int size_a, int size_x)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[0]));
  cl_mem mem_x = create_mem(env, x, size_x, CL_MEM_READ_WRITE, &(events[1]));

  // lda is set to k + 1
  cl_int err = clblasDtbsv(clblasColumnMajor, uplo, transA, diag, 
    n, k, mem_a, 0, k+1, mem_x, 0, 1, 
    1, &(env->queues[0]), 2, events, &(events[2]));
  CHECK(err);
  events[3] = *read_mem(env, mem_x, x, size_x, 1, &(events[2]));
  CHECK(clWaitForEvents(1, &(events[3])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_x));
  clblasTeardown();
  return CL_SUCCESS;
}
