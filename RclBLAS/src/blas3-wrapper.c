#include "clutil.h"
#include "routines.h"

SEXP Dgemm(
  SEXP ENV, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA, SEXP TRANSA, SEXP TRANSB)
{
  cl_env *env = get_env(ENV);
  int allocated = 0;
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(BETA, REALSXP);
  clblasTranspose transA = getTrans(TRANSA);
  clblasTranspose transB = getTrans(TRANSB);
  double alpha = SCALARREAL(ALPHA), beta = SCALARREAL(BETA);
  int ar = nrows(A), ac = ncols(A), br = nrows(B), bc = ncols(B), cr, cc;
  if (isNull(C) || LENGTH(C) == 0)
  {
    cr = transA == clblasNoTrans ? ar : ac;
    cc = transB == clblasNoTrans ? bc : br;
    C = PROTECT(allocMatrix(REALSXP, cr, cc));
    beta = 0;
    allocated = 1;
  } else {
    MATRIXCHECK(C, REALSXP);
    cr = nrows(C), cc = ncols(C);
  }

  int size_a = LENGTH(A) * sizeof(double);
  int size_b = LENGTH(B) * sizeof(double);
  int size_c = LENGTH(C) * sizeof(double);
  double *ap = REAL(A), *bp = REAL(B), *cp = REAL(C);
  cl_int err = Dgemm_internal(
    env, ap, bp, cp, alpha, beta, transA, transB, ar, ac, br, bc, cr, cc, size_a, size_b, size_c);
  CHECK(err);

  UNPROTECT(allocated);
  return C;
}

cl_int Dgemm_internal(
  cl_env *env, double *a, double *b, double *c, double alpha, double beta,
  clblasTranspose transA, clblasTranspose transB, 
  int ar, int ac, int br, int bc, int cr, int cc, int size_a, int size_b, int size_c)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  int nevent = 0;
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_b = create_mem(env, b, size_b, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_c;
  if (beta != 0) mem_c = create_mem(env, c, size_c, CL_MEM_READ_WRITE, &(events[nevent++]));
  else mem_c = create_mem(env, NULL, size_c, CL_MEM_READ_WRITE, NULL);
  
  cl_int err = clblasDgemm(clblasColumnMajor, transA, transB,
    ar, bc, ac, alpha, mem_a, 0, ar, mem_b, 0, br, beta, mem_c, 0, cr,
    1, &(env->queues[0]), nevent, events, &(events[nevent]));
  CHECK(err);
  events[nevent+1] = *read_mem(env, mem_c, c, size_c, 1, &(events[nevent]));
  CHECK(clWaitForEvents(1, &(events[nevent+1])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_b));
  CHECK(clReleaseMemObject(mem_c));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dtrmm(SEXP ENV, SEXP A, SEXP B, SEXP ALPHA, SEXP SIDE, SEXP TRANSA, SEXP UPLO, SEXP DIAG)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  clblasSide side = getSide(SIDE);
  clblasTranspose transA = getTrans(TRANSA);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  double alpha = SCALARREAL(ALPHA);
  int ar = nrows(A), ac = ncols(A), br = nrows(B), bc = ncols(B);

  int size_a = LENGTH(A) * sizeof(double);
  int size_b = LENGTH(B) * sizeof(double);
  double *ap = REAL(A), *bp = REAL(B);
  cl_int err = Dtrmm_internal(
    env, ap, bp, alpha, side, transA, uplo, diag, ar, ac, br, bc, size_a, size_b);
  CHECK(err);

  return B;
}

cl_int Dtrmm_internal(
  cl_env *env, double *a, double *b, double alpha, clblasSide side, clblasTranspose transA, 
  clblasUplo uplo, clblasDiag diag, int ar, int ac, int br, int bc, int size_a, int size_b)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  int nevent = 0;
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_b = create_mem(env, b, size_b, CL_MEM_READ_WRITE, &(events[nevent++]));

  cl_int err = clblasDtrmm(clblasColumnMajor, side, uplo, transA, diag,
    br, bc, alpha, mem_a, 0, ar, mem_b, 0, br,
    1, &(env->queues[0]), nevent, events, &(events[nevent]));
  CHECK(err);
  events[nevent+1] = *read_mem(env, mem_b, b, size_b, 1, &(events[nevent]));
  CHECK(clWaitForEvents(1, &(events[nevent+1])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_b));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dtrsm(SEXP ENV, SEXP A, SEXP B, SEXP ALPHA, SEXP SIDE, SEXP TRANSA, SEXP UPLO, SEXP DIAG)
{
  cl_env *env = get_env(ENV);
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  clblasSide side = getSide(SIDE);
  clblasTranspose transA = getTrans(TRANSA);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  double alpha = SCALARREAL(ALPHA);
  int ar = nrows(A), ac = ncols(A), br = nrows(B), bc = ncols(B);

  int size_a = LENGTH(A) * sizeof(double);
  int size_b = LENGTH(B) * sizeof(double);
  double *ap = REAL(A), *bp = REAL(B);
  cl_int err = Dtrsm_internal(
    env, ap, bp, alpha, side, transA, uplo, diag, ar, ac, br, bc, size_a, size_b);
  CHECK(err);

  return B;
}

cl_int Dtrsm_internal(
  cl_env *env, double *a, double *b, double alpha, clblasSide side, clblasTranspose transA, 
  clblasUplo uplo, clblasDiag diag, int ar, int ac, int br, int bc, int size_a, int size_b)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  int nevent = 0;
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_b = create_mem(env, b, size_b, CL_MEM_READ_WRITE, &(events[nevent++]));

  cl_int err = clblasDtrsm(clblasColumnMajor, side, uplo, transA, diag,
    br, bc, alpha, mem_a, 0, ar, mem_b, 0, br,
    1, &(env->queues[0]), nevent, events, &(events[nevent]));
  CHECK(err);
  events[nevent+1] = *read_mem(env, mem_b, b, size_b, 1, &(events[nevent]));
  CHECK(clWaitForEvents(1, &(events[nevent+1])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_b));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dsyrk(SEXP ENV, SEXP A, SEXP C, SEXP ALPHA, SEXP BETA, SEXP TRANSA, SEXP UPLO)
{
  cl_env *env = get_env(ENV);
  int allocated = 0;
  MATRIXCHECK(A, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(BETA, REALSXP);
  clblasTranspose transA = getTrans(TRANSA);
  clblasUplo uplo = getUplo(UPLO);
  double alpha = SCALARREAL(ALPHA), beta = SCALARREAL(BETA);
  int ar = nrows(A), ac = ncols(A), n;
  if (isNull(C) || LENGTH(C) == 0)
  {
    n = transA == clblasNoTrans ? ar : ac;
    C = PROTECT(allocMatrix(REALSXP, n, n));
    beta = 0;
    allocated = 1;
  } else {
    MATRIXCHECK(C, REALSXP);
    n = nrows(C);
  }

  int size_a = LENGTH(A) * sizeof(double);
  int size_c = LENGTH(C) * sizeof(double);
  double *ap = REAL(A), *cp = REAL(C);
  cl_int err = Dsyrk_internal(
    env, ap, cp, alpha, beta, transA, uplo, ar, ac, n, size_a, size_c);
  CHECK(err);

  UNPROTECT(allocated);
  return C;
}

cl_int Dsyrk_internal(
  cl_env *env, double *a, double *c, double alpha, double beta,
  clblasTranspose transA, clblasUplo uplo, int ar, int ac, int n, int size_a, int size_c)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  int nevent = 0;
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_c;
  if (beta != 0) mem_c = create_mem(env, c, size_c, CL_MEM_READ_WRITE, &(events[nevent++]));
  else mem_c = create_mem(env, NULL, size_c, CL_MEM_READ_WRITE, NULL);
  
  int k = transA == clblasNoTrans ? ar : ac;
  cl_int err = clblasDsyrk(clblasColumnMajor, uplo, transA, 
    n, k, alpha, mem_a, 0, ac, beta, mem_c, 0, n,
    1, &(env->queues[0]), nevent, events, &(events[nevent]));
  CHECK(err);
  events[nevent+1] = *read_mem(env, mem_c, c, size_c, 1, &(events[nevent]));
  CHECK(clWaitForEvents(1, &(events[nevent+1])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_c));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dsyr2k(SEXP ENV, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA, SEXP TRANSAB, SEXP UPLO)
{
  cl_env *env = get_env(ENV);
  int allocated = 0;
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(BETA, REALSXP);
  clblasTranspose transAB = getTrans(TRANSAB);
  clblasUplo uplo = getUplo(UPLO);
  double alpha = SCALARREAL(ALPHA), beta = SCALARREAL(BETA);
  int ar = nrows(A), ac = ncols(A), br = nrows(B), bc = ncols(B), cr, cc;
  if (isNull(C) || LENGTH(C) == 0)
  {
    cr = transAB == clblasNoTrans ? ar : ac;
    cc = transAB == clblasNoTrans ? br : bc;
    C = PROTECT(allocMatrix(REALSXP, cr, cc));
    beta = 0;
    allocated = 1;
  } else {
    MATRIXCHECK(C, REALSXP);
    cr = nrows(C), cc = ncols(C);
  }

  int size_a = LENGTH(A) * sizeof(double);
  int size_b = LENGTH(B) * sizeof(double);
  int size_c = LENGTH(C) * sizeof(double);
  double *ap = REAL(A), *bp = REAL(B), *cp = REAL(C);
  cl_int err = Dsyr2k_internal(
    env, ap, bp, cp, alpha, beta, transAB, uplo, ar, ac, br, bc, cr, cc, size_a, size_b, size_c);
  CHECK(err);

  UNPROTECT(allocated);
  return C;
}

cl_int Dsyr2k_internal(
  cl_env *env, double *a, double *b, double *c, double alpha, double beta,
  clblasTranspose transAB, clblasUplo uplo, 
  int ar, int ac, int br, int bc, int cr, int cc, int size_a, int size_b, int size_c)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  int nevent = 0;
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_b = create_mem(env, b, size_b, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_c;
  if (beta != 0) mem_c = create_mem(env, c, size_c, CL_MEM_READ_WRITE, &(events[nevent++]));
  else mem_c = create_mem(env, NULL, size_c, CL_MEM_READ_WRITE, NULL);
  
  int k = transAB == clblasNoTrans ? ac : ar;
  cl_int err = clblasDsyr2k(clblasColumnMajor, uplo, transAB,
    cr, k, alpha, mem_a, 0, ar, mem_b, 0, br, beta, mem_c, 0, cr,
    1, &(env->queues[0]), nevent, events, &(events[nevent]));
  CHECK(err);
  events[nevent+1] = *read_mem(env, mem_c, c, size_c, 1, &(events[nevent]));
  CHECK(clWaitForEvents(1, &(events[nevent+1])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_b));
  CHECK(clReleaseMemObject(mem_c));
  clblasTeardown();
  return CL_SUCCESS;
}

SEXP Dsymm(SEXP ENV, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA, SEXP SIDE, SEXP UPLO)
{
  cl_env *env = get_env(ENV);
  int allocated = 0;
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  SCALARCHECK(ALPHA, REALSXP);
  SCALARCHECK(BETA, REALSXP);
  clblasSide side = getSide(SIDE);
  clblasUplo uplo = getUplo(UPLO);
  double alpha = SCALARREAL(ALPHA), beta = SCALARREAL(BETA);
  int ar = nrows(A), ac = ncols(A), br = nrows(B), bc = ncols(B), cr, cc;
  if (isNull(C) || LENGTH(C) == 0)
  {
    cr = br, cc = bc;
    C = PROTECT(allocMatrix(REALSXP, cr, cc));
    beta = 0;
    allocated = 1;
  } else {
    MATRIXCHECK(C, REALSXP);
    cr = nrows(C), cc = ncols(C);
  }

  int size_a = LENGTH(A) * sizeof(double);
  int size_b = LENGTH(B) * sizeof(double);
  int size_c = LENGTH(C) * sizeof(double);
  double *ap = REAL(A), *bp = REAL(B), *cp = REAL(C);
  cl_int err = Dsymm_internal(
    env, ap, bp, cp, alpha, beta, side, uplo, ar, ac, br, bc, cr, cc, size_a, size_b, size_c);
  CHECK(err);

  UNPROTECT(allocated);
  return C;
}

cl_int Dsymm_internal(
  cl_env *env, double *a, double *b, double *c, double alpha, double beta,
  clblasSide side, clblasUplo uplo,
  int ar, int ac, int br, int bc, int cr, int cc, int size_a, int size_b, int size_c)
{
  CHECK(clblasSetup());
  cl_event events[NEVENTS];
  int nevent = 0;
  cl_mem mem_a = create_mem(env, a, size_a, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_b = create_mem(env, b, size_b, CL_MEM_READ_ONLY, &(events[nevent++]));
  cl_mem mem_c;
  if (beta != 0) mem_c = create_mem(env, c, size_c, CL_MEM_READ_WRITE, &(events[nevent++]));
  else mem_c = create_mem(env, NULL, size_c, CL_MEM_READ_WRITE, NULL);
  
  cl_int err = clblasDsymm(clblasColumnMajor, side, uplo, 
    cr, cc, alpha, mem_a, 0, ar, mem_b, 0, br, beta, mem_c, 0, cr,
    1, &(env->queues[0]), nevent, events, &(events[nevent]));
  CHECK(err);
  events[nevent+1] = *read_mem(env, mem_c, c, size_c, 1, &(events[nevent]));
  CHECK(clWaitForEvents(1, &(events[nevent+1])));
  CHECK(clReleaseMemObject(mem_a));
  CHECK(clReleaseMemObject(mem_b));
  CHECK(clReleaseMemObject(mem_c));
  clblasTeardown();
  return CL_SUCCESS;
}





