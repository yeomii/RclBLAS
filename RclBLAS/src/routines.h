#ifndef __ROUTINES_H__
#define __ROUTINES_H__

#include <R.h>
#include <Rinternals.h>


/* RclBLAS interface for level 1 BLAS routines */

SEXP Dswap(
  SEXP ENV, SEXP X, SEXP Y, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY);
cl_int Dswap_internal(
  cl_env *env, double *x, double *y, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy);

SEXP Dscal(
  SEXP ENV, SEXP X, SEXP ALPHA, SEXP N, SEXP OFFX, SEXP INCX);
cl_int Dscal_internal(
  cl_env *env, double *x, double alpha, int size_x,
  int n, int offx, int incx);

SEXP Dcopy(
  SEXP ENV, SEXP X, SEXP Y, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY);
cl_int Dcopy_internal(
  cl_env *env, double *x, double *y, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy);

SEXP Daxpy(
  SEXP ENV, SEXP X, SEXP Y, SEXP ALPHA, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY);
cl_int Daxpy_internal(
  cl_env *env, double *x, double *y, double alpha, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy);

SEXP Ddot(
  SEXP ENV, SEXP X, SEXP Y, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY, SEXP SCRATCH);
cl_int Ddot_internal(
  cl_env *env, double *x, double *y, double *res, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy, int size_scratch);

SEXP Drotg(
  SEXP ENV, SEXP A, SEXP B, SEXP C, SEXP S);
cl_int Drotg_internal(
  cl_env *env, double *a, double *b, double *c, double *s);

SEXP Drot(
  SEXP ENV, SEXP X, SEXP Y, SEXP C, SEXP S, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY);
cl_int Drot_internal(
  cl_env *env, double *x, double *y, double c, double s, int size_x, int size_y,
  int n, int offx, int incx, int offy, int incy);

SEXP Drotmg(
  SEXP ENV, SEXP D1, SEXP D2, SEXP X, SEXP Y, SEXP PARAM);
cl_int Drotmg_internal(
  cl_env *env, double *d1, double *d2, double *x, double *y, double *param, int size_param);

SEXP Drotm(
  SEXP ENV, SEXP X, SEXP Y, SEXP PARAM, SEXP N, SEXP OFFX, SEXP INCX, SEXP OFFY, SEXP INCY, SEXP OFFPARAM);
cl_int Drotm_internal(
  cl_env *env, double *x, double *y, double *param, int size_x, int size_y, int size_param,
  int n, int offx, int incx, int offy, int incy, int offparam);

SEXP Dnrm2(
  SEXP ENV, SEXP X, SEXP N, SEXP OFFX, SEXP INCX, SEXP SCRATCH);
cl_int Dnrm2_internal(
  cl_env *env, double *x, double *nrm, int size_x, 
  int n, int offx, int incx, int size_scratch);

SEXP iDamax(
  SEXP ENV, SEXP X, SEXP N, SEXP OFFX, SEXP INCX, SEXP SCRATCH);
cl_int iDamax_internal(
  cl_env *env, double *x, cl_uint *idx, int size_x,
  int n, int offx, int incx, int size_scratch);

SEXP Dasum(
  SEXP ENV, SEXP X, SEXP N, SEXP OFFX, SEXP INCX, SEXP SCRATCH);
cl_int Dasum_internal(
  cl_env *env, double *x, double *asum, int size_x,
  int n, int offx, int incx, int size_scratch);

/* RclBLAS interface for level 2 BLAS routines */

SEXP Dgemv(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP ALPHA, SEXP BETA, SEXP TRANSA);
cl_int Dgemv_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, double beta,
  clblasTranspose transA, int ar, int ac, int size_a, int size_x, int size_y);

SEXP Dsymv(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP ALPHA, SEXP BETA, SEXP UPLO);
cl_int Dsymv_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, double beta,
  clblasUplo uplo, int ar, int ac, int size_a, int size_x, int size_y);

SEXP Dtrmv(
  SEXP ENV, SEXP A, SEXP X, SEXP TRANSA, SEXP UPLO, SEXP DIAG, SEXP SCRATCH);
cl_int Dtrmv_internal(
  cl_env *env, double *a, double *x, clblasTranspose transA, clblasUplo uplo,
  clblasDiag diag, int n, int ar, int size_a, int size_x, int size_scratch);

SEXP Dtrsv(
  SEXP ENV, SEXP A, SEXP X, SEXP TRANSA, SEXP UPLO, SEXP DIAG);
cl_int Dtrsv_internal(
  cl_env *env, double *a, double *x, clblasTranspose transA, clblasUplo uplo,
  clblasDiag diag, int n, int ar, int size_a, int size_x);

SEXP Dger(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP ALPHA);
cl_int Dger_internal(
  cl_env *env, double *a, double *x, double *y, double alpha,
  int ar, int ac, int size_a, int size_x, int size_y);

SEXP Dsyr(
  SEXP ENV, SEXP A, SEXP X, SEXP ALPHA, SEXP UPLO);
cl_int Dsyr_internal(
  cl_env *env, double *a, double *x, double alpha, clblasUplo uplo,
  int ar, int ac, int size_a, int size_x);

SEXP Dsyr2(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP ALPHA, SEXP UPLO);
cl_int Dsyr2_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, clblasUplo uplo,
  int ar, int ac, int size_a, int size_x, int size_y);

SEXP Dgbmv(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP M, SEXP N, SEXP KL, SEXP KU, SEXP ALPHA, SEXP BETA, SEXP TRANSA);
cl_int Dgbmv_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, double beta, int m, int n, int kl, int ku,
  clblasTranspose transA, int size_a, int size_x, int size_y);

SEXP Dtbmv(
  SEXP ENV, SEXP A, SEXP X, SEXP N, SEXP K, SEXP TRANSA, SEXP UPLO, SEXP DIAG, SEXP SCRATCH);
cl_int Dtbmv_internal(
  cl_env *env, double *a, double *x, int n, int k, clblasTranspose transA, clblasUplo uplo, clblasDiag diag,
  int size_a, int size_x, int size_scratch);

SEXP Dsbmv(
  SEXP ENV, SEXP A, SEXP X, SEXP Y, SEXP N, SEXP K, SEXP ALPHA, SEXP BETA, SEXP UPLO);
cl_int Dsbmv_internal(
  cl_env *env, double *a, double *x, double *y, double alpha, double beta, int n, int k,
  clblasUplo uplo, int size_a, int size_x, int size_y);

SEXP Dtbsv(
  SEXP ENV, SEXP A, SEXP X, SEXP N, SEXP K, SEXP TRANSA, SEXP UPLO, SEXP DIAG);
cl_int Dtbsv_internal(cl_env *env, double *a, double *x, int n, int k, 
  clblasTranspose transA, clblasUplo uplo, clblasDiag diag, int size_a, int size_x);

/* RclBLAS interface for level 3 BLAS routines */

SEXP Dgemm(
  SEXP ENV, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA, SEXP TRANSA, SEXP TRANSB);
cl_int Dgemm_internal(
  cl_env *env, double *a, double *b, double *c, double alpha, double beta,
  clblasTranspose transA, clblasTranspose transB, 
  int ar, int ac, int br, int bc, int cr, int cc, int size_a, int size_b, int size_c);

SEXP Dtrmm(
  SEXP ENV, SEXP A, SEXP B, SEXP ALPHA, SEXP SIDE, SEXP TRANSA, SEXP UPLO, SEXP DIAG);
cl_int Dtrmm_internal(
  cl_env *env, double *a, double *b, double alpha, clblasSide side, clblasTranspose transA, 
  clblasUplo uplo, clblasDiag diag, int ar, int ac, int br, int bc, int size_a, int size_b);

SEXP Dtrsm(
  SEXP ENV, SEXP A, SEXP B, SEXP ALPHA, SEXP SIDE, SEXP TRANSA, SEXP UPLO, SEXP DIAG);
cl_int Dtrsm_internal(
  cl_env *env, double *a, double *b, double alpha, clblasSide side, clblasTranspose transA, 
  clblasUplo uplo, clblasDiag diag, int ar, int ac, int br, int bc, int size_a, int size_b);

SEXP Dsyrk(
  SEXP ENV, SEXP A, SEXP C, SEXP ALPHA, SEXP BETA, SEXP TRANSA, SEXP UPLO);
cl_int Dsyrk_internal(
  cl_env *env, double *a, double *c, double alpha, double beta,
  clblasTranspose transA, clblasUplo uplo, int ar, int ac, int n, int size_a, int size_c);

SEXP Dsyr2k(
  SEXP ENV, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA, SEXP TRANSAB, SEXP UPLO);

cl_int Dsyr2k_internal(
  cl_env *env, double *a, double *b, double *c, double alpha, double beta,
  clblasTranspose transAB, clblasUplo uplo, 
  int ar, int ac, int br, int bc, int cr, int cc, int size_a, int size_b, int size_c);

SEXP Dsymm(
  SEXP ENV, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA, SEXP SIDE, SEXP UPLO);
cl_int Dsymm_internal(
  cl_env *env, double *a, double *b, double *c, double alpha, double beta,
  clblasSide side, clblasUplo uplo,
  int ar, int ac, int br, int bc, int cr, int cc, int size_a, int size_b, int size_c);

#endif
