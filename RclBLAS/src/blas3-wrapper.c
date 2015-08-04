#include "clutil.h"


SEXP Dgemm(SEXP env_exp, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  int na = LENGTH(A), ar = nrows(A), ac = ncols(A);
  int nb = LENGTH(B), br = nrows(B), bc = ncols(B);
  if (ac != br)
    error_return("invalid matrix length\n");
  if (!IS_SCALAR(ALPHA, REALSXP) || !IS_SCALAR(BETA, REALSXP))
    error_return("alpha and beta must be double type scalar\n");
  double alpha = REAL(ALPHA)[0], beta = REAL(BETA)[0];

  if (C == R_NilValue)
  {
    C = PROTECT(allocMatrix(REALSXP, ar, bc));
    beta = 0;
  } else {
    MATRIXCHECK(C, REALSXP);
    if (ar != nrows(C) || bc != ncols(C))
      error_return("invalid matrix length\n");
  }

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_b = create_buffer(env, sizeof(double) * nb);
  cl_mem mem_c = create_buffer(env, sizeof(double) * LENGTH(C));
  write_buffer(env, mem_a, REAL(A), sizeof(double) * na);
  write_buffer(env, mem_b, REAL(B), sizeof(double) * nb);
  if (beta != 0)
    write_buffer(env, mem_c, REAL(C), sizeof(double) * LENGTH(C));
  CHECK(clFinish(env->queues[0]));
  
  cl_event event = NULL;
  cl_int err = clblasDgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                           ar, bc, br, alpha, mem_a, 0, ar, mem_b, 0, br, beta, mem_c, 0, ar,
                           env->num_queues, env->queues, 0, NULL, &event);
  CHECK(clWaitForEvents(1, &event));

  read_buffer(env, mem_c, REAL(C), sizeof(double) * LENGTH(C));
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return C;
}

SEXP Dtrmm(SEXP env_exp, SEXP A, SEXP B, SEXP ALPHA, SEXP UPLO, SEXP DIAG, SEXP SIDE)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  clblasSide side = getSide(SIDE);
  int na = LENGTH(A), ar = nrows(A), ac = ncols(A);
  int nb = LENGTH(B), br = nrows(B), bc = ncols(B);
  if ((side == clblasLeft && ac != br) || (side == clblasRight && bc != ar))
    error_return("invalid matrix length\n");
  int lda = side == clblasLeft ? ac : ar;
  if (!IS_SCALAR(ALPHA, REALSXP))
    error_return("alpha and beta must be double type scalar\n");
  double alpha = REAL(ALPHA)[0];
  
  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_b = create_buffer(env, sizeof(double) * nb);
  write_buffer(env, mem_a, REAL(A), sizeof(double) * na);
  write_buffer(env, mem_b, REAL(B), sizeof(double) * nb);
  CHECK(clFinish(env->queues[0]));
  
  cl_event event = NULL;
  cl_int err = clblasDtrmm(clblasColumnMajor, side, uplo, clblasNoTrans, diag,
                           br, bc, alpha, mem_a, 0, lda, mem_b, 0, br,
                           env->num_queues, env->queues, 0, NULL, &event);
  CHECK(clWaitForEvents(1, &event));

  read_buffer(env, mem_b, REAL(B), sizeof(double) * nb);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return B;
}

SEXP Dtrsm(SEXP env_exp, SEXP A, SEXP B, SEXP ALPHA, SEXP UPLO, SEXP DIAG, SEXP SIDE)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  clblasSide side = getSide(SIDE);
  int na = LENGTH(A), ar = nrows(A), ac = ncols(A);
  int nb = LENGTH(B), br = nrows(B), bc = ncols(B);
  if ((side == clblasLeft && ac != br) || (side == clblasRight && bc != ar))
    error_return("invalid matrix length\n");
  int lda = side == clblasLeft ? ac : ar;
  if (!IS_SCALAR(ALPHA, REALSXP))
    error_return("alpha and beta must be double type scalar\n");
  double alpha = REAL(ALPHA)[0];
  
  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_b = create_buffer(env, sizeof(double) * nb);
  write_buffer(env, mem_a, REAL(A), sizeof(double) * na);
  write_buffer(env, mem_b, REAL(B), sizeof(double) * nb);
  CHECK(clFinish(env->queues[0]));
  
  cl_event event = NULL;
  cl_int err = clblasDtrsm(clblasColumnMajor, side, uplo, clblasNoTrans, diag,
                           br, bc, alpha, mem_a, 0, lda, mem_b, 0, br,
                           env->num_queues, env->queues, 0, NULL, &event);
  CHECK(clWaitForEvents(1, &event));

  read_buffer(env, mem_b, REAL(B), sizeof(double) * nb);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return B;
}

SEXP Dsyrk(SEXP env_exp, SEXP A, SEXP C, SEXP ALPHA, SEXP BETA, SEXP UPLO)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  int na = LENGTH(A), ar = nrows(A), ac = ncols(A);
  if (ar != ac)
    error_return("invalid matrix length\n");
  if (!IS_SCALAR(ALPHA, REALSXP) || !IS_SCALAR(BETA, REALSXP))
    error_return("alpha and beta must be double type scalar\n");
  double alpha = REAL(ALPHA)[0], beta = REAL(BETA)[0];
  
  if (C == R_NilValue)
  {
    C = PROTECT(allocMatrix(REALSXP, ar, ac));
    beta = 0;
  } else {
    MATRIXCHECK(C, REALSXP);
    if (ar != nrows(C) || ac != ncols(C))
      error_return("invalid matrix length\n");
  }
  
  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_c = create_buffer(env, sizeof(double) * LENGTH(C));
  write_buffer(env, mem_a, REAL(A), sizeof(double) * na);
  if (beta != 0)
    write_buffer(env, mem_c, REAL(C), sizeof(double) * LENGTH(C));
  CHECK(clFinish(env->queues[0]));
  
  cl_event event = NULL;
  cl_int err = clblasDsyrk(clblasColumnMajor, uplo, clblasNoTrans, 
                           ncols(C), ac, alpha, mem_a, 0, ac, beta, mem_c, 0, ncols(C),
                           env->num_queues, env->queues, 0, NULL, &event);
  CHECK(clWaitForEvents(1, &event));

  read_buffer(env, mem_c, REAL(C), sizeof(double) * LENGTH(C));
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return C;
}

SEXP Dsyr2k(SEXP env_exp, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA, SEXP UPLO)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  int na = LENGTH(A), ar = nrows(A), ac = ncols(A);
  int nb = LENGTH(B), br = nrows(B), bc = ncols(B);
  if (ar != br || ac != bc)
    error_return("invalid matrix length\n");
  if (!IS_SCALAR(ALPHA, REALSXP) || !IS_SCALAR(BETA, REALSXP))
    error_return("alpha and beta must be double type scalar\n");
  double alpha = REAL(ALPHA)[0], beta = REAL(BETA)[0];
  
  if (C == R_NilValue)
  {
    C = PROTECT(allocMatrix(REALSXP, ar, ac));
    beta = 0;
  } else {
    MATRIXCHECK(C, REALSXP);
    if (ar != nrows(C) || ac != ncols(C))
      error_return("invalid matrix length\n");
  }
  
  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_b = create_buffer(env, sizeof(double) * nb);
  cl_mem mem_c = create_buffer(env, sizeof(double) * LENGTH(C));
  write_buffer(env, mem_a, REAL(A), sizeof(double) * na);
  write_buffer(env, mem_b, REAL(B), sizeof(double) * nb);
  if (beta != 0)
    write_buffer(env, mem_c, REAL(C), sizeof(double) * LENGTH(C));
  CHECK(clFinish(env->queues[0]));
  
  cl_event event = NULL;
  cl_int err = clblasDsyr2k(clblasColumnMajor, uplo, clblasNoTrans, 
                           ncols(C), ac, alpha, mem_a, 0, ac, mem_b, 0, bc, beta, mem_c, 0, ncols(C),
                           env->num_queues, env->queues, 0, NULL, &event);
  CHECK(clWaitForEvents(1, &event));

  read_buffer(env, mem_c, REAL(C), sizeof(double) * LENGTH(C));
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return C;
}

SEXP Dsymm(SEXP env_exp, SEXP A, SEXP B, SEXP C, SEXP ALPHA, SEXP BETA, SEXP UPLO, SEXP SIDE)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  MATRIXCHECK(B, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  clblasSide side = getSide(SIDE);
  int na = LENGTH(A), ar = nrows(A), ac = ncols(A);
  int nb = LENGTH(B), br = nrows(B), bc = ncols(B);
  int lda = side == clblasLeft ? ar : ac;
  if ((side == clblasLeft && ac != br) || (side == clblasRight && bc != ar))
    error_return("invalid matrix length\n");
  if (!IS_SCALAR(ALPHA, REALSXP) || !IS_SCALAR(BETA, REALSXP))
    error_return("alpha and beta must be double type scalar\n");
  double alpha = REAL(ALPHA)[0], beta = REAL(BETA)[0];
  
  if (C == R_NilValue)
  {
    if (side == clblasLeft)
      C = PROTECT(allocMatrix(REALSXP, ar, bc));
    else
      C = PROTECT(allocMatrix(REALSXP, br, ac));
    beta = 0;
  } else {
    MATRIXCHECK(C, REALSXP);
    if ((side == clblasLeft && (nrows(C) != ar || ncols(C) != bc)) ||
        (side == clblasRight && (nrows(C) != br || ncols(C) != ac)))
      error_return("invalid matrix length\n");
  }
  
  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_b = create_buffer(env, sizeof(double) * nb);
  cl_mem mem_c = create_buffer(env, sizeof(double) * LENGTH(C));
  write_buffer(env, mem_a, REAL(A), sizeof(double) * na);
  write_buffer(env, mem_b, REAL(B), sizeof(double) * nb);
  if (beta != 0)
    write_buffer(env, mem_c, REAL(C), sizeof(double) * LENGTH(C));
  CHECK(clFinish(env->queues[0]));
  
  cl_event event = NULL;
  cl_int err = clblasDsymm(clblasColumnMajor, side, uplo,
                           nrows(C), ncols(C), alpha, mem_a, 0, lda, mem_b, 0, br, beta, mem_c, 0, nrows(C),
                           env->num_queues, env->queues, 0, NULL, &event);
  CHECK(clWaitForEvents(1, &event));

  read_buffer(env, mem_c, REAL(C), sizeof(double) * LENGTH(C));
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return C;
}




