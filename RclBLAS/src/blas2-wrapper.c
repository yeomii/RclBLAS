#include "clutil.h"

#define DEBUG() {fprintf(stderr, "%d\n", __LINE__);}

SEXP Dgemv(SEXP env_exp, SEXP A, SEXP X, SEXP Y, SEXP alpha, SEXP beta)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  if (!IS_SCALAR(alpha, REALSXP) || !IS_SCALAR(beta, REALSXP))
    Rf_error("alpha and beta must be a double scalar\n");
  int nx = LENGTH(X), ny = LENGTH(Y), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ac != nx || ar != ny)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  double da = REAL(alpha)[0], db = REAL(beta)[0];

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDgemv(clblasColumnMajor, clblasNoTrans, 
                           ar, ac, da, mem_a, 0, ar, mem_x, 0, 1, db, mem_y, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return Y;
}

SEXP Dsymv(SEXP env_exp, SEXP A, SEXP X, SEXP Y, SEXP alpha, SEXP beta, SEXP UPLO)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  clblasUplo uplo = getUplo(UPLO); 
  if (!IS_SCALAR(alpha, REALSXP) || !IS_SCALAR(beta, REALSXP))
    Rf_error("alpha and beta must be a double scalar\n");
  int nx = LENGTH(X), ny = LENGTH(Y), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ac != nx || ar != ny || ac != ar)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  double da = REAL(alpha)[0], db = REAL(beta)[0];

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDsymv(clblasColumnMajor, uplo,
                           ar, da, mem_a, 0, ar, mem_x, 0, 1, db, mem_y, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return Y;
}

SEXP Dtrmv(SEXP env_exp, SEXP A, SEXP X, SEXP UPLO, SEXP DIAG)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  int nx = LENGTH(X), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ac != nx || ac != ar)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X);

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem scratch = create_buffer(env, sizeof(double) * ar);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDtrmv(clblasColumnMajor, uplo, clblasNoTrans, diag, 
                           ar, mem_a, 0, ar, mem_x, 0, 1, scratch,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  clReleaseMemObject(scratch);
  return X;
}

SEXP Dtrsv(SEXP env_exp, SEXP A, SEXP X, SEXP UPLO, SEXP DIAG)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  int nx = LENGTH(X), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ac != nx || ac != ar)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X);

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));

  cl_int err = clblasDtrsv(clblasColumnMajor, uplo, clblasNoTrans, diag, 
                           ar, mem_a, 0, ar, mem_x, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return X;
}

SEXP Dger(SEXP env_exp, SEXP A, SEXP X, SEXP Y, SEXP alpha)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  if (!IS_SCALAR(alpha, REALSXP))
    Rf_error("alpha and beta must be a double scalar\n");
  int nx = LENGTH(X), ny = LENGTH(Y), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ar != nx || ac != ny)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  double da = REAL(alpha)[0];

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDger(clblasColumnMajor,
                          ar, ac, da, mem_x, 0, 1, mem_y, 0, 1, mem_a, 0, ar,
                          env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_a, ap, sizeof(double) * na);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return A;
}

SEXP Dsyr(SEXP env_exp, SEXP A, SEXP X, SEXP alpha, SEXP UPLO)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  if (!IS_SCALAR(alpha, REALSXP))
    Rf_error("alpha and beta must be a double scalar\n");
  int nx = LENGTH(X), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ar != ac || ac != nx)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X);
  double da = REAL(alpha)[0];

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDsyr(clblasColumnMajor, uplo,
                          ar, da, mem_x, 0, 1, mem_a, 0, ar,
                          env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_a, ap, sizeof(double) * na);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return A;
}

SEXP Dsyr2(SEXP env_exp, SEXP A, SEXP X, SEXP Y, SEXP alpha, SEXP UPLO)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  if (!IS_SCALAR(alpha, REALSXP))
    Rf_error("alpha and beta must be a double scalar\n");
  int nx = LENGTH(X), ny = LENGTH(Y), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ar != ac || ar != nx || ar != ny)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  double da = REAL(alpha)[0];

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDsyr2(clblasColumnMajor, uplo,
                          ar, da, mem_x, 0, 1, mem_y, 0, 1, mem_a, 0, ar,
                          env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_a, ap, sizeof(double) * na);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return A;
}

SEXP Dgbmv(SEXP env_exp, SEXP A, SEXP X, SEXP Y, SEXP alpha, SEXP beta)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  if (!IS_SCALAR(alpha, REALSXP) || !IS_SCALAR(beta, REALSXP))
    Rf_error("alpha and beta must be a double scalar\n");
  int nx = LENGTH(X), ny = LENGTH(Y), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ar != ac || ac != nx || ar != ny)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  double da = REAL(alpha)[0], db = REAL(beta)[0];
  
  int kl = 0, ku = 0;
  for (int i = 0; i < ar; i++)
  {
    if (ap[i * ac] == 0)
      ku++;
    if (ap[(i + 1) * ac - 1] == 0)
      kl++;
  }

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDgbmv(clblasColumnMajor, clblasNoTrans,
                           ar, ac, kl, ku, da, mem_a, 0, ar, mem_x, 0, 1, db, mem_y, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return Y;
}

SEXP Dtbmv(SEXP env_exp, SEXP A, SEXP X, SEXP UPLO, SEXP DIAG) 
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  int nx = LENGTH(X), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ar != ac || ac != nx)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X);
  
  int k = 0;
  for (int i = 0; i < ar; i++)
    if (ap[i * ac] == 0)
      k++;

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem scratch = create_buffer(env, sizeof(double) * ar);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDtbmv(clblasColumnMajor, uplo, clblasNoTrans, diag,
                           ar, k, mem_a, 0, ar, mem_x, 0, 1, scratch,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  clReleaseMemObject(scratch);
  return X;
}

SEXP Dsbmv(SEXP env_exp, SEXP A, SEXP X, SEXP Y, SEXP alpha, SEXP beta, SEXP UPLO)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  int nx = LENGTH(X), ny = LENGTH(Y), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ar != ac || ac != nx || ar != ny)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X), *yp = REAL(Y);
  double da = REAL(alpha)[0], db = REAL(beta)[0];
  
  int k = 0;
  for (int i = 0; i < ar; i++)
    if (ap[i * ac] == 0)
      k++;

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDsbmv(clblasColumnMajor, uplo, 
                           ar, k, da, mem_a, 0, ar, mem_x, 0, 1, db, mem_y, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return Y;
}

SEXP Dtbsv(SEXP env_exp, SEXP A, SEXP X, SEXP UPLO, SEXP DIAG)
{
  cl_env *env = get_env(env_exp);
  MATRIXCHECK(A, REALSXP);
  TYPECHECK(X, REALSXP);
  clblasUplo uplo = getUplo(UPLO);
  clblasDiag diag = getDiag(DIAG);
  int nx = LENGTH(X), na = LENGTH(A);
  int ac = ncols(A), ar = nrows(A);
  if (ar != ac || ac != nx)
    Rf_error("invalid matrix length\n");
  double *ap = REAL(A), *xp = REAL(X);
  
  int k = 0;
  for (int i = 0; i < ar; i++)
    if (ap[i * ac] == 0)
      k++;

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double) * na);
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  write_buffer(env, mem_a, ap, sizeof(double) * na);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDtbsv(clblasColumnMajor, uplo, clblasNoTrans, diag, 
                           ar, k, mem_a, 0, ar, mem_x, 0, 1, 
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return X;
}
