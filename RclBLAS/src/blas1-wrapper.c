#include "clutil.h"

SEXP Dswap(SEXP env_exp, SEXP X, SEXP Y)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  int nx = LENGTH(X), ny = LENGTH(Y);
  if (nx != ny || nx == 0)
    Rf_error("length of x and y must be the same\n");
  double *xp = REAL(X), *yp = REAL(Y);
  
  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDswap(nx, mem_x, 0, 1, mem_y, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double) * nx);
  read_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0])); 
  clblasTeardown();
  return X;
}

SEXP Dscal(SEXP env_exp, SEXP alpha, SEXP X)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  if (!IS_SCALAR(alpha, REALSXP))
    Rf_error("alpha must be a double scalar\n");
  int nx = LENGTH(X);
  double *xp = REAL(X), a = REAL(alpha)[0];

  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDscal(nx, a, mem_x, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i = 0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return X;
}

SEXP Dcopy(SEXP env_exp, SEXP X)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  int nx = LENGTH(X);
  SEXP Y = PROTECT(allocVector(REALSXP, nx));
  double *xp = REAL(X), *yp = REAL(Y);

  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * nx);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDcopy(nx, mem_x, 0, 1, mem_y, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_y, yp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  UNPROTECT(1);
  return Y;
}

SEXP Daxpy(SEXP env_exp, SEXP alpha, SEXP X, SEXP Y)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  if (!IS_SCALAR(alpha, REALSXP))
    Rf_error("alpha must be a double scalar\n");
  int nx = LENGTH(X), ny = LENGTH(Y);
  if (nx != ny || nx == 0)
    Rf_error("length of x and y must be the same\n");
  double *xp = REAL(X), *yp = REAL(Y), a = REAL(alpha)[0];
  
  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDaxpy(nx, a, mem_x, 0, 1, mem_y, 0, 1,
                           env->num_queues, env->queues, 0, NULL, NULL);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return Y;
}

SEXP Ddot(SEXP env_exp, SEXP X, SEXP Y)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  int nx = LENGTH(X), ny = LENGTH(Y);
  if (nx != ny || nx == 0)
    Rf_error("length of x and y must be the same\n");
  SEXP Z = PROTECT(allocVector(REALSXP, 1));
  double *xp = REAL(X), *yp = REAL(Y), *zp = REAL(Z);
  
  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  cl_mem mem_z = create_buffer(env, sizeof(double));
  cl_mem scratch = create_buffer(env, sizeof(double) * nx);
  // TODO : scratch buffer size option
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDdot(nx, mem_z, 0, mem_x, 0, 1, mem_y, 0, 1, scratch,
                          env->num_queues, env->queues, 0, NULL, NULL);
  CHECK(err);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_z, zp, sizeof(double));
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  clReleaseMemObject(scratch);
  return Z;
}

SEXP Drotg(SEXP env_exp, SEXP A, SEXP B, SEXP C, SEXP S)
{
  cl_env *env = get_env(env_exp);
  if (!IS_SCALAR(A, REALSXP) || !IS_SCALAR(B, REALSXP) || 
      !IS_SCALAR(C, REALSXP) || !IS_SCALAR(S, REALSXP))
    Rf_error("arguments must be double scalar\n");
  double *ap = REAL(A), *bp = REAL(B), *cp = REAL(C), *sp = REAL(S);

  CHECK(clblasSetup());
  cl_mem mem_a = create_buffer(env, sizeof(double));
  cl_mem mem_b = create_buffer(env, sizeof(double));
  cl_mem mem_c = create_buffer(env, sizeof(double));
  cl_mem mem_s = create_buffer(env, sizeof(double));
  write_buffer(env, mem_a, ap, sizeof(double));
  write_buffer(env, mem_b, bp, sizeof(double));
  write_buffer(env, mem_c, cp, sizeof(double));
  write_buffer(env, mem_s, sp, sizeof(double));
  CHECK(clFinish(env->queues[0]));

  cl_int err = clblasDrotg(mem_a, 0, mem_b, 0, mem_c, 0, mem_s, 0, 
                           env->num_queues, env->queues, 0, NULL, NULL);
  CHECK(err);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_a, ap, sizeof(double));
  read_buffer(env, mem_b, bp, sizeof(double));
  read_buffer(env, mem_c, cp, sizeof(double));
  read_buffer(env, mem_s, sp, sizeof(double));
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return R_NilValue; 
}

SEXP Drot(SEXP env_exp, SEXP X, SEXP Y, SEXP C, SEXP S)
{ 
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  int nx = LENGTH(X), ny = LENGTH(Y);
  if (nx != ny || nx == 0)
    Rf_error("length of x and y must be the same\n");
  if (!IS_SCALAR(C, REALSXP) || !IS_SCALAR(S, REALSXP))
    Rf_error("arguments must be double scalar\n");
  double *xp = REAL(X), *yp = REAL(Y), c = REAL(C)[0], s = REAL(S)[0];

  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));

  cl_int err = clblasDrot(nx, mem_x, 0, 1, mem_y, 0, 1, c, s,
                          env->num_queues, env->queues, 0, NULL, NULL);
  CHECK(err);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double) * nx);
  read_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return R_NilValue; 
}

SEXP Drotmg(SEXP env_exp, SEXP D1, SEXP D2, SEXP X, SEXP Y, SEXP PARAM)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(PARAM, REALSXP);
  int np = LENGTH(PARAM);
  if (np < 5)
    Rf_error("length of param must be greater than 4\n");
  if (!IS_SCALAR(X, REALSXP) || !IS_SCALAR(Y, REALSXP) ||
      !IS_SCALAR(D1, REALSXP) || !IS_SCALAR(D2, REALSXP))
    Rf_error("arguments must be double scalar\n");
  double *xp = REAL(X), *yp = REAL(Y), *dp1 = REAL(D1), *dp2 = REAL(D2), *pp = REAL(PARAM);

  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double));
  cl_mem mem_y = create_buffer(env, sizeof(double));
  cl_mem mem_d1 = create_buffer(env, sizeof(double));
  cl_mem mem_d2 = create_buffer(env, sizeof(double));
  cl_mem mem_p = create_buffer(env, sizeof(double) * np);
  write_buffer(env, mem_x, xp, sizeof(double));
  write_buffer(env, mem_y, yp, sizeof(double));
  write_buffer(env, mem_d1, dp1, sizeof(double));
  write_buffer(env, mem_d2, dp2, sizeof(double));
  write_buffer(env, mem_p, pp, sizeof(double) * np);
  CHECK(clFinish(env->queues[0]));

  cl_int err = clblasDrotmg(mem_d1, 0, mem_d2, 0, mem_x, 0, mem_y, 0, mem_p, 0,
                            env->num_queues, env->queues, 0, NULL, NULL);
  CHECK(err);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double));
  read_buffer(env, mem_d1, dp1, sizeof(double));
  read_buffer(env, mem_d2, dp2, sizeof(double));
  read_buffer(env, mem_p, pp, sizeof(double) * np);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return R_NilValue; 
} 

SEXP Drotm(SEXP env_exp, SEXP X, SEXP Y, SEXP PARAM)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  TYPECHECK(Y, REALSXP);
  TYPECHECK(PARAM, REALSXP);
  int nx = LENGTH(X), ny = LENGTH(Y), np = LENGTH(PARAM);
  if (nx != ny || nx == 0)
    Rf_error("length of x and y must be the same\n");
  if (np < 5)
    Rf_error("length of param must be greater than 4\n");
  double *xp = REAL(X), *yp = REAL(Y), *pp = REAL(PARAM);
  
  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_y = create_buffer(env, sizeof(double) * ny);
  cl_mem mem_p = create_buffer(env, sizeof(double) * np);
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  write_buffer(env, mem_y, yp, sizeof(double) * ny);
  write_buffer(env, mem_p, pp, sizeof(double) * np);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDrotm(nx, mem_x, 0, 1, mem_y, 0, 1, mem_p, 0,
                          env->num_queues, env->queues, 0, NULL, NULL);
  CHECK(err);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_x, xp, sizeof(double) * nx);
  read_buffer(env, mem_y, yp, sizeof(double) * ny);
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  return R_NilValue;
}

SEXP Dnrm2(SEXP env_exp, SEXP X)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  int nx = LENGTH(X);
  SEXP NRM2 = PROTECT(allocVector(REALSXP, 1));
  double *xp = REAL(X), *np = REAL(NRM2);
  
  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_n = create_buffer(env, sizeof(double));
  cl_mem scratch = create_buffer(env, sizeof(double) * nx * 2);
  // TODO : scratch buffer size option
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDnrm2(nx, mem_n, 0, mem_x, 0, 1, scratch,
                           env->num_queues, env->queues, 0, NULL, NULL);
  CHECK(err);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_n, np, sizeof(double));
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  clReleaseMemObject(scratch);
  return NRM2;
}

SEXP iDamax(SEXP env_exp, SEXP X)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  int nx = LENGTH(X);
  SEXP iMax = PROTECT(allocVector(INTSXP, 1));
  double *xp = REAL(X); int *ip = INTEGER(iMax);
  cl_uint idx;
  
  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_i = create_buffer(env, sizeof(cl_uint));
  cl_mem scratch = create_buffer(env, sizeof(double) * nx * 2);
  // TODO : scratch buffer size option
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasiDamax(nx, mem_i, 0, mem_x, 0, 1, scratch,
                           env->num_queues, env->queues, 0, NULL, NULL);
  CHECK(err);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_i, &idx, sizeof(cl_uint));
  CHECK(clFinish(env->queues[0]));
  *ip = (int)idx;
  clblasTeardown();
  clReleaseMemObject(scratch);
  return iMax;
}

SEXP Dasum(SEXP env_exp, SEXP X)
{
  cl_env *env = get_env(env_exp);
  TYPECHECK(X, REALSXP);
  int nx = LENGTH(X);
  SEXP ASUM = PROTECT(allocVector(REALSXP, 1));
  double *xp = REAL(X), *sp = REAL(ASUM);
  
  CHECK(clblasSetup());
  cl_mem mem_x = create_buffer(env, sizeof(double) * nx);
  cl_mem mem_s = create_buffer(env, sizeof(double));
  cl_mem scratch = create_buffer(env, sizeof(double) * nx);
  // TODO : scratch buffer size option
  write_buffer(env, mem_x, xp, sizeof(double) * nx);
  CHECK(clFinish(env->queues[0]));
  
  cl_int err = clblasDasum(nx, mem_s, 0, mem_x, 0, 1, scratch,
                           env->num_queues, env->queues, 0, NULL, NULL);
  CHECK(err);
  for (int i=0; i < env->num_queues; i++)
    CHECK(clFinish(env->queues[i]));
  
  read_buffer(env, mem_s, sp, sizeof(double));
  CHECK(clFinish(env->queues[0]));
  clblasTeardown();
  clReleaseMemObject(scratch);
  return ASUM;
}
