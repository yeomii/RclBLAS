create_env <- function(dev_type="g", dev_num=1) 
  .Call("create_env", dev_type, dev_num)
type <- function(x) .Call("get_type", x)

swap <- function(env, x, y, alpha) 
  .External("RclBLAS", routine="swap", env=env, x=x, y=y, alpha=alpha)

dswap <- function(env, x, y, n=-1, offx=0, incx=1, offy=0, incy=1) 
  .Call("Dswap", 
    env, 
    as.double(x), 
    as.double(y), 
    as.integer(n), 
    as.integer(offx), 
    as.integer(incx), 
    as.integer(offy), 
    as.integer(incy))

dscal <- function(env, x, alpha, n=-1, offx=0, incx=1) 
  .Call("Dscal", 
    env, 
    as.double(x), 
    as.double(alpha), 
    as.integer(n), 
    as.integer(offx),  
    as.integer(incx))

dcopy <- function(env, x, y=NULL, n=-1, offx=0, incx=1, offy=0, incy=1) 
  .Call("Dcopy", 
    env, 
    as.double(x), 
    as.double(y), 
    as.integer(n), 
    as.integer(offx), 
    as.integer(incx), 
    as.integer(offy), 
    as.integer(incy))

daxpy <- function(env, x, y, alpha, n=-1, offx=0, incx=1, offy=0, incy=1) 
  .Call("Daxpy", 
    env, 
    as.double(x), 
    as.double(y),
    as.double(alpha), 
    as.integer(n), 
    as.integer(offx), 
    as.integer(incx), 
    as.integer(offy), 
    as.integer(incy))

ddot <- function(env, x, y, n=-1, offx=0, incx=1, offy=0, incy=1, scratch=-1) 
  .Call("Ddot", 
    env, 
    as.double(x), 
    as.double(y),
    as.integer(n), 
    as.integer(offx), 
    as.integer(incx), 
    as.integer(offy), 
    as.integer(incy),
    as.integer(scratch))

drotg <- function(env, a, b, c, s) 
  .Call("Drotg", 
    env, 
    as.double(a), 
    as.double(b),
    as.double(c), 
    as.double(s))

drot <- function(env, x, y, c, s, n=-1, offx=0, incx=1, offy=0, incy=1) 
  .Call("Drot", 
    env, 
    as.double(x), 
    as.double(y),
    as.double(c),
    as.double(s),
    as.integer(n), 
    as.integer(offx), 
    as.integer(incx), 
    as.integer(offy), 
    as.integer(incy))

drotmg <- function(env, d1, d2, x, y, param) 
  .Call("Drotmg", 
    env, 
    as.double(d1), 
    as.double(d2),
    as.double(x), 
    as.double(y),
    as.double(param))

drotm <- function(env, x, y, param, n=-1, offx=0, incx=1, offy=0, incy=1, offparam=0) 
  .Call("Drotm", 
    env, 
    as.double(x), 
    as.double(y),
    as.double(param),
    as.integer(n), 
    as.integer(offx), 
    as.integer(incx), 
    as.integer(offy), 
    as.integer(incy),
    as.integer(offparam))
  
dnrm2 <- function(env, x, n=-1, offx=0, incx=1, scratch=-1) 
  .Call("Dnrm2", 
    env, 
    as.double(x), 
    as.integer(n), 
    as.integer(offx),  
    as.integer(incx),
    as.integer(scratch))

idamax <- function(env, x, n=-1, offx=0, incx=1, scratch=-1) 
  .Call("iDamax", 
    env, 
    as.double(x), 
    as.integer(n), 
    as.integer(offx),  
    as.integer(incx),
    as.integer(scratch))

dasum <- function(env, x, n=-1, offx=0, incx=1, scratch=-1) 
  .Call("Dasum",
    env, 
    as.double(x), 
    as.integer(n), 
    as.integer(offx),  
    as.integer(incx),
    as.integer(scratch))

dgemv <- function(env, a, x, y=NULL, alpha=1, beta=0, transA='n') 
  .Call("Dgemv", 
    env, 
    a, 
    as.double(x), 
    as.double(y), 
    as.double(alpha), 
    as.double(beta), 
    transA)

dsymv <- function(env, a, x, y=NULL, alpha=1, beta=0, uplo='u') 
  .Call("Dsymv", 
    env, 
    a, 
    as.double(x), 
    as.double(y), 
    as.double(alpha), 
    as.double(beta), 
    uplo)

dtrmv <- function(env, a, x, transA='n', uplo='u', diag='n', scratch=-1) 
  .Call("Dtrmv", 
    env, 
    a, 
    as.double(x), 
    transA,
    uplo, 
    diag,
    as.integer(scratch))

dtrsv <- function(env, a, x, transA='n', uplo='u', diag='n') 
  .Call("Dtrsv", 
    env, 
    a, 
    as.double(x),
    transA, 
    uplo, 
    diag)

dger <- function(env, a, x, y, alpha=1) 
  .Call("Dger", 
    env, 
    a, 
    as.double(x), 
    as.double(y), 
    as.double(alpha))

dsyr <- function(env, a, x, alpha=1, uplo='u') 
  .Call("Dsyr", 
    env, 
    a, 
    as.double(x), 
    as.double(alpha), 
    uplo)

dsyr2 <- function(env, a, x, y, alpha=1, uplo='u') 
  .Call("Dsyr2", 
    env, 
    a, 
    as.double(x), 
    as.double(y), 
    as.double(alpha), 
    uplo)

dgbmv <- function(env, a, x, y, m, n, kl, ku, alpha=1, beta=1, transA='n') 
  .Call("Dgbmv", 
    env, 
    a, 
    as.double(x), 
    as.double(y), 
    as.integer(m),
    as.integer(n),
    as.integer(kl), 
    as.integer(ku),
    as.double(alpha), 
    as.double(beta), 
    transA)

dtbmv <- function(env, a, x, n, k, transA='n', uplo='u', diag='n', scratch=-1) 
  .Call("Dtbmv", 
    env, 
    a, 
    as.double(x),
    as.integer(n),
    as.integer(k),
    transA, 
    uplo, 
    diag, 
    as.integer(scratch))
  
dsbmv <- function(env, a, x, y, n, k, alpha=1, beta=1, uplo='u') 
  .Call("Dsbmv", 
    env, 
    a, 
    as.double(x), 
    as.double(y), 
    as.integer(n),
    as.integer(k),
    as.double(alpha), 
    as.double(beta), 
    uplo)

dtbsv <- function(env, a, x, n, k, transA='n', uplo='u', diag='n') 
  .Call("Dtbsv", 
    env, 
    a, 
    as.double(x),
    as.integer(n),
    as.integer(k),
    transA, 
    uplo, 
    diag)

dgemm <- function(env, a, b, c=NULL, alpha=1, beta=0, transA='n', transB='n') 
  .Call("Dgemm", 
    env, 
    a, 
    b, 
    c, 
    as.double(alpha), 
    as.double(beta),
    transA,
    transB)
dtrmm <- function(env, a, b, alpha=1, side='l', transA='n', uplo='u', diag='n') 
  .Call("Dtrmm", 
    env, 
    a, 
    b, 
    as.double(alpha), 
    side,
    transA,
    uplo, 
    diag)
  
dtrsm <- function(env, a, b, alpha=1, side='l', transA='n', uplo='u', diag='n') 
  .Call("Dtrsm", 
    env, 
    a, 
    b, 
    as.double(alpha), 
    side,
    transA,
    uplo, 
    diag)

dsyrk <- function(env, a, c=NULL, alpha=1, beta=0, transA='n', uplo='u') 
  .Call("Dsyrk", 
    env, 
    a, 
    c, 
    as.double(alpha), 
    as.double(beta), 
    transA,
    uplo)

dsyr2k <- function(env, a, b, c=NULL, alpha=1, beta=0, transAB='n', uplo='u') 
  .Call("Dsyr2k", 
    env, 
    a, 
    b, 
    c, 
    as.double(alpha), 
    as.double(beta), 
    transAB,
    uplo)

dsymm <- function(env, a, b, c=NULL, alpha=1, beta=0, side='l', uplo='u') 
  .Call("Dsymm", 
    env, 
    a, 
    b, 
    c, 
    as.double(alpha), 
    as.double(beta), 
    side,
    uplo) 
