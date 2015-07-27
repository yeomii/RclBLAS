env <- function(dev_type="g", dev_num=4) .Call("create_env", dev_type, dev_num)
dgemm <- function(env, a, b) .Call("dgemm", env, a, b)
