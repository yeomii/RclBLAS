library(RclBLAS)

a <- matrix(1:6, 3, 2)
b <- matrix(1:6, 2, 3)

x <- env("g", 2)

c <- dgemm(x, a, b)
c
a %*% b
