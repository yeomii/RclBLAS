#!/bin/sh

#R CMD REMOVE RclBLAS
R CMD build RclBLAS/
R CMD INSTALL RclBLAS_1.0.tar.gz
