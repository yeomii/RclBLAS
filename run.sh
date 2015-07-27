#!/bin/sh

if [ $# -eq 1 ]
then
  thorq --add --device gpu ~/R/R-3.2.1/bin/R "<" $1 --no-save
else
  thorq --add --device gpu ~/R/R-3.2.1/bin/R "< " "test.R" --no-save
fi
