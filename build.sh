#!/bin/sh
rm *.so
gcc -O2 -shared -fPIC -Wall -Wextra -lm -Wno-unused-parameter -o libkernel.so kernel.c
