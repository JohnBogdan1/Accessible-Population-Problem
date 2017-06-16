CC=g++
CFLAGS=-g -Wall -std=c++11 -D FEP_OPENCL
LIB=-L/usr/local/cuda-8.0/lib64/

.PHONY: all clean

all: build

build: accpop

accpop: host_skl.cpp helper.cpp
	$(CC) $(CFLAGS) $(INC) $(LIB) host_skl.cpp helper.cpp -lOpenCL -o accpop

clean:
	rm -f accpop
