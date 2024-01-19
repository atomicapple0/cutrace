CUDA_PATH = /usr/local/cuda

CFLAGS = -g -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib
LDLIBS = -lcuda -lcudart

all: alphabet

alphabet: alphabet.c mod.ptx cuda_error.o
	$(CC) $(CFLAGS) -o alphabet alphabet.c cuda_error.o $(LDLIBS)

cuda_error.o: cuda_error.c
	$(CC) $(CFLAGS) -c -o cuda_error.o cuda_error.c \
	    $(LDLIBS)

mod.ptx: mod.cu
	nvcc -ptx mod.cu