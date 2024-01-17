all: saxpy

saxpy: saxpy.cu
	nvcc -o saxpy saxpy.cu -lcuda --cudart shared

clean:
	rm -f saxpy
