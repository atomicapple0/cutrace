#ifndef _ERR_H_
#define _ERR_H_

#include <assert.h>
#include <cuda.h>
#include <stdio.h>

#define PANIC(fmt, ...)                                                  \
	do {                                                                 \
		printf("\n\n");                                                  \
		printf("/------------------------------------------------\\\n"); \
		printf("PANIC @ %s:%d %s\n", __FILE__, __LINE__,                 \
		       __FUNCTION__);                                            \
		printf(fmt "\n", ##__VA_ARGS__);                                 \
		printf("\\------------------------------------------------/\n"); \
		exit(1);                                                         \
	} while (0)

// panics on cuda failure
#define CUDA_OK(ans)                                          \
	do {                                                      \
		CUresult err = (CUresult) (ans);                      \
		const char *buf;                                      \
		cuGetErrorString(err, &buf);                          \
		if (err != CUDA_SUCCESS) {                            \
			PANIC("Cuda assertion failed!\nerror code = %d, " \
			      "\"%s\"",                                   \
			      err, buf);                                  \
		}                                                     \
	} while (0)

#endif // defined(_ERR_H_)
