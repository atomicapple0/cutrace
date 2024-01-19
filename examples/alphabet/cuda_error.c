#define _GNU_SOURCE

#include <cuda.h>

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>

void cuda_error(int status, CUresult errnum, const char *format, ...)
{
	va_list ap;
	const char *errstr;

	fflush(stdout);
	fputs(program_invocation_name, stderr);
	if (format != NULL) {
		fputs(": ", stderr);
		va_start(ap, format);
		vfprintf(stderr, format, ap);
		va_end(ap);
	}
	if (errnum != CUDA_SUCCESS) {
		(void)cuGetErrorString(errnum, &errstr);
		fprintf(stderr, ": %s", errstr);
	}
	fputc('\n', stderr);
	if (status != 0)
		exit(status);
}