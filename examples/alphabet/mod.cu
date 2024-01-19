#include <cstdio>

extern "C" {
    __global__ void alphabet(int a, float *b, char c, char d, char e, float* f, float g, char h, int i, char j, int* k, char l) {
        printf("\ra = 0x%lx\n", (size_t) a);
        printf("\rb = 0x%lx\n", (size_t) b);
        printf("\rc = 0x%lx\n", (size_t) c);
        printf("\rd = 0x%lx\n", (size_t) d);
        printf("\re = 0x%lx\n", (size_t) e);
        printf("\rf = 0x%lx\n", (size_t) f);
        printf("\rg = 0x%lx\n", (size_t) g);
        printf("\rh = 0x%lx\n", (size_t) h);
        printf("\ri = 0x%lx\n", (size_t) i);
        printf("\rj = 0x%lx\n", (size_t) j);
        printf("\rk = 0x%lx\n", (size_t) k);
        printf("\rl = 0x%lx\n", (size_t) l);
    }
}
