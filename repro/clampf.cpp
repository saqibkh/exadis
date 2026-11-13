#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void k(const float* a, const float* b, float* o) {
  int i = threadIdx.x;
  o[i] = fminf(fmaxf(a[i] / b[i], 0.0f), 1.0f);
}
int main() {
  float ha[8] = {-3, -0.5f, 0.25f, 0.5f, 2, 7, -1e-3f, 1.5f};
  float hb[8] = {1, 2, 1, 1, 1, 2, 1, 1};
  float *a, *b, *o, ho[8];
  (void)hipMalloc(&a, 32); (void)hipMalloc(&b, 32); (void)hipMalloc(&o, 32);
  (void)hipMemcpy(a, ha, 32, hipMemcpyHostToDevice); (void)hipMemcpy(b, hb, 32, hipMemcpyHostToDevice);
  k<<<1, 8>>>(a, b, o);
  (void)hipMemcpy(ho, o, 32, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < 8; i++) { float e = fminf(fmaxf(ha[i] / hb[i], 0.0f), 1.0f); if (e != ho[i]) { bad++; printf("i=%d expected %g got %g\n", i, e, ho[i]); } }
  printf("%d mismatches\n", bad);
  return bad != 0;
}
