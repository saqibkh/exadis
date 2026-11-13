#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void k(const double* a, const double* b, double* o) {
  int i = threadIdx.x;
  o[i] = fmin(fmax(a[i] / b[i], 0.0), 1.0);           // div result clamped
  o[8 + i] = fmin(fmax(a[i] * b[i] + a[i], 0.0), 1.0); // fma result clamped
  o[16 + i] = fmin(fmax(a[i] + b[i], 0.0), 1.0);       // add result clamped
}
int main() {
  double ha[8] = {-3, -0.5, 0.25, 0.5, 2, 7, -1e-3, 1.5};
  double hb[8] = {1, 2, 1, 1, 1, 2, 1, 1};
  double *a, *b, *o, ho[24];
  (void)hipMalloc(&a, 64); (void)hipMalloc(&b, 64); (void)hipMalloc(&o, 192);
  (void)hipMemcpy(a, ha, 64, hipMemcpyHostToDevice); (void)hipMemcpy(b, hb, 64, hipMemcpyHostToDevice);
  k<<<1, 8>>>(a, b, o);
  (void)hipMemcpy(ho, o, 192, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < 8; i++) {
    double e[3] = {fmin(fmax(ha[i] / hb[i], 0.0), 1.0), fmin(fmax(ha[i] * hb[i] + ha[i], 0.0), 1.0), fmin(fmax(ha[i] + hb[i], 0.0), 1.0)};
    for (int t = 0; t < 3; t++) if (e[t] != ho[8 * t + i]) { bad++; printf("case %d i=%d expected %g got %g\n", t, i, e[t], ho[8 * t + i]); }
  }
  printf("%d mismatches\n", bad);
  return bad != 0;
}
