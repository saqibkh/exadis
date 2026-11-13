#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
struct Vec3 { double x, y, z;
  __host__ __device__ Vec3() : x(0), y(0), z(0) {}
  __host__ __device__ Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
  __host__ __device__ double norm2() const { return x*x + y*y + z*z; } };
__host__ __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline Vec3 operator*(double s, const Vec3& a) { return Vec3(s*a.x, s*a.y, s*a.z); }
__host__ __device__ inline double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ __attribute__((noinline))
double get_min_dist2(const Vec3 &r1, const Vec3 &r2, const Vec3 &r3, const Vec3 &r4)
{
    double  eps = 1.0e-12;

    Vec3 r1mr3 = r1 - r3;
    Vec3 r2mr1 = r2 - r1;
    Vec3 r4mr3 = r4 - r3;

    Vec3 seg1L = r2mr1;
    Vec3 seg2L = r4mr3;

    double M[2][2];
    M[0][0] =  dot(r2mr1, r2mr1);
    M[1][0] = -dot(r4mr3, r2mr1);
    M[1][1] =  dot(r4mr3, r4mr3);
    M[0][1] =  M[1][0];

    double rhs[2];
    rhs[0] = -dot(r2mr1, r1mr3);
    rhs[1] =  dot(r4mr3, r1mr3);

    double detM = 1.0 - M[1][0] * M[1][0] / M[0][0] / M[1][1];

    double A = M[0][0];
    double B = -2.0 * rhs[0];
    //double C = -2.0 * M[1][0];
    double D = -2.0 * rhs[1];
    double E = M[1][1];

    int didDist2 = 0;
    double L1, L2, dist2;

    if (A < eps) {
        // If segment 1 is just a point...
        L1 = 0.0;
        if (E < eps) L2 = 0.0;
        else L2 = -0.5 * D / E;
        
    } else if (E < eps) {
        // If segment 2 is just a point...
        L2 = 0.0;
        if (A < eps) L1 = 0.0;
        else L1 = -0.5 * B / A;

    } else if (detM < 1e-6) {
        // If segments are parallel
        Vec3 r4mr1 = r4 - r1;
        Vec3 r3mr2 = r3 - r2;
        Vec3 r4mr2 = r4 - r2;

        double dist[4];
        dist[0] = dot(r1mr3, r1mr3);
        dist[1] = dot(r4mr1, r4mr1);
        dist[2] = dot(r3mr2, r3mr2);
        dist[3] = dot(r4mr2, r4mr2);

        dist2 = dist[0];
        int pos = 1;
        for (int i = 1; i < 4; i++) {
            if (dist[i] < dist2) {
                dist2 = dist[i];
                pos = i+1;
            }
        }

        L1 = floor((double)pos/2.1);
        L2 = (double)(1 - (pos % 2));
        didDist2 = 1;
    
    } else {
        // Solve the general case
        detM *= M[0][0]*M[1][1];

        double sol[2];
        sol[0] = ( M[1][1]*rhs[0] - M[0][1]*rhs[1]) / detM;
        sol[1] = (-M[1][0]*rhs[0] + M[0][0]*rhs[1]) / detM;

        if ((sol[0]>=0) && (sol[0]<=1) && (sol[1]>=0) && (sol[1]<=1)) {
            // we are done here
            L1 = sol[0];
            L2 = sol[1];

        } else {
            // enumerate four cases
            int icase;
            double trial[4][2];

            // alpha = 0
            icase = 0;
            trial[icase][0] = 0;
            trial[icase][1] = (rhs[1] - M[1][0]*trial[icase][0]) / M[1][1];

            // alpha = 1
            icase = 1;
            trial[icase][0] = 1;
            trial[icase][1] = (rhs[1] - M[1][0]*trial[icase][0]) / M[1][1];

            // beta = 0
            icase = 2;
            trial[icase][1] = 0;
            trial[icase][0] = (rhs[0] - M[0][1]*trial[icase][1]) / M[0][0];

            // beta = 1
            icase = 3;
            trial[icase][1] = 1;
            trial[icase][0] = (rhs[0] - M[0][1]*trial[icase][1]) / M[0][0];

            // find the minimum out of four trials
            double d2min = 1e100;
            for (icase = 0; icase < 4; icase++) {
                trial[icase][0] = fmin(fmax(trial[icase][0], 0.0), 1.0);
                trial[icase][1] = fmin(fmax(trial[icase][1], 0.0), 1.0);  
                Vec3 dist = r1 + trial[icase][0]*seg1L - r3 - trial[icase][1]*seg2L;
                double d2 = dist.norm2();
                if (d2 < d2min) {
                    L1 = trial[icase][0];
                    L2 = trial[icase][1];
                    d2min = d2;
                }
            }
            dist2 = d2min;
            didDist2 = 1;
        }
    }

    // Make sure L1 and L2 are between 0 and 1
    L1 = fmin(fmax(L1, 0.0), 1.0);
    L2 = fmin(fmax(L2, 0.0), 1.0);

    if (!didDist2) {
        Vec3 dist = r1 + L1*seg1L - r3 - L2*seg2L;
        dist2 = dist.norm2();
    }

    return dist2;
}
__global__ void k(const double* in, double* out, int n) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  const double* q = in + 13*p;
  out[p] = get_min_dist2(Vec3(q[0],q[1],q[2]), Vec3(q[3],q[4],q[5]), Vec3(q[6],q[7],q[8]), Vec3(q[9],q[10],q[11]));
}
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "r"); std::vector<double> v; double x;
  while (fscanf(f, "%lf", &x) == 1) v.push_back(x);
  int n = v.size() / 13; double *din, *dout; std::vector<double> out(n);
  hipMalloc(&din, v.size()*8); hipMalloc(&dout, n*8);
  hipMemcpy(din, v.data(), v.size()*8, hipMemcpyHostToDevice);
  k<<<(n+63)/64, 64>>>(din, dout, n);
  hipMemcpy(out.data(), dout, n*8, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int p = 0; p < n; p++) {
    const double* q = &v[13*p];
    double h = get_min_dist2(Vec3(q[0],q[1],q[2]), Vec3(q[3],q[4],q[5]), Vec3(q[6],q[7],q[8]), Vec3(q[9],q[10],q[11]));
    if (!(fabs(h - out[p]) <= 1e-9 * fmax(1.0, fabs(h)))) { if (bad < 8) printf("p=%d host=%.17g dev=%.17g\n", p, h, out[p]); bad++; }
  }
  printf("%d of %d mismatch\n", bad, n);
  return bad != 0;
}
