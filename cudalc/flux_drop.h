// Uniform limb-darkening
__host__ __device__ double Flux_drop_analytical_uniform( double z, double k, double SBR, double f);


// power-2
__device__ __host__ double  Flux_drop_analytical_power_2(double z, double k, double c, double a, double f, double eps);

// Quadratic
// d = z and p = k
__device__ __host__ double  Flux_drop_analytical_quadratic(double d, double p, double c1, double c2, double tol);
