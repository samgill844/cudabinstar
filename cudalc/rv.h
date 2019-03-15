__host__ __device__ void rv (const double * time, 
        double t_zero, double period, 
        double K1, double fs, double fc, 
        double V0, double dV0, double VS, double VC, 
        double radius_1, double k, double incl, int ld_law_1, double ldc_1_1, double ldc_1_2,
        int Accurate_t_ecl, double t_ecl_tolerance,  int Accurate_Eccentric_Anomaly, double E_tol,
        double * RV_,  int N_start, int RV_N);


__host__ __device__ double rv_loglike (const double * time, double * RV, double *RV_ERR, double J,
    double t_zero, double period, 
    double K1, double fs, double fc, 
    double V0, double dV0, double VS, double VC, 
    double radius_1, double k, double incl, int ld_law_1, double ldc_1_1, double ldc_1_2,
    int Accurate_t_ecl, double t_ecl_tolerance,  int Accurate_Eccentric_Anomaly, double E_tol,
    int RV_N);