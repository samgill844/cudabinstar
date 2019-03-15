__host__ __device__ void lc(const double * time, double * LC,
                        double t_zero, double period,
                        double radius_1, double k ,
                        double fs, double fc, 
                        double incl,
                        int ldc_law_1, double ldc_1_1, double ldc_1_2, 
                        double SBR, double light_3,
                        int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
                        int N_LC );


__host__ __device__ double lc_loglike(const double * time, const double * LC, const double * LC_ERR, double zp, double J,
    double t_zero, double period,
    double radius_1, double k ,
    double fs, double fc, 
    double incl,
    int ld_law_1, double ldc_1_1, double ldc_1_2, 
    double SBR, double light_3,
    int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
    int N_LC );