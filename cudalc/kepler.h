/************************************
*        Keplers equation           *
************************************/
__device__ __host__ double kepler (double M, double E, double e);
__device__ __host__ double dkepler (double E, double e);

/************************************
*        Eccentric Anomaly          *
************************************/
__device__ __host__ double getEccentricAnomaly (double M, double e, int Accurate_Eccentric_Anomaly, double tol);
__device__ __host__ double t_ecl_to_peri(double t_ecl, double e, double w, double incl, double radius_1, double p_sid, double t_ecl_tolerance, int Accurate_t_ecl);
__device__ __host__ double getTrueAnomaly(double time, double  e, double w, double period, double t_zero, double incl, double radius_1, double t_ecl_tolerance, int Accurate_t_ecl,  int Accurate_Eccentric_Anomaly, double E_tol );

/***************************************************
*        Calculate the projected seperaton         *
***************************************************/
__device__ __host__ double get_z(double nu, double e, double incl, double w, double radius_1) ;
__device__ __host__ double get_z_(double nu, double * z);

/***************************************************
*        Calculate the projected position          *
***************************************************/
__device__ __host__ double getProjectedPosition(double nu, double w, double incl);

/*********************************************
*        Calculate the mass function         *
*********************************************/
__device__ __host__ double mass_function_1(double e, double P, double K1);
__device__ __host__ double mass_function_1_(double M2, double * z);