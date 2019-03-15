#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <unistd.h>
#include <getopt.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../include/sampler_helper.h"
#include "../include/ensemble.h"
#include "../include/lc.h"

/* Flag set by ‘--verbose’. */
static int verbose_flag;

#define MAXCHAR 1000


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\nGPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int count_3_col(char * filename, double t_zero, double period, double pcut)
{

    // count the number of lines in the file called filename                                    
    FILE *fp = fopen(filename,"r");
    if (fp == NULL)
    {
        if (!fp)perror("fopen");
        printf("\nI've failed :(");
    }  

    char str[MAXCHAR];

    double a,b,c,d,e, phase;
    int count=0;
    while (fgets(str, MAXCHAR, fp) != NULL)
    {
        sscanf(str, "%lf %lf %lf %lf %lf", &a, &b, &c, &d, &e);
        phase = ((a - t_zero)/period) - floor((a - t_zero)/period);
        if (phase < pcut || phase > (1-pcut)) count ++;
    }
        //printf("%s", str);
    fclose(fp);
    return count;
}

void readlread_3_col(char * filename, int lines_to_read, double * x, double * y, double * z, double t_zero, double period, double pcut)
{

    // count the number of lines in the file called filename                                    
    FILE *fp = fopen(filename,"r");
    if (fp == NULL)
    {
        if (!fp)perror("fopen");
        printf("\nI've failed :(");
    }  

    char str[MAXCHAR];

    double a,b,c,d,e, phase;
    int count=0;
    while (fgets(str, MAXCHAR, fp) != NULL)
    {
        sscanf(str, "%lf %lf %lf %lf %lf", &a, &b, &c, &d, &e);
        phase = ((a - t_zero)/period) - floor((a - t_zero)/period);
        if (phase < pcut || phase > (1-pcut))
        {
            x[count]=a;
            y[count]=b;
            z[count]=c;
            count ++;
        }
    }
        //printf("%s", str);
    fclose(fp);
}



void historgram_plot_2_axis()
{
    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
    const char * commandsForGnuplot[] = {"set term qt size 1600, 700",
                                   "set multiplot layout 2,4 rowsfirst" ,
                                   "set key off", 
                                   "set datafile separator ','",
                                   "bin_width = 0.0005",
                                   "bin_number(x) = floor(x/bin_width)",
                                   "rounded(x) = bin_width * ( bin_number(x) + 0.5 )",

                                   "set title 'T0'",
                                   "plot 'output_gpu.dat' using (rounded($4)):(1) smooth frequency with boxes", 

                                   "bin_width = 0.0003",
                                   "bin_number(x) = floor(x/bin_width)",
                                   "rounded(x) = bin_width * ( bin_number(x) + 0.5 )",
                                   "set title 'Period'",
                                   "plot 'output_gpu.dat' using (rounded($5)):(1) smooth frequency with boxes",
                                
                                   "bin_width = 0.01",
                                   "bin_number(x) = floor(x/bin_width)",
                                   "rounded(x) = bin_width * ( bin_number(x) + 0.5 )",
                                   "set title 'radius_1'",
                                   "plot 'output_gpu.dat' using (rounded($6)):(1) smooth frequency with boxes",

                                   "set title 'k'",
                                   "plot 'output_gpu.dat' using (rounded($7)):(1) smooth frequency with boxes",
                                
                                   "set title 'h1'",
                                   "plot 'output_gpu.dat' using (rounded($8)):(1) smooth frequency with boxes",
                                
                                   "bin_width = 0.1",
                                   "bin_number(x) = floor(x/bin_width)",
                                   "rounded(x) = bin_width * ( bin_number(x) + 0.5 )",
                                   "set title 'h2'",
                                   "plot 'output_gpu.dat' using (rounded($9)):(1) smooth frequency with boxes",
                                
                                   "bin_width = 0.01",
                                   "bin_number(x) = floor(x/bin_width)",
                                   "rounded(x) = bin_width * ( bin_number(x) + 0.5 )",
                                   "set title 'zp'",
                                   "plot 'output_gpu.dat' using (rounded($10)):(1) smooth frequency with boxes",

};
    int i;
    for (i=0; i < 33; i++)
        fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.
}



int main(int argc, char* argv[])
{
    printf("\n------------------------------------");
    printf("\n-          NGTSfit V0.1            -");
    printf("\n-      samgill844@gmail.com        -");
    printf("\n------------------------------------");
    printf("\e[?25l"); // stop blinking cursor

    // Filename
    char *input_filename = "ngts.lc";
    char *output_filename = "output.dat";


    // Lightcurve parameters
    double t_zero = 0.0;
    double period = 1.0;
    double pcut = 0.1;
    double radius_1 = 0.2;
    double k = 0.2;
    double zp = 0.;
    double jitter = 0.001;
    double b = 0.1;

    // Limb-darkening parameters
    int ld_law = 0;
    double ldc_1 = 0.65;
    double ldc_2 = 0.37;

    // Fitting parameters
    int nsteps = 1000;
    int burn_in = 950;
    int nwalkers = 10240;
    int threads_per_block = 256;

    // GPU or CPU
    int CPU_OR_GPU = 0; // 0 = CPU, 1 = GPU
    int device = 0; // the GPU device

    // Need to re-jig this according to
    // https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html#Getopt-Long-Option-Example
    opterr = 0;

    while (1)
    {
        static struct option long_options[] =
        {
            /* These options set a flag. */
            {"verbose", no_argument,       &verbose_flag, 1},
            {"brief",   no_argument,       &verbose_flag, 0},
            {"gpu", no_argument,       &CPU_OR_GPU, 1},

            /* These options don’t set a flag.
                We distinguish them by their indices. */
            {"filename",     required_argument,       0, 'f'},
            {"output",       required_argument,       0, 'o'},
            {"t_zero",       required_argument,       0, 't'},
            {"period",       required_argument,       0, 'p'},
            {"pcut",         required_argument,       0, 'c'},
            {"radius_1",     required_argument,       0, 'r'},
            {"k",            required_argument,       0, 'k'},
            {"zp",            required_argument,       0, 'z'},
            {"jitter",            required_argument,  0, 'j'},
            {"impact",            required_argument,  0, 'u'},

            {"ld_1",         required_argument,       0, 'l'},
            {"ldc_1",        required_argument,       0, 'q'},
            {"ldc_2",        required_argument,       0, 'w'},

            {"nsteps",       required_argument,       0, 'n'},
            {"burn_in",      required_argument,       0, 'b'},
            {"walkers",      required_argument,       0, 'w'},
            {"threads_per_block", required_argument,       0, 'y'},

            {"device",       required_argument,       0, 'd'},
            {0, 0, 0, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        int c = getopt_long (argc, argv, "f:o:t:p:c:r:k:l:q:w:n:b:w:y:d:",
                        long_options, &option_index);
    
        /* Detect the end of the options. */
        if (c == -1) break;   


        switch (c)
        {
        case 0:
          /* If this option set a flag, do nothing else now. */
          if (long_options[option_index].flag != 0) break;
          printf ("option %s", long_options[option_index].name);
          if (optarg) printf (" with arg %s", optarg);
          printf ("\n");
          break;

        case 'f':
            input_filename = optarg;
            break;

        case 'o':
            output_filename = optarg;
            break;

        case 't':
            sscanf(optarg,"%lf",&t_zero);
            break;

        case 'p':
            sscanf(optarg,"%lf",&period);
            break;
     
        case 'c':
            sscanf(optarg,"%lf",&pcut);
            break;
    
        case 'r':
            sscanf(optarg,"%lf",&radius_1);
            break;

        case 'k':
            sscanf(optarg,"%lf",&k);
            break;

        case 'z':
            sscanf(optarg,"%lf",&zp);
            break;

        case 'j':
            sscanf(optarg,"%lf",&jitter);
            break;

        case 'u':
            sscanf(optarg,"%lf",&b);
            break;

        case 'l':
            sscanf(optarg,"%d",&ld_law);
            break;

        case 'q':
            sscanf(optarg,"%lf",&ldc_1);
            break;

        case 'w':
            sscanf(optarg,"%lf",&ldc_2);
            break;

        case 'n':
            sscanf(optarg,"%d",&nsteps);
            break;

        case 'b':
            sscanf(optarg,"%d",&burn_in);
            break;

        case 'e':
            sscanf(optarg,"%d",&nwalkers);
            break;

        case 'y':
            sscanf(optarg,"%d",&threads_per_block);
            break;

        case 'd':
            sscanf(optarg,"%d",&device);
            break;

        case '?':
          /* getopt_long already printed an error message. */
          break;

        default:
          abort ();
        }

    }
    
    printf("\nExample use:");
    printf("\nngtsfit [filename] [t_zero] [period] [pcut] [radius_1=0.2] [k=0.2] [h1=0.65] [h2=0.37]");
    printf("\n\t\t[nsteps=1000] [burn_in=950] [nwalkers=10240] [threads_per_block=256]");
    printf("\n\t\t[output_file=NGTSfit_results.dat] [gpu // cpu]\n\n");

    int blocks = (int) ceil(nwalkers/threads_per_block);
    

    /*---------------------------
     Part 0 - report choices
     ---------------------------*/
    printf("\nFitting parameters:");
    printf("\n\tnsteps : %d", nsteps); fflush(stdout);
    printf("\n\tburn in : %d", burn_in); fflush(stdout);
    printf("\n\tnwalkers : %d", nwalkers); fflush(stdout);
    printf("\n\tpcut : %f", pcut); fflush(stdout);

    printf("\n\tthreads_per_block : %d", threads_per_block); fflush(stdout);
    printf("\n\tblocks: %d", blocks); fflush(stdout);
    printf("\n\tinput file : %s", input_filename);
    printf("\n\toutput file: %s", output_filename); fflush(stdout);
    printf("\n\tdevice used: "); fflush(stdout);
    switch(CPU_OR_GPU){ case 1 : printf("GPU\n"); break;  case 0 : printf("CPU\n"); break;}

    printf("\n\tt_zero : %f", t_zero);
    printf("\n\tperiod : %f", period);
    printf("\n\tradius_1 : %f", radius_1);
    printf("\n\tk : %f", k);
    printf("\n\tzp : %f", zp);
    printf("\n\tjitter : %f", jitter);
    printf("\n\timpact : %f", b);

    printf("\n\tldc_1 : %f", ldc_1);
    printf("\n\tldc_2 : %f", ldc_2);


    /*---------------------------
     Part 1 - read the LC 
     ---------------------------*/
    printf("\n\nReading data from %s:", input_filename);
    double *time, *d_time, *LC, *d_LC, *LC_ERR, *d_LC_ERR, *d_N_LC;
    int N_LC = count_3_col(input_filename, t_zero, period, pcut);
    printf("\n\tNumber of lines : %d", N_LC);fflush(stdout);
    time = (double *) malloc(N_LC*sizeof(double));
    LC = (double *) malloc(N_LC*sizeof(double));
    LC_ERR = (double *) malloc(N_LC*sizeof(double));
    readlread_3_col(input_filename, N_LC, time, LC, LC_ERR,  t_zero, period, pcut);
    printf("\n\tRead in OK!");fflush(stdout);


    /*---------------------------
     Part 2 - initialise theta
     ---------------------------*/
    double * theta;
    int ndim=9;
    theta = (double *) malloc(ndim*sizeof(double));
    // 0 : t_zero
    // 1 : period
    // 2 : radius_1
    // 3 : k
    // 4 : h1
    // 5 : h2
    // 6 : b
    // 7 : zp
    // 8 : J

    theta[0] = t_zero;
    theta[1] = period;
    theta[2] = radius_1;
    theta[3] =  k;
    theta[4] = ldc_1;
    theta[5] =  ldc_2;
    theta[6] = b;
    theta[7] =  zp;
    theta[8] =  jitter;

    double loglik =     lc_loglike(time, LC, LC_ERR, theta[7],theta[8],
        theta[0], theta[1],
        theta[2], theta[3] ,
        0., 0., 
        90.,
        0, theta[4], theta[5], 
        0., 0.,
        0, 0.001, 0, 0.001,
        N_LC );
    printf("\n\n------------------------------\nInitial loglike : %f\n------------------------------\n", loglik);

    /*---------------------------
    Part 3 - configure the arguments for either GPU ot CPU
    ---------------------------*/
    printf("Configuring arguments for the "); fflush(stdout);
    double ** args, **d_args;
    switch(CPU_OR_GPU){ case 1 : printf("GPU... "); break;  case 0 : printf("CPU... "); break;}

    double * tmpp;
    tmpp = (double *) malloc(5*sizeof(double));
    tmpp[0] = (double) N_LC;
    tmpp[1] = t_zero;
    tmpp[2] = period;
    tmpp[3] = ldc_1;
    tmpp[4] = ldc_2;
    switch(CPU_OR_GPU){ case 1 :
                    // Now set up the args
                    cudaMalloc(&d_time, N_LC*sizeof(double)); 
                    cudaMalloc(&d_LC, N_LC*sizeof(double)); 
                    cudaMalloc(&d_LC_ERR, N_LC*sizeof(double)); 
                    cudaMalloc(&d_N_LC, 5*sizeof(double)); 

                    cudaMemcpy(d_time, time, N_LC*sizeof(double), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_LC, LC, N_LC*sizeof(double), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_LC_ERR, LC_ERR, N_LC*sizeof(double), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_N_LC, tmpp, 5*sizeof(double), cudaMemcpyHostToDevice);

                    args = (double **) malloc(4*sizeof(double *));
                    args[0] = d_time;
                    args[1] = d_LC;
                    args[2] = d_LC_ERR;
                    args[3] = d_N_LC;
                    cudaMalloc(&d_args, 4*sizeof(double**)); 
                    cudaMemcpy(d_args, args, 4*sizeof(double**), cudaMemcpyHostToDevice);
                    break;
                
                 case 0:
                    args = (double **) malloc(4*sizeof(double *));
                    args[0] = time;
                    args[1] = LC;
                    args[2] = LC_ERR;
                    args[3] = tmpp;
                    break;
                }
    printf("done."); fflush(stdout);



    /*---------------------------
    Part 4 - Create the starting positions
    ---------------------------*/
    printf("\nCreating the sarting positions."); fflush(stdout);
    double scatter = 0.0001;
    double * d_positions, * d_loglikliehoods;
    switch(CPU_OR_GPU)
    {
        case 1:
            // First malloc the device arrays
            gpuErrchk(cudaMalloc(&d_positions, nwalkers*nsteps*ndim*sizeof(double))); 
            gpuErrchk(cudaMalloc(&d_loglikliehoods, nwalkers*nsteps*sizeof(double))); 

            create_starting_positions(theta,
                nsteps, ndim, nwalkers,
                blocks, threads_per_block,
                scatter, 
                d_positions, d_loglikliehoods);
            printf(" done."); fflush(stdout);
            break;

        case 0:
            printf(" done."); fflush(stdout);
            break    ;     
    }

    /*---------------------------
    Part 5 - Create the curand states if requireds
    ---------------------------*/
    curandState *devState;
    if (CPU_OR_GPU==1)
    {
        // Create the curand states
        printf("\nCreating the curand states... "); fflush(stdout);
        gpuErrchk(cudaMalloc((void**)&devState, nwalkers*sizeof(curandState)));
        initCurand<<<ceil(nwalkers/256),256>>>(devState, 1);
        printf(" done."); fflush(stdout);
    }

    /*---------------------------
    Part 5 - Create the blocks to monitor progress
    ---------------------------*/
    int * d_block_progress;
    int i;
    if (CPU_OR_GPU==1)
    {
        int d_block_progress__[1] = {0};
        gpuErrchk(cudaMalloc(&d_block_progress, blocks*sizeof(int))); 
        for (i=0; i < blocks; i++) gpuErrchk(cudaMemcpy(&d_block_progress[i], &d_block_progress__, sizeof(int), cudaMemcpyHostToDevice));
    }


    /*---------------------------
    Part 5 - Launch the sampler
    ---------------------------*/
    clock_t start, end;
    if (CPU_OR_GPU==1)
    {
        cudaStream_t streams[2];
        cudaStreamCreate(&streams[0]);
        cudaStreamCreate(&streams[1]);


        // Now run
        printf("\n-----------------------------------");
        printf("\nCommencing Bayesian sampleing [GPU]\n"); fflush(stdout);
        printf("Progress bar"); 

        // Start the progress bar
        sampler_progress<<<1, 1, 0,streams[0] >>>(blocks, d_block_progress);

        start = clock();
        GPU_parallel_stretch_move_sampler<<<blocks, threads_per_block, 0 , streams[1]>>>(nsteps, ndim, nwalkers, blocks, threads_per_block, d_args, 
            d_loglikliehoods, d_positions,
            2.0,  devState, d_block_progress );
        end = clock();
        cudaGetLastError();
        cudaDeviceSynchronize();
        //printf("\tfinished in %.2f s [%d models / s].", (double) (end-start), (int) (nsteps*nwalkers/(end-start))); 
        printf("\n-----------------------------------");fflush(stdout);
    }


    /*---------------------------
    Part 6 - Write out results
    ---------------------------*/
    // Write out
    printf("\nWriting results... "); fflush(stdout);
    write_out_results(burn_in, nsteps, ndim, nwalkers,
        blocks, threads_per_block,
        d_positions, d_loglikliehoods, output_filename);
    printf(" done.\n\n"); fflush(stdout);


    
    // free up host memory
    free(theta);
    free(args);
    free(time);
    free(LC);
    free(LC_ERR);
    cudaFree(d_time);
    cudaFree(d_LC);
    cudaFree(d_LC_ERR);


    // free up device memory
    if (CPU_OR_GPU)
    {
        cudaFree(d_positions);
        cudaFree(d_loglikliehoods);
        cudaFree(devState);
        cudaFree(d_time);
        cudaFree(d_LC);
        cudaFree(d_LC_ERR);
        cudaFree(d_args);
        cudaFree(d_block_progress);
    }


    //historgram_plot_2_axis();
    
    
}