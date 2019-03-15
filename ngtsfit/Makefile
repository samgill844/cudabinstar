cc=nvcc
FLAGS= -O3 -Xcompiler -fpic  --default-stream per-thread  -Xptxas -O3
SHARED_OBJECTS = ../lib

all : ngtsfit

ngtsfit : ngtsfit.cu loglike_functions.o
	${cc} ${FLAGS}  ${SHARED_OBJECTS}/*.o ngtsfit.cu loglike_functions.o -o ngtsfit
	rm *.o

loglike_functions.o : loglike_functions.cu
	${cc} --default-stream per-thread ${CUDALC}/*.o loglike_functions.cu -dc

clean:
	rm *.o
	rm ngtsfit

install : 
	mv ngtsfit ~/bin
	cp ngtsbin ~/bin
	cp ngtscorner ~/bin
	cp ngtstry ~/bin
	cp exoplanet.eu_catalog.csv ~/bin