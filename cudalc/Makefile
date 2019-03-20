cc=nvcc
COMPILER_FLAGS = -O3 -Xcompiler -fpic -rdc=true --default-stream per-thread -Xptxas -O3 


all : kepler.o rv.o flux_drop.o lc.o 

lc.o : lc.cu flux_drop.o kepler.o
	${cc} ${COMPILER_FLAGS} lc.cu flux_drop.o kepler.o -dc

flux_drop.o : flux_drop.cu
	${cc} ${COMPILER_FLAGS} flux_drop.cu -dc

rv.o : rv.cu kepler.o
	${cc} ${COMPILER_FLAGS} rv.cu kepler.o -dc

kepler.o : kepler.cu
	${cc} ${COMPILER_FLAGS} kepler.cu -dc


clean:
	rm *.o

install : 
	mv *.o ../lib
	cp *.h ../include
	nvcc -Xcompiler -fPIC -shared -o cudalc.so *.cu
	mv cudalc.so ../libso