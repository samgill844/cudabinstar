cc=nvcc
COMPILER_FLAGS = -O3 -Xcompiler -fpic -rdc=true --default-stream per-thread -Xptxas -O3

all : ensemble.o sampler_helper.o

ensemble.o : ensemble.cu sampler_helper.o
	${cc} ${COMPILER_FLAGS} ensemble.cu sampler_helper.o   -c 

sampler_helper.o : sampler_helper.cu 
	${cc} ${COMPILER_FLAGS} sampler_helper.cu  -c 

clean:
	rm *.o

install : 
	mv *.o ../lib
	cp *.h ../include