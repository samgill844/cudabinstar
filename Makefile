all : 
	mkdir -p lib 
	mkdir -p include
	mkdir -p ~/bin
	(cd cudabinstar/cudalc; make; make install)
	(cd cudabinstar/cudamc; make; make install)
	(cd ngtsfit; make; make install)
	(cd ngtsfieldfit; make install)


clean : 
	rm -f cudabinstar/cudalc/*.o
	rm -f cudabinstar/cudamc/*.o
	rm -f ngtsfit/*.o
	rm -f ngtsfit/ngtsfit

	rm -f include/*
	rm -f lib/*
	rm -f libso/*