all : cudalc cudamc ngtsfit
	mkdir -p lib 
	mkdir -p libso
	mkdir -p include
	mkdir -p ~/bin
	(cd cudalc; make; make install)
	(cd cudamc; make; make install)
	(cd ngtsfit; make; make install)
	(cd ngtsfieldfit; make install)


clean : 
	rm -f cudalc/*.o
	rm -f cudamc/*.o
	rm -f ngtsfit/*.o
	rm -f ngtsfit/ngtsfit
	
	rm -f include/*
	rm -f lib/*
	rm -f libso/*