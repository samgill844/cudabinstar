all : cudalc cudamc ngtsfit
	mkdir -p lib 
	mkdir -p include
	mkdir -p ~/bin
	(cd cudalc; make; make install)
	(cd cudamc; make; make install)
	(cd ngtsfit; make; make install)


clean : 
	rm -f include/*
	rm -f lib/*