all : cudalc cudamc ngtsfit
	(cd cudalc; make; make install)
	(cd cudamc; make; make install)

clean : 
	rm -f include/*
	rm -f lib/*