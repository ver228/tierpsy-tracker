all : setup.py imageDifferenceMask.pyx
	python3 setup.py build_ext --inplace -I`python3 -c "import numpy; print(numpy.get_include())"`
	

.PHONY : clean
clean :
	rm imageDifferenceMask.c
	rm -R build