all:
	g++ -O4 -std=c++11 -msse2 -msse3 -msse4 -mavx -mavx2 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -Ieigen-eigen-323c052e1731 -omnist mnist.cpp

mkl:
	g++ -O4 -std=c++11 -msse2 -msse3 -msse4 -mavx -mavx2 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DEIGEN_USE_MKL_ALL -Ieigen-eigen-323c052e1731 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_core -lmkl_sequential -lmkl_blas95_lp64 -lmkl_gf_lp64 -lmkl_lapack95_lp64 -lgomp -omnist mnist.cpp

clean:
	rm -f mnist mnist.exe
