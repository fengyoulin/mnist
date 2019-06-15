all:
	g++ -O4 -std=c++11 -msse2 -msse3 -msse4 -mavx -mavx2 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -Ieigen-eigen-323c052e1731 -omnist mnist.cpp

clean:
	rm -f mnist mnist.exe
