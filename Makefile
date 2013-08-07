CXX=g++
CUDAPATH?=/usr/local/cuda

libcudahook.so: cudahook.cpp
	$(CXX) -I$(CUDAPATH)/include -Wall -fPIC -shared -ldl -o libcudahook.so cudahook.cpp

clean:
	-rm libcudahook.so
