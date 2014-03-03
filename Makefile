CXX=g++
CUDAPATH?=/usr/local/cuda

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
OPENCL_LIB = -framework OpenCL
OPENCL_INC =
all: libcudahook.so libclhook.so
endif

ifeq ($(UNAME), Linux)
.PHONY: .check-env
.check-env:
	@if [ ! -d "$(CLDIR)" ]; then \
		echo "ERROR: set CLDIR variable."; exit 1; \
	fi
OPENCL_LIB = -L$(CLDIR)/lib -lOpenCL
OPENCL_INC = -I $(CLDIR)/include
all: libcudahook.so libclhook.so
endif

COMMONFLAGS=-Wall -fPIC -shared -ldl

libcudahook.so: cudahook.cpp
	$(CXX) -I$(CUDAPATH)/include $(COMMONFLAGS) -o libcudahook.so cudahook.cpp

libclhook.so: clhook.cpp
	$(CXX) $(OPENCL_INC) $(OPENCL_LIB) $(COMMONFLAGS) -o libclhook.so clhook.cpp

libclhook.dylib: clhook.cpp
	$(CXX) $(OPENCL_INC) $(OPENCL_LIB) -Wall -dynamiclib -o libclhook.dylib clhook.cpp

clean:
	-rm libcudahook.so
