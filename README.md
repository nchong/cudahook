cudahook
========

A library for intercepting CUDA runtime calls with LD_PRELOAD and dlsym.

Compiling
---------

We use hook mechanisms that are specific to GNU/Linux.

    $ make

should work, although you need to edit the CUDAPATH variable in the Makefile if CUDA is not installed to /usr/local/cuda

Using
-----

    $ LD_PRELOAD=/path/to/cudahook/libcudahook.so ./your-program
   
will run the CUDA program and intercept the following runtime calls:

   * cudaConfigureCall
   * cudaLaunch
   * cudaRegisterFunction
   * cudaSetupArgument

At the moment, we use this information to print out information about invoked kernels.
