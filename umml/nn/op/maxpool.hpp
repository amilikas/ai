#ifndef UMML_MAXPOOL_INCLUDED
#define UMML_MAXPOOL_INCLUDED


#include "../../utils.hpp"


// CPU implementation
#include "maxpool_cpu.hpp"

// CUDA implementation
#ifdef __USE_CUDA__
#include "../../cuda.hpp"
#include "maxpool_cuda.hpp"
#endif

// OpenCL implementation
#ifdef __USE_OPENCL__
#include "../../ocl.hpp"
#include "maxpool_ocl.hpp"
#endif


namespace umml {


template <typename T, 
		  template <typename> class CubeIn, 
		  template <typename> class CubeOut>
void maxpool2d(const CubeIn<T>& a, int k, int stride, CubeOut<T>& c)
{
	assert(a.dev()==c.dev());
	assert(a.zdim()==c.zdim());

	#ifdef __USE_CUDA__
	if (a.dev()==device::GPU) {
	__cudapool__.maxpool2d<T>(a.cdmem(), a.zdim(), a.ydim(), a.xdim(), a.xsize(), a.zsize(),
							  k, stride, c.dmem(), c.xsize(), c.zsize());
	return;
	}
	#endif

	#ifdef __USE_OPENCL__
	if (a.dev()==device::GPU) {
	__oclpool__.maxpool2d<T>(a.cdmem(), a.zdim(), a.ydim(), a.xdim(), a.xsize(), a.zsize(),
							 k, stride, c.dmem(), c.xsize(), c.zsize());
	return;
	}
	#endif

	cpu_maxpool2d(a.cmem(), a.zdim(), a.ydim(), a.xdim(), a.xsize(), a.zsize(),
				  k, stride, c.mem(), c.xsize(), c.zsize());
}


template <typename T, 
		  template <typename> class CubeIn, 
		  template <typename> class CubeOut,
 		  class CubIdx>
void maxpool2d(const CubeIn<T>& a, int k, int stride, CubeOut<T>& c, CubIdx& idcs)
{
	assert(a.dev()==c.dev() && a.dev()==idcs.dev());
	assert(a.zdim()==c.zdim() && a.zdim()==idcs.zdim());
	

	#ifdef __USE_CUDA__
	if (a.dev()==device::GPU) {
	__cudapool__.maxpool2d<T>(a.cdmem(), a.zdim(), a.ydim(), a.xdim(), a.xsize(), a.zsize(),
							  k, stride, c.dmem(), c.xsize(), c.zsize(), idcs.dmem());
	return;
	}
	#endif

	#ifdef __USE_OPENCL__
	if (a.dev()==device::GPU) {
	__oclpool__.maxpool2d<T>(a.cdmem(), a.zdim(), a.ydim(), a.xdim(), a.xsize(), a.zsize(),
							 k, stride, c.dmem(), c.xsize(), c.zsize(), idcs.dmem());
	return;
	}
	#endif

	cpu_maxpool2d(a.cmem(), a.zdim(), a.ydim(), a.xdim(), a.xsize(), a.zsize(),
				  k, stride, c.mem(), c.xsize(), c.zsize(), idcs.mem());
}

};     // namespace umml

#endif // UMML_MAXPOOL_INCLUDED
