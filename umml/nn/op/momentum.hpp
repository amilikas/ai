#ifndef UMML_MOMENTUM_INCLUDED
#define UMML_MOMENTUM_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Gradient Descent

 FILE:     momentum.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: ann, neural networks, gradient descent, momentum
 
 Namespace
 ~~~~~~~~~
 umml::gdstep

 Description
 ~~~~~~~~~~~
 Momentum gradient descent step
 r: learning rate
 b: momentum
 m: previous gradient
 
 step = r * (b*m + g)
*/


#include "../../utils.hpp"
#ifdef __USE_CUDA__
#include "../../cuda.hpp"
#endif
#ifdef __USE_OPENCL__
#include "../../ocl.hpp"
#endif


namespace umml {
namespace gdstep {


// CPU implementation
template <typename Type>
void cpu_momentum(Type* g, Type r, Type b, Type* m, int n)
{
	for (int i=0; i<n; ++i) {
		m[i] = b*m[i] + g[i];
		g[i] = r*m[i];
	}
}


// CUDA implementation
#ifdef __USE_CUDA__
template <typename Type>
__global__ void cuda_momentum(Type* g, Type r, Type b, Type* m, int n)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		m[i] = b*m[i] + g[i];
		g[i] = r*m[i];
	}
}
#endif


// OpenCL implementation
#ifdef __USE_OPENCL__
std::string ocl_momentum_code = R"(
__kernel void __name__(__global __type__* g, __type__ r, __type__ b, __global __type__* m, int n) 
{ 
	const int i = get_global_id(0); 
	if (i < n) {
		m[i] = b*m[i] + g[i];
		g[i] = r*m[i];
	}
}
)";

struct __oclmomentum {
	cl::Kernel fmomentum;
	cl::Kernel dmomentum;

	__oclmomentum() {
		cl::Program::Sources sources;
		__ocl__.push_source_code(sources, ocl_momentum_code,  "fmomentum",  "float");	
		__ocl__.push_source_code(sources, ocl_momentum_code,  "dmomentum",  "double");
		cl::Program program = __ocl__.compile_sources(sources);
		fmomentum  = cl::Kernel(program, "fmomentum");
		dmomentum  = cl::Kernel(program, "dmomentum");
	}

	template <typename Type>
	void momentum(cl::Buffer& g, Type r, Type b, cl::Buffer& m, int n) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? dmomentum : fmomentum);
		kernel.setArg(0, g);
		kernel.setArg(1, r);
		kernel.setArg(2, b);
		kernel.setArg(3, m);
		kernel.setArg(4, n);
		__ocl__.execute(kernel, cl::NullRange, cl::NDRange(DIMPAD(n)), cl::NullRange);
	}
}; // struct

// actual interface via the static instance
static __oclmomentum __oclmomentum__;
#endif


template <typename T, 
		  template <typename> class GVector, 
		  template <typename> class Vector>
void apply_momentum(GVector<T>& g, T r, T b, Vector<T>& m)
{
	assert(g.dev()==m.dev());

	if (g.dev()==device::GPU) {
	#if defined(__USE_CUDA__)
	const int GROUPS = (g.len()+THREADS-1) / THREADS;
	cuda_momentum<T><<<GROUPS,THREADS>>>(g.dmem(), r, b, m.dmem(), g.len());
	__cuda__.synchronize();
	#elif defined(__USE_OPENCL__)
	__oclmomentum__.momentum<T>(g.dmem(), r, b, m.dmem(), g.len());
	#endif
	return;
	}

	cpu_momentum(g.mem(), r, b, m.mem(), g.len());
}


};     // namespace gdstep
};     // namespace umml

#endif // UMML_MOMENTUM_INCLUDED
