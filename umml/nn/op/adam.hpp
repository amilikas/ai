#ifndef UMML_ADAM_INCLUDED
#define UMML_ADAM_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Gradient Descent

 FILE:     adam.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: ann, neural networks, gradient descent, adam
 
 Namespace
 ~~~~~~~~~
 umml::gdstep

 Description
 ~~~~~~~~~~~
 Adam gradient descent step
 r: stepsize (learning rate)
 b1, b2: exponential decay rates for the moment estimates
 m, v: 1st and 2nd moment vector
 e: regularizer to control divisions by zero
 t: timestep

 m = b1*m + (1-b1)*dw
 v = b2*v + (1-b2)*dw.squared()
 mh = m / (1-powi(b1,t))
 vh = v / (1-powi(b2,t))
 g = (r*mh) / (vh.sqrt()+e)
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
void cpu_adam(Type* g, Type r, Type b1, Type b2, Type b1t, Type b2t, Type e, Type* m, Type* v, int n)
{
	// update moments
	for (int i=0; i<n; ++i) {
		m[i] = b1*m[i] + (1-b1)*g[i];
		v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
	}
//std::cout << "m: "; for (int i=0; i<n; ++i) std::cout << m[i] << " "; std::cout << "\n";
//std::cout << "v: "; for (int i=0; i<n; ++i) std::cout << v[i] << " "; std::cout << "\n";

//std::stringstream smh, svh;
	// compute bias-corrected moments estimates
	for (int i=0; i<n; ++i) {
		Type mh = m[i]/b1t;
		Type vh = v[i]/b2t;
//smh << mh << " ";
//svh << vh << " ";
		g[i] = (r*mh) / (std::sqrt(vh) + e);
	}

//std::cout << "m_hat: " << smh.str() << "\n";
//std::cout << "v_hat: " << svh.str() << "\n";
//std::cout << "g: "; for (int i=0; i<n; ++i) std::cout << g[i] << " "; std::cout << "\n";

}


// CUDA implementation
#ifdef __USE_CUDA__
template <typename Type>
__global__ void cuda_adam1(Type* g, Type b1, Type b2, Type* m, Type* v, int n)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		m[i] = b1*m[i] + (1-b1)*g[i];
		v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
	}
}
template <typename Type>
__global__ void cuda_adam2(Type* g, Type r, Type b1t, Type b2t, Type e, Type* m, Type* v, int n)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		Type mh = m[i]/b1t;
		Type vh = v[i]/b2t;
		g[i] = (r*mh) / (sqrtf(vh) + e);
	}
}
template <>
__global__ void cuda_adam2<double>(double* g, double r, double b1t, double b2t, double e, double* m, double* v, int n)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		double mh = m[i]/b1t;
		double vh = v[i]/b2t;
		g[i] = (r*mh) / (sqrt(vh) + e);
	}
}
#endif


// OpenCL implementation
#ifdef __USE_OPENCL__
std::string ocl_adam1_code = R"(
__kernel void __name__(__global __type__* g, __type__ b1, __type__ b2, __global __type__* m, __global __type__* v, int n)
{ 
	const int i = get_global_id(0); 
	if (i < n) {
		m[i] = b1*m[i] + (1-b1)*g[i];
		v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
	}
}
)";
std::string ocl_adam2_code = R"(
__kernel void __name__(__global __type__* g, __type__ r, __type__ b1t, __type__ b2t, __type__ e, 
						__global __type__* m, __global __type__* v, int n)
{ 
	const int i = get_global_id(0); 
	if (i < n) {
		__type__ mh = m[i]/b1t;
		__type__ vh = v[i]/b2t;
		g[i] = (r*mh) / (sqrt(vh) + e);
	}
}
)";

struct __ocladam {
	cl::Kernel fadam1;
	cl::Kernel dadam1;
	cl::Kernel fadam2;
	cl::Kernel dadam2;

	__ocladam() {
		cl::Program::Sources sources;
		__ocl__.push_source_code(sources, ocl_adam1_code,  "fadam1",  "float");	
		__ocl__.push_source_code(sources, ocl_adam1_code,  "dadam1",  "double");
		__ocl__.push_source_code(sources, ocl_adam2_code,  "fadam2",  "float");	
		__ocl__.push_source_code(sources, ocl_adam2_code,  "dadam2",  "double");
		cl::Program program = __ocl__.compile_sources(sources);
		fadam1 = cl::Kernel(program, "fadam1");
		dadam1 = cl::Kernel(program, "dadam1");
		fadam2 = cl::Kernel(program, "fadam2");
		dadam2 = cl::Kernel(program, "dadam2");
	}

	template <typename Type>
	void adam(cl::Buffer& g, Type r, Type b1, Type b2, Type b1t, Type b2t, Type e, cl::Buffer& m, cl::Buffer& v, int n) {
		cl::Kernel& kernel1 = (sizeof(Type)==sizeof(double) ? dadam1 : fadam1);
		kernel1.setArg(0, g);
		kernel1.setArg(1, b1);
		kernel1.setArg(2, b2);
		kernel1.setArg(3, m);
		kernel1.setArg(4, v);
		kernel1.setArg(5, n);
		__ocl__.execute(kernel1, cl::NullRange, cl::NDRange(DIMPAD(n)), cl::NullRange);
		
		cl::Kernel& kernel2 = (sizeof(Type)==sizeof(double) ? dadam2 : fadam2);
		kernel2.setArg(0, g);
		kernel2.setArg(1, r);
		kernel2.setArg(2, b1t);
		kernel2.setArg(3, b2t);
		kernel2.setArg(4, e);
		kernel2.setArg(5, m);
		kernel2.setArg(6, v);
		kernel2.setArg(7, n);
		__ocl__.execute(kernel2, cl::NullRange, cl::NDRange(DIMPAD(n)), cl::NullRange);
	}
}; // struct

// actual interface via the static instance
static __ocladam __ocladam__;
#endif


template <typename T, 
		  template <typename> class GVector, 
		  template <typename> class Vector>
void apply_adam(GVector<T>& g, T r, T b1, T b2, T b1t, T b2t, T e, Vector<T>& m, Vector<T>& v)
{
	assert(g.dev()==m.dev() && g.dev()==v.dev());

	if (g.dev()==device::GPU) {
	#if defined(__USE_CUDA__)
	const int GROUPS = (g.len()+THREADS-1) / THREADS;
	cuda_adam1<T><<<GROUPS,THREADS>>>(g.dmem(), b1, b2, m.dmem(), v.dmem(), g.len());
	cuda_adam2<T><<<GROUPS,THREADS>>>(g.dmem(), r, b1t, b2t, e, m.dmem(), v.dmem(), g.len());
	__cuda__.synchronize();
	#elif defined(__USE_OPENCL__)
	__ocladam__.adam<T>(g.dmem(), r, b1, b2, b1t, b2t, e, m.dmem(), v.dmem(), g.len());
	#endif
	return;
	}

	cpu_adam(g.mem(), r, b1, b2, b1t, b2t, e, m.mem(), v.mem(), g.len());
}


};     // namespace gdstep
};     // namespace umml

#endif // UMML_ADAM_INCLUDED
