#ifndef UMML_FUNC_INCLUDED
#define UMML_FUNC_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 General functions and Activation functions.

 FILE:     func.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2023-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 Binary step: 0 (x<0) or 1 (x>=0). Derivative: 1 (x<>0) or undefined (x=0)
 Softplus: ln(1+e^x). Derivative: 1/(1+e^-x)
 Logistic: 1/(1+e^-x). Derivative: f(1-f)
 Hyperbolic Tangent (tanh): tanh(x). Derivative: 1-f^2
 ReLU: max(0,x). Derivative: 0 (x<0), 1 (x>0) or undefined (x=0)
 LReLU: max(0.1x,x). Derivative: 0.1 (x<0), 1 (x>0) or undefined (x=0)
 ?? Swish: x*logistix(x). Derivative: f+logistic(x)*(1–f)  <-- x?
 
 [1] Pragati Baheti: 12 Types of Neural Network Activation Functions
 https://www.v7labs.com/blog/neural-networks-activation-functions
*/


#include "dev.hpp"
#include <cmath>


namespace umml {


//
// BEWARE!!!
// Changing these values will break OpenCL implementation!
//
enum {
	fLinear, dLinear,
	fLog, dLog,
	fStep, dStep,
	fSoftplus, dSoftplus,
	fLogistic, dLogistic,
	fTanh, dTanh,
	fReLU, dReLU,
	fExp, dExp,
};

#ifdef __USE_OPENCL__
std::string ocl_fenums_code = R"(
enum {
	fLinear, dLinear,
	fLog, dLog,
	fStep, dStep,
	fSoftplus, dSoftplus,
	fLogistic, dLogistic,
	fTanh, dTanh,
	fReLU, dReLU,
	fExp, dExp,
};
)";
#endif


std::string function_name(int f)
{
	switch (f) {
	case fLinear: 	return "Linear";
	case fLog: 		return "log";
	case fStep: 	return "Step";
	case fSoftplus: return "Softplus";
	case fLogistic: return "Logistic";
	case fTanh: 	return "tanh";
	case fReLU: 	return "ReLU";
	case fExp: 		return "Exp";
	}
	return "-other-";
}


template <typename Type>
Type cpu_function(Type x, int f)
{
	switch (f) {
	case fLinear: 	return x;
	case dLinear: 	return 1;
	case fLog: 		return x != 0 ? log(x) : 0;
	case dLog: 		return x != 0 ? 1/x : 0;
	case fStep: 	return x < 0 ? 0 : 1; 
	case dStep: 	return x != 0 ? 1 : 0; 
	case fSoftplus: return log(1 + exp(x));
	case dSoftplus: return 1 / (1+exp(-x));
	case fLogistic: return 1 / (1+exp(-x));
	case dLogistic: return x * (1 - x);
	case fTanh: 	return tanh(x);
	case dTanh: 	return 1 - x*x;
	case fReLU: 	return x <= 0 ? 0 : x;
	case dReLU: 	return x <= 0 ? 0 : 1;
	case fExp: 		return exp(x);
	case dExp: 		return exp(x);
	}
	return 0;
}


#ifdef __USE_CUDA__
template <typename Type>
__device__ Type gpu_function(Type x, int f)
{
	switch (f) {
	case fLinear: 	return x;
	case dLinear: 	return 1;
	case fLog: 		return x != 0 ? log(x) : 0;
	case dLog: 		return x != 0 ? 1/x : 0;
	case fStep: 	return x < 0 ? 0 : 1; 
	case dStep: 	return x != 0 ? 1 : 0; 
	case fSoftplus: return log(1 + exp(x));
	case dSoftplus: return 1 / (1+exp(-x));
	case fLogistic: return 1 / (1+exp(-x));
	case dLogistic: return x * (1 - x);
	case fTanh: 	return tanh(x);
	case dTanh: 	return 1 - x*x;
	case fReLU: 	return x <= 0 ? 0 : x;
	case dReLU: 	return x <= 0 ? 0 : 1;
	case fExp: 		return exp(x);
	case dExp: 		return exp(x);
	}
	return 0;
}
#endif


#ifdef __USE_OPENCL__
std::string ocl_funcs_code = R"(
__type__ ocl___type___func(__type__ x, int f) 
{
	switch (f) {
	case fLinear:  	return x;
	case dLinear:  	return (__type__)1;
	case fLog: 		return x != (__type__)0 ? log(x) : (__type__)0;
	case dLog: 		return x != (__type__)0 ? 1/x : (__type__)0;
	case fStep: 	return x < (__type__)0 ? (__type__)0 : (__type__)1; 
	case dStep: 	return x != (__type__)0 ? (__type__)1 : (__type__)0; 
	case fSoftplus: return (__type__)log(1 + exp(x));
	case dSoftplus: return (__type__)1 / (1+exp(-x));
	case fLogistic: return (__type__)1 / (1+exp(-x));
	case dLogistic: return x * (1 - x);
	case fTanh: 	return (__type__)tanh(x);
	case dTanh: 	return (__type__)1 - x*x;
	case fReLU: 	return x <= (__type__)0 ? (__type__)0 : x;
	case dReLU: 	return x <= (__type__)0 ? (__type__)0 : (__type__)1;
	case fExp: 		return exp(x);
	case dExp: 		return exp(x);
	}
	return 0;
}
)";
#endif


};     // namespace umml

#endif // UMML_FUNC_INCLUDED
