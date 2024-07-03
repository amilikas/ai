#ifndef UMML_REGRSETS_INCLUDED
#define UMML_REGRSETS_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 FILE:   regrsets.hpp
 AUTHOR: Anastasios Milikas (amilikas@csd.auth.gr)
 
 Namespace
 ~~~~~~~~~
 umml
 
 Requirements
 ~~~~~~~~~~~~
 umml matrix
 umml rand
 STL string
  
 Description
 ~~~~~~~~~~~
 Functions to create datasets appropriate for optimization and regression problems: 
 * Unimodal1
 * Unimodal2
 * Easom
 * Ackley
 * Himmelblau
 * Peaks
 * Eggholder

 References
 ~~~~~~~~~~
 [1] Jason Brownlee: Two-Dimensional (2D) Test Functions for Function Optimization
 https://machinelearningmastery.com/2d-test-functions-for-function-optimization/ 

 Usage
 ~~~~~ 
 generate_regression_set(X, 3000, Himmelblau<double>());
 generate_swissroll(X, 2000);
 generate_scurve(X, 2000);
*/

#include "../umat.hpp"


namespace umml {


// global minimum at (0,0)
template <typename T=float>
struct Unimodal1 {
	T operator()(T x0, T x1) {
		return x0*x0 + x1*x1;
	}
};

// global minimum at (0,0)
template <typename T=float>
struct Unimodal2 {
	T operator()(T x0, T x1) {
		return 0.26 * (x0*x0 + x1*x1) - 0.48 * x0 * x1;
	}
};

// global minimum -1 at (pi,pi)
template <typename T=float>
struct Easom {
	T operator()(T x0, T x1) {
		const double pi = 3.14159265358979;
		double xp0 = x0-pi;
		double xp1 = x1-pi;
		double z = -std::cos(x0) * std::cos(x1) * std::exp(-(xp0*xp0 + xp1*xp1));
		return static_cast<T>(z);
	}
};

// global minimum 0 at (0,0)
template <typename T=float>
struct Ackley {
	T operator()(T x0, T x1) {
		const double pi = 3.14159265358979;
		double z = -20.0 * std::exp(-0.2 * std::sqrt(0.5 * (x0*x0 + x1*x1))) - 
					std::exp(0.5 * (std::cos(2*pi*x0) + std::cos(2*pi*x1))) + std::exp(1) + 20;
		return static_cast<T>(z);
	}
};

// global optima at [3.0,2.0], [-2.805118,3.131312], [-3.779310,-3.283186], [3.584428,-1.848126]
template <typename T=float>
struct Himmelblau {
	T operator()(T x0, T x1) {
		double f1 = x0*x0 + x1 - 11.0;
		double f2 = x0 + x1*x1 - 7.0;
		double z = (f1*f1 + f2*f2) / 100.0;
		return static_cast<T>(z);
	}
};

template <typename T=float>
struct Peaks {
	T operator()(T x0, T x1) {
		double z =  3*(1-x0)*(1-x0)*std::exp(-(x0*x0) - (x1+1)*(x1+1)) -
					10*(x0/5 - x0*x0*x0 - x1*x1*x1*x1*x1) * std::exp(-x0*x0-x1*x1) -
					1/3.0*std::exp(-(x0+1)*(x0+1) - x1*x1);
		return static_cast<T>(z);
	}
};

// global minimum -959.6407 at (512,404.2319)
template <typename T=float>
struct Eggholder {
	T operator()(T x0, T x1) {
		double z = -(x1+47) * std::sin(std::sqrt(std::abs(0.5*x0+x1+47))) - 
					x0 * std::sin(std::sqrt(std::abs(x0-(x1+47))));
		return static_cast<T>(z);
	}
};


// generate a 2d dataset (+1 dim for output) using function f
template <typename T, template <typename> class Functor>
void generate_regression_set(umat<T>& X, int n, Functor<T> f, 
							 double minval=-5.0, double maxval=5.0)
{
	uniform_real_distribution<double> dist(static_cast<double>(minval), static_cast<double>(maxval));
	X.resize(n,3);
	for (int i=0; i<n; i++) {
		double x0 = dist(global_rng());
		double x1 = dist(global_rng());
		X(i,0) = static_cast<T>(x0);
		X(i,1) = static_cast<T>(x1);
		X(i,2) = static_cast<T>(f(x0,x1));
	}
}

// generate a 3d swiss roll dataset
template <typename T>
void generate_swissroll(umat<T>& X, uvec<T>& y, int n, int rnd_state=-1)
{
	uniform_real_distribution<T> dist(0, 1);
	const double pi = 3.14159265358979;
	X.resize(n,3);
	y.resize(n);
	for (int i=0; i<n; i++) {
		double t = 1.5 * pi * (1 + 2*dist(global_rng()));
		X(i,0) = static_cast<T>(t * std::cos(t));
		X(i,1) = static_cast<T>(21 * dist(global_rng()));
		X(i,2) = static_cast<T>(t * std::sin(t));
		y(i) = static_cast<T>(t);
	}
}

// generate a 3d s-curve dataset
template <typename T>
void generate_scurve(umat<T>& X, uvec<T>& y, int n, int rnd_state=-1)
{
	uniform_real_distribution<T> dist(0, 1);
	const double pi = 3.14159265358979;
	X.resize(n,3);
	y.resize(n);
	for (int i=0; i<n; i++) {
		double t = 3 * pi * (dist(global_rng()) - 0.5);
		X(i,0) = static_cast<T>(std::sin(t));
		X(i,1) = static_cast<T>(2 * dist(global_rng()));
		X(i,2) = static_cast<T>((t < 0.0 ? -1 : 1) * (std::cos(t)-1));
		y(i) = static_cast<T>(t);
	}
}


};     // namespace umml

#endif // UMML_REGRSETS_INCLUDED
