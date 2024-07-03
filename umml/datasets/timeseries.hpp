#ifndef UMML_TIMESERIES_INCLUDED
#define UMML_TIMESERIES_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 FILE:   timeseries.hpp
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
 Time series datasets
 * shampoo (univariate)
 * binary univariate time series
 * univariate time series generation

 References
 ~~~~~~~~~~
 [1] Recurrent Neural Networks in Tensorflow I
 https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
  
 Usage
 ~~~~~ 
 generate_wave(X, 3000, Sinewave<float>(), noise, resolution);
*/

#include "../umat.hpp"


namespace umml {


// Sinewave univariate time series
template <typename T=float>
struct Sinewave {
	T operator()(T x) { return std::sin(x); }
};



// loads the shampoo univariate dataset.
// This dataset describes the monthly number of sales of shampoo over a 3 year period.
// The units are a sales count and there are 36 observations. The original dataset is 
// credited to Makridakis, Wheelwright and Hyndman (1998).
// Format: 36 rows of monthly sales
template <typename T>
void load_shampoo(umat<T>& X)
{
	X.resize(3*12,1);
	X.set("266.0, 145.9, 183.1, 119.3, 180.3, 168.5, 231.8, 224.5, 192.8, 122.9, 336.5, 185.9,"
		  "194.3, 149.5, 210.1, 273.3, 191.4, 287.0, 226.0, 303.6, 289.9, 421.6, 264.5, 342.3,"
		  "339.7, 440.4, 315.9, 439.3, 401.3, 437.4, 575.5, 407.6, 682.0, 475.3, 581.3, 646.9");
}


// generate a simple binary time series dataset with 'n' time steps and
// two time depedencies t1 and t2 [1]
template <typename T>
void generate_binary_timeseries(umat<T>& X, int n, int t1=3, int t2=8)
{
    uniform_real_distribution<T> dist(0, 1);
    X.resize(n,1);
    for (int t=0; t<n; ++t) {
        double threshold = 0.5;
        if (t >= t1 && X(t-t1,0)==T(1)) threshold += 0.5; 
        if (t >= t2 && X(t-t2,0)==T(1)) threshold -= 0.25; 
        T x = (dist(global_rng()) > threshold) ? T(0) : T(1);
        X(t,0) = x;
    }
}


// generate a simple univariate time series dataset with 'n' time steps
template <typename T, template <typename> class Functor>
void generate_wave(umat<T>& X, int n, Functor<T> f, double noise=0.0, double resolution=10.0)
{
	uniform_real_distribution<T> dist(0, noise);
	X.resize(n,1);
	for (int t=0; t<n; ++t) {
		T x = static_cast<T>((double)(t+1)/resolution);
		X(t,0) = f(x) + dist(global_rng());
	}
}


};     // namespace umml

#endif // UMML_TIMESERIES_INCLUDED
