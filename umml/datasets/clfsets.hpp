#ifndef UMML_CLFSETS_INCLUDED
#define UMML_CLFSETS_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 FILE:   clfsets.hpp
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
 Datasets appropriate for classification problems: 
 * generate_moons: 2 class dataset
 * generate_circles: 2 class dataset
 * generate_blobs: multiclass dataset, linear separable
 
 Usage
 ~~~~~ 
*/

#include "../umat.hpp"


namespace umml {


// Moons, two interleaving half circles in 2d
// A simple toy dataset to visualize clustering and classification algorithms.
// X(n,2): the generated dataset.
// y(n): the generated labels (0 or 1)
// n: total number of points (moon1+moon2)
template <typename XT, typename YT>
void generate_moons(umat<XT>& X, uvec<YT>& y, int n, double noise=0.1)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);
	
	const double pi = 3.14159265358979;	
	uniform_real_distribution<XT> dist(0, noise);

	if (n < 2) n = 2;
	if (n & 1) --n;
	X.resize(n,2);
	y.resize(n);
	
	double step = pi/(n/2);
	double x0 = 0.5-dist(global_rng());
	double x1 = 0.5-dist(global_rng());
	for (int i=0; i<n; i+=2) {
		X(i,0) = static_cast<XT>(std::cos(x0));
		X(i,1) = static_cast<XT>(std::sin(x1)+dist(global_rng()));
		y(i) = static_cast<YT>(0);
		X(i+1,0) = static_cast<XT>(1) - X(i,0);
		X(i+1,1) = static_cast<XT>(0.5+dist(global_rng())) - X(i,1);
		y(i+1) = static_cast<YT>(1);
		x0 += step;
		x1 += step;
	}
}


// Circles, a large circle containing a smaller circle in 2d
// A simple toy dataset to visualize clustering and classification algorithms.
// X(n,2): the generated dataset.
// y(n): the generated labels (0 or 1)
// n: total number of points (circle1+circle2)
template <typename XT, typename YT>
void generate_circles(umat<XT>& X, uvec<YT>& y, int n, double noise=0.1)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);
	
	const double pi = 3.14159265358979;	
	uniform_real_distribution<XT> dist(0, noise);

	if (n < 2) n = 2;
	if (n & 1) --n;
	X.resize(n,2);
	y.resize(n);
	
	double factor = 0.75;
	double step = 2.0*pi/(n/2);
	double x0 = 0.5-dist(global_rng());
	double x1 = 0.5-dist(global_rng());
	for (int i=0; i<n; i+=2) {
		X(i,0) = static_cast<XT>(std::cos(x0));
		X(i,1) = static_cast<XT>(std::sin(x1)+dist(global_rng()));
		y(i) = static_cast<YT>(0);
		X(i+1,0) = static_cast<XT>(X(i,0)*factor);
		X(i+1,1) = static_cast<XT>(X(i,1)*factor);
		y(i+1) = static_cast<YT>(1);
		x0 += step;
		x1 += step;
	}
}


// Blobs
// A simple toy dataset to visualize clustering and classification algorithms.
// X(n,d): the generated dataset.
// y(n): the generated labels (0..nb-1)
// n: total number of points for all blobs
// d: dimensions (default: 2)
// nb: number of blobs (default: 2)
template <typename XT, typename YT>
void generate_blobs(umat<XT>& X, uvec<YT>& y, int n, int nb=2, int d=2, double noise=0.1)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);

	uniform_real_distribution<XT> dist(0, noise);

	n += (n % nb);
	if (n < nb) n = nb;
	X.resize(n,d);
	y.resize(n);
	
	umat<XT> centers(nb,d);
	for (int k=0; k<nb; ++k)
	for (int j=0; j<d; ++j) centers(k,j) = static_cast<XT>(nb*k*noise + dist(global_rng()));
	
	for (int i=0; i<n; i+=nb) {
		for (int k=0; k<nb; ++k) {
			for (int j=0; j<d; j++) {
				X(i+k,j) = centers(k,j) + dist(global_rng());
			}
			y(i+k) = static_cast<YT>(k);
		}
	}
}


};     // namespace umml

#endif // UMML_CLFSETS_INCLUDED
