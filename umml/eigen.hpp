#ifndef UMML_EIGEN_INCLUDED
#define UMML_EIGEN_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Eigevectors and Eigenvalues using the Power Iteration method.

 FILE:     eigen.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: eigen decomposition
 
 Namespace
 ~~~~~~~~~
 umml

 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix

 Internal dependencies:
 STL vector
 
 eigen algorithm
 ~~~~~~~~~~~~~~~
 Uses Power Iteration method to find the first 'neigs' eigenvalues and eigenvectors.
 If 'neigs' is 0 it finds the min(m,n) eigenvalues and eigenvectors (m=nrows, n=ncols).
 (re)Allocateds memory for vector 'eigvals' and matrix 'eigvecs' to store the results.
 
 eigvals:
   { val1, val2, ... valn }, val1 >= val2 >= ... valn

 eigvecs:
   v11  v21  ...
   v12  v22  ...
   v13  v23  ...
    .    .   ...
   v1m  v2m  ...

 
 Functions
 ~~~~~~~~~
 * eigen(in:matrix, out:eigenvalues, out:eigenvecs, neigen, maxiterations, tolerance)
 
 TODO
 ~~~~
*/


#include "umat.hpp"


namespace umml {


template <typename Type=float>
void eigen(const umat<Type>& M, uvec<Type>& eigvals, umat<Type>& eigvecs, int neigs=0, 
		   size_t maxiters=100, double tolerance=1e-8) 
{ 
	assert(M.dev()==device::CPU && eigvals.dev()==device::CPU && eigvecs.dev()==device::CPU);
	
	// number of eigenvectors must be min(nrows,ncols)	
	int n = M.ydim();
	if (neigs==0 || neigs > M.xdim()) {
		neigs = M.xdim();
		if (neigs > n) neigs = n;
	}
	
	// the power iteration method
	uvec<Type> u(n), s(n), tmp(n);
	umat<Type> uu(M.dev());
	std::vector<uvec<Type>> ulist;
	eigvals.resize(neigs);
	eigvals.zero_active_device();
	for (int i=0; i<neigs; ++i) {
		u.random_reals(-0.5, 0.5);
		Type eigval = std::numeric_limits<Type>::max();
		for (size_t k=0; k<maxiters; k++) {
			Type old_eigval = eigval;
			tmp.set(u);
			u.mul(M, tmp);
			if (i > 0) {
				s.zero_active_device();
				for (int j=0; j<i; ++j) {
					// s += (u.ulist[j]) * ulist[j];
					s.plus(ulist[j], u.dot(ulist[j]));
				}
				u.plus(s, Type(-1.0));
			}
			eigval = u.normalize();
			if (std::abs(eigval-old_eigval) < tolerance) break;
		}
		if (std::abs(eigval) < 1e-6/*tolerance*/) break;
		ulist.push_back(u);
		eigvals.set_element(i, eigval);
	}
	
	// store results and tranpose eigenvecs so the vectors are stored vertically.
	umat<Type> eigvecsT(M.dev());
	eigvecsT.resize(neigs, n);
	eigvecsT.zero_active_device();
	for (int i=0; i<(int)ulist.size(); ++i) eigvecsT.set_row(i, ulist[i].active_mem(), n);
	eigvecs.transpose(eigvecsT);
}


};     // namespace umml

#endif // UMML_EIGEN_INCLUDED
