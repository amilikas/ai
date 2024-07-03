#ifndef UMML_PCA_INCLUDED
#define UMML_PCA_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Principal Component Analysis (PCA)

 FILE:     pca.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: unsupervised, dimensionality reduction
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 PCA<double> works better than PCA<float>, since the later introduces many rounding errors
 due to lower precission. This is especially noticeable after an inverse_transform() 
  
 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 
 Internal dependencies:
 umml eigen
 STL vector

 Usage example
 ~~~~~~~~~~~~~
 PCA<> pca(2); // 2=number of the final (reduced) dimensions
 pca.fit_transform(X_train, X_reduced)
 pca.transform(X_test, X_test_reduced);
  
 Set parameters before fit() is called, eg:
 PCA<>::params opt; 
 opt.max_iters = 10;
 pca.set_params(opt);
*/


#include "eigen.hpp"


namespace umml {


/*
 Fast PCA algorithm for computing leading eigenvectors.

 Dimensionality reduction by projecting each data point onto only the first few principal 
 components to obtain lower-dimensional data while preserving as much of the data's variation 
 as possible.

 [1] Alok Sharma: Fast principal component analysis using fixed-point algorithm
 https://maxwell.ict.griffith.edu.au/spl/publications/papers/prl07_alok_pca.pdf

 1. Center the data
 2. Calculate the covariance matrix
 3. Choose h, the number of principal axes or eigenvectors required
    to estimate. Compute covariance Sx and set p=1
 4. Initialize eigenvector u(p) of size d*1 e.g. randomly
 5. Save u'(p) = u(p) and Update u(p) as u(p) = Sx * u(p)
 6. Do the Gram-Schmidt orthogonalization process u(p) = u(p)-Σ[u(p)^T*u(j))*u(j)]_j=1..p-1
 7. Normalize u(p) by dividing it by its norm: u(p) = u(p) / ||u(p)||
 8. If u(p) has not converged, go back to step 5
 9. Increment counter p = p + 1 and go to step 4 until p equals h
 Converge of u(p): abs(|u(p)|-|u'(p)|) < tolerance
*/

template <typename Type=float>
class PCA 
{
 public:
	// parameters
	struct params {
		double etolerance; // tolerance for eigen solver [default: 1e-8]
		size_t emax_iters; // maximum number of iterations for eigen solver [default: 100]
		params() {
			etolerance = 1e-8;
			emax_iters = 100;
		}
	};
 
	// constructor
	PCA(int n=0): n_components(n) {}
	void   set_params(const params& _opt) { opt = _opt; }
	params get_params() const { return opt; }
	
	// fit calculates the covariance matrix S and then the eigenvalues and eigenvectors of S.
	void fit(const umat<Type>& X) {
		assert(X.dev()==device::CPU && X.ydim() > 1);
		int nrows = X.ydim();
		int ncols = X.xdim();
		int ndims = n_components;
		if (ndims==0 || ndims > ncols) ndims = ncols;
		if (nrows < ndims) ndims = nrows;
		umat<Type> C(X.dev()), S(X.dev());
		C.resize_like(X);
		C.set(X);
		// center the data C -= C.mean(AxisX)
		v_mean.resize(ncols);
		C.reduce_sum(v_mean, AxisX);
		v_mean.mul(Type(1.0)/nrows);
		C.plus_vector(v_mean, AxisX, Type(-1.0));
		// calculate the covariance matrix S = 1/(nrows-1)*C.T()*C
		S.gram(C);
		S.mul(Type(1.0/(nrows-1)));
		// eigen decomposition
		eigen<Type>(S, eigvals, eigvecs, ndims, opt.emax_iters, opt.etolerance);
	}

	// transform the dataset X to Xreduced = (X-v_mean)*eigvecs;
	void transform(const umat<Type>& X, umat<Type>& Xreduced) {
		umat<Type> C(X.dev());
		C.resize_like(X);
		C.set(X);
		C.plus_vector(v_mean, AxisX, Type(-1.0));
		Xreduced.mul(C, eigvecs);
	}

	// fit and then transform the dataset X to Xreduced
	void fit_transform(const umat<Type>& X, umat<Type>& Xreduced) {
		fit(X);
		transform(X, Xreduced);
	}
	
	// reverse transformation of the Xreduced dataset to the original X 
	void inverse_transform(const umat<Type>& Xreduced, umat<Type>& X) {
		umat<Type> eigvecsT(eigvecs.dev());
		eigvecsT.transpose(eigvecs);
		X.mul(Xreduced, eigvecsT);
		X.plus_vector(v_mean, AxisX);
	}
	
	// get the eigenvalues and the eigenvectors
	const uvec<Type>& get_eigenvalues() const { return eigvals; }
	const umat<Type>& get_eigenvectors() const { return eigvecs; }
	
 private:
	// private data
	int        n_components;
	uvec<Type> v_mean;
	uvec<Type> eigvals;
	umat<Type> eigvecs;
	params     opt;
};


};     // namespace umml

#endif // UMML_PCA_INCLUDED
