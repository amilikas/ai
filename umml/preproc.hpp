#ifndef UMML_PREPROC_INCLUDED
#define UMML_PREPROC_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 FILE:     preproc.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: preprocessing, normalization, standardization, one-hot encoding, 

 
 Namespace
 ~~~~~~~~~
 mml
 
 Dependencies
 ~~~~~~~~~~~~
 mml uvec
 mml umat
 STL string
 STL vector

 Internal dependencies:
 STL algorithm
 STL map
  
 Classes
 ~~~~~~~
 * minmaxscaler: min-max scaler (normalization)
 * stdscaler: standardize scaler (standardization)
 * binary_conv: binary labels converter (+1/-1)
 * string_enc: text labels encoder, convert std::strings to ordered integers
 * onehot_enc: onehot encoder for integer categorical data.
 
 Functions
 ~~~~~~~~~
 uniques: returns a vector with the unique elements of a vector
*/




#include "umat.hpp"
#include <map>


namespace umml {


/* 
 minmaxscaler: min-max scaler
 range ZeroToOne (0..1): (x-xmin)/(xmax-xmin)
 range MinusOneToOne (-1..1): 2*(x-xmin)/(xmax-xmin)-1

 Usage:
 minmaxscaler<> scaler;
 scaler.fit_transform(X_train, X_train_scaled);
 scaler.transform(X_test, X_test_scaled);
*/
template <typename Type=float>
class minmaxscaler 
{
 public:
	// desired range of the scaled values
	enum {
		ZeroToOne,
		MinusOneToOne,
	};
	
	// constructor
	minmaxscaler(int _range=ZeroToOne): range(_range) {}
	
	// fits the scaler with the contents of matrix 'A'
	void fit(const umat<Type>& A) {
		assert(A.dev()==device::CPU && "minmaxscaler requires data stored on CPU memory");
		int nrows = A.ydim();
		int ncols = A.xdim();
		minv.resize(ncols);
		maxminv.resize(ncols);
		for (int j=0; j<ncols; j++) minv(j) = maxminv(j) = A(0,j);
		for (int i=1; i<nrows; i++) {
			for (int j=0; j<ncols; j++) {
				if (A(i,j) < minv(j)) minv(j) = A(i,j);
				else if (A(i,j) > maxminv(j)) maxminv(j) = A(i,j);
			}
		}
		for (int j=0; j<ncols; j++) maxminv(j) -= minv(j);
	}
	
	// transform matrix 'A' to matrix 'N'
	void transform(const umat<Type>& A, umat<Type>& N) {
		assert(A.dev()==device::CPU && "minmaxscaler requires data stored on CPU memory");
		assert(A.dev()==N.dev());
		int nrows = A.ydim();
		int ncols = A.xdim();
		N.resize(nrows, ncols);
		if (range==ZeroToOne) {
			for (int i=0; i<nrows; i++) {
				for (int j=0; j<ncols; j++) 
					N(i,j) = (A(i,j)-minv(j)) / maxminv(j);
			}
		} else {
			for (int i=0; i<nrows; i++) {
				for (int j=0; j<ncols; j++) 
					N(i,j) = (2*(A(i,j)-minv(j)) / maxminv(j)) - 1;
			}
		}
	}

	
	// fit and transform matrix 'A' to scaled matrix 'N'
	void fit_transform(const umat<Type>& A, umat<Type>& N) {
		fit(A);
		transform(A, N);
	}
	
	// inverse transformation of matrix 'N' to original 'A'
	void inverse_transform(const umat<Type>& N, umat<Type>& A) {
		assert(A.dev()==device::CPU && "minmaxscaler requires data stored on CPU memory");
		assert(A.dev()==N.dev());
		int nrows = N.ydim();
		int ncols = N.xdim();
		A.resize(nrows, ncols);
		if (range==ZeroToOne) {
			for (int i=0; i<nrows; i++) {
				for (int j=0; j<ncols; j++) 
					A(i,j) = N(i,j)*maxminv(j) + minv(j);
			}
		} else {
			for (int i=0; i<nrows; i++) {
				for (int j=0; j<ncols; j++) 
					A(i,j) = ((N(i,j)+1)*maxminv(j))/2 + minv(j);
			}
		}
	}

	// transform a single vector 'a' to scaled vector 'n'
	void transform(const uvec<Type>& a, uvec<Type>& n) {
		assert(a.dev()==device::CPU && "minmaxscaler requires data stored on CPU memory");		
		assert(a.dev()==n.dev());
		assert(a.len()==minv.len());
		n.resize(a.len());
		if (range==ZeroToOne) {
			for (int j=0; j<a.len(); j++) 
				n(j) = (a(j)-minv(j)) / maxminv(j);
		} else {
			for (int j=0; j<a.len(); j++) 
				n(j) = (2*(a(j)-minv(j)) / maxminv(j)) - 1;
		}
	}

	// inverse trasformation of a single vector 'n' to original vector 'a'
	void inverse_transform(const uvec<Type>& n, uvec<Type>& a) {
		assert(a.dev()==device::CPU && "minmaxscaler requires data stored on CPU memory");		
		assert(a.dev()==n.dev());
		assert(n.len()==minv.len());
		a.resize(n.len());
		if (range==ZeroToOne) {
			for (int j=0; j<n.len(); j++) 
				a(j) = n(j)*maxminv(j) + minv(j);
		} else {
			for (int j=0; j<n.len(); j++) 
				a(j) = ((n(j)+1)*maxminv(j))/2 + minv(j);
		}
	}
	
 protected:
	int        range;
	uvec<Type> minv;
	uvec<Type> maxminv;
};


/*
 stdscaler: standardize scaler
 (x - μ) / σ
 μ=mean of the training samples, σ is the standard deviation

 Usage:
 stdscaler<> scaler;
 scaler.fit_transform(X_train, X_train_scaled);
 scaler.transform(X_test, X_test_scaled);
*/
template <typename Type=float>
class stdscaler 
{
 public:
	stdscaler() {}
	
	void fit(const umat<Type>& A) {
		assert(A.dev()==device::CPU && "stdscaler requires data stored on CPU memory");
		int nrows = A.ydim();
		int ncols = A.xdim();
		u.resize(ncols); u.zero_active_(); 
		s.resize(ncols); s.zeros();
		for (int i=0; i<nrows; i++) {
			for (int j=0; j<ncols; j++) {
				u(j) += A(i,j);
			}
		}
		for (int j=0; j<ncols; j++) u(j) /= nrows;
		for (int i=0; i<nrows; i++) {
			for (int j=0; j<ncols; j++) {
				Type xu = A(i,j) - u(j);
				s(j) += xu*xu;
			}
		}
		for (int j=0; j<ncols; j++) {
			s(j) = std::sqrt(s(j)/nrows);
		}
	}
	
	// transform matrix 'A' to matrix 'N'
	void transform(const umat<Type>& A, umat<Type>& N) {
		assert(A.dev()==device::CPU && "stdscaler requires data stored on CPU memory");
		assert(A.dev()==N.dev());
		int nrows = A.ydim();
		int ncols = A.xdim();
		N.resize(nrows, ncols);
		for (int i=0; i<nrows; i++) {
			for (int j=0; j<ncols; j++) 
				N(i,j) = (A(i,j)-u(j)) / s(j);
		}
	}
	
	// fit and transform matrix 'A' to scaled matrix 'N'
	void fit_transform(const umat<Type>& A, umat<Type>& N) {
		fit(A);
		transform(A, N);
	}
	
	// inverse transformation of matrix 'N' to original 'A'
	void inverse_transform(const umat<Type>& N, umat<Type>& A) {
		assert(A.dev()==device::CPU && "stdscaler requires data stored on CPU memory");
		assert(A.dev()==N.dev());
		int nrows = N.ydim();
		int ncols = N.xdim();
		A.resize(nrows, ncols);
		for (int i=0; i<nrows; i++) {
			for (int j=0; j<ncols; j++) 
				A(i,j) = N(i,j)*s(j) + u(j);
		}
	}

	// transform a single vector 'a' to scaled vector 'n'
	void transform(const uvec<Type>& a, uvec<Type>& n) {
		assert(a.dev()==device::CPU && "stdscaler requires data stored on CPU memory");
		assert(a.dev()==n.dev());
		assert(a.len()==u.len());
		n.resize(a.len());
		for (int j=0; j<a.len(); j++) n(j) = (a(j)-u(j)) / s(j);
	}

	// inverse trasformation of a single vector 'n' to original vector 'a'
	void inverse_transform(const uvec<Type>& n, uvec<Type>& a) {
		assert(a.dev()==device::CPU && "stdscaler requires data stored on CPU memory");
		assert(a.dev()==n.dev());
		assert(n.len()==u.len());
		a.resize(n.len());
		for (int j=0; j<n.len(); j++) a(j) = n(j)*s(j) + u(j);
	}
	
 protected:
	uvec<Type> u;
	uvec<Type> s;
};



/*
 binary_conv
 binary labels converter to +1/-1.

 Usage:
 Y = { 2, 0, 2, 0, 0};
 binary_conv<>(2).convert(Y, y)
 result: y = { 1, -1, 1, -1, -1}
*/

template <typename Type=int>
class binary_conv 
{
 public:
	binary_conv(const Type& plusone_value) { plusone = plusone_value; }
	// convert labels in 'l' to the ivec 'cl'
	void convert(const uvec<Type>& l, uvec<Type>& cl) {
		assert(l.dev()==device::CPU && "binary_conv requires data stored on CPU memory");
		int n = l.len();
		cl.resize(n);
		for (int i=0; i<n; ++i) 
			cl(i) = (l(i)==plusone) ? static_cast<Type>(1) : static_cast<Type>(-1); 
	}

 private:
	Type plusone;
};


/*
 string_enc
 convert vector of std::strings to ordered integers.

 Usage:
 x = { "A", "B", "D", "A", "D" };
 string_enc<> enc;
 enc.fit(x);
 enc.encode(x, y);
 enc.decode(y, x2);
 result: 
  y  = { 1, 2, 3, 1, 3 }
  x2 = { "A", "B", "D", "A", "D" }
*/

template <typename Type=int>
class string_enc 
{
 public:
	string_enc( const Type& startfrom=static_cast<Type>(1), 
				bool encnans=false, const Type& nanval=static_cast<Type>(std::nan(""))) {
		start_from  = startfrom;
		encode_nans = encnans;
		nan_value   = nanval;
		err_idx     = -1;
	}
	
	// the index of an unknown label during encoding/decoding.
	// this happens if the label didn't present while fitting.
	int get_error_index() const { return err_idx; }
	
	// fit the encoder.
	// creates the map with the string labels and their ordered numerical value.
	void fit(const std::vector<std::string>& strl) {
		size_t n = strl.size();
		smap.clear();
		tmap.clear();
		Type cur = start_from;
		for (size_t i=0; i<n; ++i) {
			typename strmap::const_iterator it = smap.find(strl[i]);
			if (it==smap.end()) {
				if (strl[i].empty() && !encode_nans) {
					smap.insert({ strl[i], nan_value });
					tmap.insert({ nan_value, strl[i] });
					continue;
				}
				smap.insert({ strl[i], cur });
				tmap.insert({ cur, strl[i] });
				cur++;
			}
		}
	}

	// encode string labels in 'strl' to 'encoded' labels.
	// returns true in success (all labels encoded).
	// if an unknown label is found, sets the err_idx with its index and returns false
	bool encode(const std::vector<std::string>& strl, uvec<Type>& encoded) {
		assert(encoded.dev()==device::CPU && "string_enc requires data stored on CPU memory");
		int n = (int)strl.size();
		encoded.resize((int)n);
		err_idx = -1;
		for (int i=0; i<n; ++i) {
			typename strmap::const_iterator it = smap.find(strl[i]);
			if (it==smap.end()) {
				err_idx = i;
				return false;
			}
			encoded(i) = it->second;
		}
		return true;
	}

	// decode previously encoded labels 'encoded' to string labels 'strl'.
	// returns true in success (all labels decoded).
	// if an unknown label is found, sets the err_idx with its index and returns false
	bool decode(const uvec<Type>& encoded, std::vector<std::string>& strl) {
		int n = encoded.len();
		strl.resize(n);
		err_idx = -1;
		for (int i=0; i<n; ++i) {
			typename typemap::const_iterator it = tmap.find(encoded(i));
			if (it==tmap.end()) {
				err_idx = i;
				return false;
			}
			strl[i] = it->second;
		}
		return true;
	}
	
 private:
	typedef std::map<std::string, Type> strmap;
	typedef std::map<Type, std::string> typemap;
 
	Type    start_from;
	bool    encode_nans;
	Type    nan_value;
	int     err_idx;
	strmap  smap;
	typemap tmap;
};


/*
 onehot_enc
 onehot encoder of _integer_ categorical data.

 x = { 1, 
       2, 
       3, 
       3 };
 onehot_enc<> enc;
 enc.fit(x);
 enc.encode(x, X1hot);
 result: 
  X1hot = { 1,0,0,
            0,1,0
            0,0,1
            0,0,1 }
            
  With drop_first set to true:
  X1hot = { 0,0,
            1,0
            0,1
            0,1 }
  
  Usage
  ~~~~~
  X: dataset
  uvec<int> c3=X.get_vector_column(2);
  umat<int> C3_1hot;
  onehot_enc<int,float> enc;
  enc.fit_encode(x, C3_1hot);
  X.drop_column(2);
  X.hstack(X, C3);
*/


template <typename Type=int>
class onehot_enc 
{
 public:
	// handling unknown (faulty) samples in decoding
	enum {
		SkipFaulty,       // skip faulty samples and continue decoding
		DoNotSkipFaulty,  // stop decoding
	}; 

	onehot_enc(bool drop1st=false) {
		drop_first = drop1st;
		err_idx    = -1;
	}
	
	// the index of an unknown label during encoding/decoding.
	// this happens if the label didn't present while fitting.
	int get_error_index() const { return err_idx; }
	
	// fit the encoder.
	// creates a map with all unique labels in vector 'x'
	void fit(const uvec<Type>& x) {
		assert(x.dev()==device::CPU && "onehot_enc requires data stored on CPU memory");
		int n = x.len();
		categories.clear();
		for (int i=0; i<n; ++i) {
			typename catmap::iterator it = std::find(categories.begin(), categories.end(), x(i));
			if (it==categories.end()) categories.push_back(x(i));
		}
		std::sort(categories.begin(), categories.end());
	}

	// encode vector 'x' to matrix 'encoded' using onehot encoding.
	// returns true in success (all elements encoded).
	// if an unknown element is found, sets the err_idx with its index and returns false
	template <typename OType>
	bool encode(const uvec<Type>& x, umat<OType>& encoded) {
		assert(x.dev()==device::CPU && "onehot_enc requires data stored on CPU memory");
		assert(x.dev()==encoded.dev());
		int n = x.len();
		int ncols = (int)categories.size();
		if (ncols < 2) drop_first = false;
		if (drop_first) ncols--;
		encoded.resize(n, ncols);
		encoded.zero_active_device();
		err_idx = -1;
		for (int i=0; i<n; ++i) {
			typename catmap::iterator it = std::find(categories.begin(), categories.end(), x(i));
			if (it==categories.end()) {
				err_idx = i;
				return false;
			}
			int j = it - categories.begin();
			if (drop_first) {
				if (j==0) continue;
				j--;
			}
			encoded(i,j) = static_cast<OType>(1);
		}
		return true;
	}

	// fit and encode vector 'x' to matrix 'encoded' using onehot encoding.
	// see encode() for the returned value.
	template <typename OType>
	bool fit_encode(const uvec<Type>& x, umat<OType>& encoded) {
		fit(x);
		return encode(x, encoded);
	}
	
	// decode previously onehot encoded data 'encoded' to vector 'x'.
	// returns true in success (all data decoded).
	// if an unknown element is found, sets the err_idx with its index and returns false
	template <typename OType>
	bool decode(const umat<OType>& encoded, uvec<Type>& x, int onfaulty=DoNotSkipFaulty) {
		assert(x.dev()==device::CPU && "onehot_enc requires data stored on CPU memory");
		assert(x.dev()==encoded.dev());
		int n = encoded.ydim();
		int ncols = encoded.xdim();
		x.resize(n);
		x.zero_active_device();
		err_idx = -1;
		for (int i=0; i<n; ++i) {
			uv_ref<OType> r(encoded.row_offset(i).get_cmem(), device::CPU, ncols, ncols);
			int j = r.argmax();
			if (drop_first) {
				if (j==0 && r(j)==static_cast<OType>(0)) {
					x(i) = categories[0];
				} else {
					x(i) = categories[j+1];
				}
			} else {
				if (j==0 && r(j)==static_cast<OType>(0)) {
					err_idx = i;
					if (onfaulty==DoNotSkipFaulty) return false;
				} else {
					x(i) = categories[j];
				}
			}
		}
		return (err_idx==-1);
	}

 private:
	typedef std::vector<Type> catmap;

	bool    drop_first;
	int     err_idx;
	catmap  categories;
};


/*
 uniques
 returns a vector containing only the unique values from the input vector.
 
 Usage:
 vect v = { 3, 1, 1, 4, 5, 3 };
 vect u = uniques(v)
 Result: { 3, 1, 4, 5 }
*/

template <typename Type>
uvec<Type> uniques(const uvec<Type>& v)
{
	assert(v.dev()==device::CPU && "uniques requires data stored on CPU memory");	
	std::vector<Type> u;
	for (int i=0; i<v.len(); ++i) {
		typename std::vector<Type>::iterator it = std::find(u.begin(), u.end(), v(i));
		if (it==u.end()) u.push_back(v(i));
	}
	uvec<Type> out(u.size());
	out.set(u);
	return out;
}


};     // namespace umml

#endif // UMML_PREPROC_INCLUDED
