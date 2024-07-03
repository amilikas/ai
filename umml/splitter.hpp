#ifndef UMML_SPLITTER_INCLUDED
#define UMML_SPLITTER_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Dataset train-test set splitter.

 FILE:     splitter.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 
 Internal dependencies:
 umml rand
 STL vector
 STL string
 STL algorithm
 STL random
  
 Usage
 ~~~~~
 * train/test split
   splitter<> ds(Xdata, Ylabels)
   ds.shuffle(); // optional
   ds.split_train_test_sets(0.75);
   ds.get_splits(X_train, X_test, y_train, y_test);
   ...
 
 * cross validation splits
   for (int fold=0; fold < k_folds; ++fold) {
       ds.split_for_cross_validation(fold, k_folds);
       ds.get_splits(X_train, X_test, y_train, y_test);
       ...
   }
*/


#include "umat.hpp"


namespace umml {
	

template <typename TX=float, typename TY=int>
class splitter
{
 public:
	// multi-column samples, single labels (vector)
	splitter(const umat<TX>& __X, const uvec<TY>& __y): _X(__X), _y(__y), _Y(_Ydummy) {
		_init();
	}
	
	// multi-column samples and targets
	splitter(const umat<TX>& __X, const umat<TY>& __Y): _X(__X), _y(_ydummy), _Y(__Y) {
		_init();
	}
	
	// time-series or unlabeled data
	splitter(const umat<TX>& __X): _X(__X), _y(_ydummy), _Y(_Ydummy) {
		_init();
	}

	// returns the internal stored data and labels
	umat<TX>& data() { return _X; }
	uvec<TY>& labels() { return _y; }
	umat<TY>& targets() { return _Y; }
		
	void shuffle(rng32& __rng=global_rng()) {
		// suffle indeces
		std::shuffle(_idx.begin(), _idx.end(), __rng);
		_t1idx.clear();
		_t2idx.clear();
		_t1idx.reserve(_t1_end-_t1_start+1);
		_t2idx.reserve(_t2_end-_t2_start+1);
		for (int i=_t1_start; i<=_t1_end; ++i) _t1idx.push_back(_idx[i]);
		for (int i=_t2_start; i<=_t2_end; ++i) _t2idx.push_back(_idx[i]);
	}

	// splits data set to train/test sets.
	void split_train_test_sets(float __partitioning) {
		int n = _X.ydim();
		_t1_start = 0;
		_t1_end = (n * __partitioning) - 1;
		if (_t1_end < 0) _t1_end = 0;
		_t2_start = _t1_end+1;
		_t2_end = n-1;
		if (_t2_start > _t2_end) return;
		_t1idx.clear();
		_t2idx.clear();
		_t1idx.reserve(_t1_end-_t1_start+1);
		_t2idx.reserve(_t2_end-_t2_start+1);
		for (int i=_t1_start; i<=_t1_end; ++i) _t1idx.push_back(_idx[i]);
		for (int i=_t2_start; i<=_t2_end; ++i) _t2idx.push_back(_idx[i]);
	}

	void split_train_test_sets(size_t __train_samples, size_t __test_samples) {
		int n = _X.ydim();
		assert(n >= (int)(__train_samples+__test_samples));
		_t1_start = 0;
		_t1_end = (int)(__train_samples - 1);
		if (_t1_end < 0) _t1_end = 0;
		_t2_start = _t1_end+1;
		_t2_end = _t2_start+__test_samples-1;
		if (_t2_start > _t2_end) return;
		_t1idx.clear();
		_t2idx.clear();
		_t1idx.reserve(_t1_end-_t1_start+1);
		_t2idx.reserve(_t2_end-_t2_start+1);
		for (int i=_t1_start; i<=_t1_end; ++i) _t1idx.push_back(_idx[i]);
		for (int i=_t2_start; i<=_t2_end; ++i) _t2idx.push_back(_idx[i]);
	}

	// cross validation
	void split_for_cross_validation(int __fold, int __nfolds) {
		assert(__fold >= 0 && __fold < __nfolds);
		int pn, r1, r2, t1, t2, r3, r4;
		int n = _X.ydim();
		pn = (int)((double)n / __nfolds);
		t1 = __fold*pn;
		t2 = t1+pn-1;
		r1 = 0;
		r2 = t1-1;
		r3 = t2+1;
		r4 = n-1;
		//std::cout << "r1="<<r1<<" r2="<<r2<<" t1="<<t1<<" t2="<<t2<<" r3="<<r3<<" r4="<<r4<<"\n";
		
		// now build trnidx and tstidx
		_t1idx.clear();
		_t1idx.reserve(n-pn);
		for (int i=r1; i<=r2; ++i) _t1idx.push_back(_idx[i]);
		for (int i=r3; i<=r4; ++i) _t1idx.push_back(_idx[i]);
		_t2idx.clear();
		_t2idx.reserve(pn);
		for (int i=t1; i<=t2; ++i) _t2idx.push_back(_idx[i]);
	}

	// build train/test sets
	void get_splits(umat<TX>& __Xtrain, umat<TX>& __Xtest, uvec<TY>& __ytrain, uvec<TY>& __ytest) {
		_make_set(__Xtrain, _t1idx);
		_make_set(__Xtest, _t2idx);
		_make_labels(__ytrain, _t1idx);
		_make_labels(__ytest, _t2idx);
	}

	void get_splits(umat<TX>& __Xtrain, umat<TX>& __Xtest, umat<TY>& __Ytrain, umat<TY>& __Ytest) {
		_make_set(__Xtrain, _t1idx);
		_make_set(__Xtest, _t2idx);
		_make_targets(__Ytrain, _t1idx);
		_make_targets(__Ytest, _t2idx);
	}
	
 private:
	// private methods
	void _init() {
		int n = _X.ydim();
		_idx.resize((size_t)n);
		_t1idx.resize((size_t)n);
		_t2idx.resize((size_t)n);
		// use whole dataset as training set and test set (call split_train_test_sets to change this)
		for (int i=0; i<n; ++i) _idx[i] = _t1idx[i] = _t2idx[i] = i;
		_t1_start = _t2_start = 0;
		_t1_end = _t2_end = n - 1;
	}

	// returns a matrix with the training or test samples
	void _make_set(umat<TX>& __X, const std::vector<int>& __idx) { 
		int nrows = (int)__idx.size();
		int ncols = _X.xdim();
		__X.resize(nrows, ncols);
		for (int i=0; i<nrows; ++i)
		for (int j=0; j<ncols; ++j) __X(i,j) = _X(__idx[i],j);
	}
	
	// returns a vector with the training or test labels
	void _make_labels(uvec<TY>& __y, const std::vector<int>& __idx) { 
		int n = __idx.size();
		__y.resize(n);
		for (int i=0; i<n; ++i) __y(i) = _y(__idx[i]);
	}

	// returns a matrix with the training or test targets
	void _make_targets(umat<TY>& __Y, const std::vector<int>& __idx) { 
		int nrows = (int)__idx.size();
		int ncols = _Y.xdim();
		__Y.resize(nrows, ncols);
		for (int i=0; i<nrows; ++i)
		for (int j=0; j<ncols; ++j) __Y(i,j) = _Y(__idx[i],j);
	}
	
 private:
	const umat<TX>& _X;
	const uvec<TY>& _y;
	const umat<TY>& _Y;
	uvec<TY> _ydummy;
	umat<TY> _Ydummy;
	std::vector<int> _idx, _t1idx, _t2idx;
	int  _t1_start, _t1_end, _t2_start, _t2_end;
};


};     // namespace umml

#endif // UMML_SPLITTER_INCLUDED
