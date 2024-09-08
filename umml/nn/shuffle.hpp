#ifndef UMML_SHUFFLE_INCLUDED
#define UMML_SHUFFLE_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 SGD shuffle methods.

 FILE:     shuffle.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: backpropagation, sgd
 
 Namespace
 ~~~~~~~~~
 umml::shuffle
 
 Methods
 ~~~~~~~
 * null: does not alter the order of training samples
 * deterministic: in each training session produces the same random set of training indeces
 * stochastic: in each training session produces a different random set of training indeces
  
 Dependencies
 ~~~~~~~~~~~~
 std::vector

 Internal dependencies:
 umml rand
  
 Usage example
 ~~~~~~~~~~~~~
 shuffle::stochastic sh;
 bp.train(net, loss, step, sh, Xtrain, Ytrain);

 shuffle::deterministic sh;
 bp.train(net, loss, step, sh, Xtrain, Ytrain);
  
 TODO
 ~~~~ 
*/


#include "../rand.hpp"


namespace umml {
namespace shuffle {


// does not alter the order of training samples
struct null {
	void seed() {}
	void shuffle(std::vector<int>& idcs, int n) {
		idcs.clear();
		idcs.reserve(n);
		for (int i=0; i<n; ++i) idcs.push_back(i);
	}
};

// uses a local RNG, in each training session produces the same random set of training indeces
struct deterministic {
	void seed() {
		_rng.seed(48,48+13);
	}
	void shuffle(std::vector<int>& idcs, int n) {
		build_shuffled_indeces(idcs, n, _rng);
	}
	rng32 _rng;
};

// uses the global RNG, in each training session produces a different random set of training indeces
struct stochastic {
	void seed() {}
	void shuffle(std::vector<int>& idcs, int n) {
		build_shuffled_indeces(idcs, n);
	}
};


};     // namespace shuffle
};     // namespace umml

#endif // UMML_SHUFFLE_INCLUDED
