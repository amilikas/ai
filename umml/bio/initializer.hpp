#ifndef UMML_INITIALIZER_INCLUDED
#define UMML_INITIALIZER_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Genetic Algorithms: member initialization.

 FILE:     initializer.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2022-2024
 
 Namespace
 ~~~~~~~~~
 umml::initializer::values
 umml::initializer::subset
 umml::initializer::multiset
 umml::initializer::permut
 
 Notes
 ~~~~~
 noinit: no initialization (eg a model loaded from a disk file)
 random: initialization with random values
 
 Dependencies
 ~~~~~~~~~~~~
 umml rand
 STL string
 STL vector
  
 Usage example
 ~~~~~~~~~~~~~

 * Binary (0,1) encoding:
   initializer::values::random<bool> init;

 * Integer encoding in range {1..10}:
   initializer::values::random<int> init(1, 10);

 * Permutations of indeces in range {0..99}:
   initializer::permut::random<int,0> init(100);

 * Permutations of indeces in range {1..100}:
   initializer::permut::random<int,1> init(100);

 * Encoding of 3 sets of integers: 50 values in {0..99}, 5 values in {0..9} and 10 values in {0..19}:
   initializer::multiset::random<int,0> init({50,5,10}, {100,10,20});
*/


#include "../compiler.hpp"
#include "../rand.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <cstdarg>


namespace umml {
namespace initializer {


// no init
template <typename Type=int>
struct null {
	using Vals = std::vector<Type>;
	null() {}
	void apply(Vals& m) {}
	std::string info() const { return "null"; }
};



////////////////////////////////////////////////////////////
// Value encoding & Binary (bool) encoding
//

namespace values {


// uniform random numbers
template <typename Type>
struct random {
	using Vals = std::vector<Type>;
	Type minval, maxval;
	
	random(Type _min, Type _max): minval(_min), maxval(_max) {}	
	void apply(Vals& m) {
		uniform_random(&m[0], (int)m.size(), minval, maxval);
	}
	std::string info() const { return "random"; }
};

// specialization for binary encoding (Type=bool)
template <>
struct random<bool> {
	using Vals = std::vector<bool>;
	std::string name;
	
	random() {}	
	void apply(Vals& m) {
		for (size_t i=0; i<m.size(); ++i)
			m[i] = uniform_random_int<bool>(0, 1);
	}
	std::string info() const { return "random"; }
};

// gaussian random numbers
template <typename Type>
struct gaussian {
	using Vals = std::vector<Type>;
	Type mean, stdev;
	
	gaussian(Type _mean, Type _stdev): mean(_mean), stdev(_stdev) {}	
	void apply(Vals& m) {
		std::normal_distribution<Type> g(mean, stdev);
		for (size_t i=0; i<m.size(); ++i) m[i] = g(global_rng());
    }
	std::string info() const { return "gaussian"; }
};


}; // namespace values



////////////////////////////////////////////////////////////
// Subset
//

namespace subset {


/*
// subset random values
// requires the lengths of each set to be specified.
template <typename Type=int, int Base=0>
struct random {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "initializer::subset::random: Type must be integral.");
	#endif

	using Vals = std::vector<Type>;
	int n;
	bool sort;

	random(int __n, bool __sort=false): n(__n), sort(__sort) {}

	void apply(Vals& m) {
		std::vector<Type> set;
		build_shuffled_indeces(set, n);
		int len = uniform_random_int<int>(n/2-1, n-1);
		std::fill(m.begin(), m.end(), Type(0));
		for (int i=0; i<len; ++i) m[i] = set[i] + Base;
		if (sort) std::sort(m.begin(), m.end());
	}
	std::string info() const { return "random"; }
};
*/

}; // namespace subset



////////////////////////////////////////////////////////////
// Multiset - multiple subsets
//

namespace multiset {


// multi set random values
// requires the lengths of each set to be specified.
template <typename Type=int, int Base=0>
struct random {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "initializer::multiset::random: Type must be integral.");
	#endif

	using Vals = std::vector<Type>;
	std::vector<int> L, N;
	bool sort;

	random(const std::vector<int> __L, const std::vector<int> __N, bool __sort=false):
		L(__L), N(__N), sort(__sort) { assert(L.size()==N.size()); }

	void apply(Vals& m) {
		int nsets = N.size();
		int k = 0;
		for (int s=0; s<nsets; ++s) {
			std::vector<Type> set;
			build_shuffled_indeces(set, N[s]);
			for (int i=0; i<L[s]; ++i)
				m[k++] = set[i] + Base;
			if (sort) std::sort(m.begin()+k-L[s], m.begin()+k);
		}
	}
	std::string info() const { return "random"; }
};


}; // namespace multiset



////////////////////////////////////////////////////////////
// Permutations
//

namespace permut {


// single set random indeces. {0..N-1} for Base=0 or {1..N} for Base=1 indexing. 
template <typename Type=int, int Base=0>
struct random {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "initializer::permut::random: Type must be integral.");
	#endif

	using Vals = std::vector<Type>;
	int N;

	random(int __N): N(__N) {}
	void apply(Vals& m) {
		std::vector<Type> indeces;
		build_shuffled_indeces(indeces, N);
		for (size_t i=0; i<m.size(); ++i)
			m[i] = indeces[i]+Base;
	}
	std::string info() const { return "random"; }
};


}; // namespace permut


}; // namespace initializer
}; // namespace umml

#endif // UMML_INITIALIZER_INCLUDED
