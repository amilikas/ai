#ifndef UMML_MUTATION_INCLUDED
#define UMML_MUTATION_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Genetic Algorithms: mutation.

 FILE:     mutation.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2022-2024
 
 Namespace
 ~~~~~~~~~
 umml::mutation::values
 umml::mutation::subset
 umml::mutation::permut
 
 Notes
 ~~~~~
 noinit: no initialization (eg a model loaded from a disk file)
 random: initialization with random values
 
 Dependencies
 ~~~~~~~~~~~~
 umml algo
 umml rand
 STL string
 STL vector
 STL algorithm
  
 Usage example
 ~~~~~~~~~~~~~
 mutation::values::flip<> mut(1.0, 0.5);
 mutation::permut::swap<int> mut(1.0, 0.5);
 mutation::subset::replace<int> mut({lengths}, {maxvals}, 1.0, 0.5);
 mut.apply(m); 
*/


#include "../algo.hpp"
#include <cstdlib>


namespace umml {
namespace mutation {


// null (no mutation)
template <typename Type=bool>
struct null {
	using Vals = std::vector<Type>;
	void apply(Vals& m) {}
	std::string info() const { return "null"; }
};


////////////////////////////////////////////////////////////
// Value encoding & Binary encoding (Type=bool)
//

namespace values {


// flip only applies to binary encoding (bool)
template <typename Type=bool>
struct flip {
	using Vals = std::vector<Type>;
	flip(double, double) { 
		std::cerr << "mutation::flip can only be used with binary encoding (bool).\n";
		std::abort(); 
	}
	void apply(Vals&) {}
	std::string info() const { return "flip"; }
};

template <>
struct flip<bool> {
	using Vals = std::vector<bool>;
	double prob, perc;
	flip(double prb=0.1, double prc=0.01): prob(prb), perc(prc) {}
	void apply(Vals& m) {
		if (uniform_random_real<double>(0.0, 1.0) < prob) {
			int n = (int)m.size();
			int nm = std::max(1, (int)(n*perc));
			std::vector<int> idcs;
			build_shuffled_indeces(idcs, n);
			for (int i=0; i<nm; ++i) m[idcs[i]] = (m[idcs[i]]) ? 0 : 1;
		}
	}
	std::string info() const { return "flip"; }
};

template <typename Type>
struct add {
	using Vals = std::vector<Type>;
	double prob, perc;
	Type val;
	add(double prb=0.1, double prc=0.1, Type v=Type(1)): prob(prb), perc(prc), val(v) {}
	void apply(Vals& m) {
		if (uniform_random_real<double>(0.0, 1.0) < prob) {
			int n = (int)m.size();
			int nm = std::max(1, (int)(n*perc));
			std::vector<int> idcs;
			build_shuffled_indeces(idcs, n);
			for (int i=0; i<nm; ++i) m[idcs[i]] += val;
		}
	}
	std::string info() const { return "add"; }
};

template <typename Type>
struct replace {
	using Vals = std::vector<Type>;
	Type minval, maxval;
	double prob, perc;
	replace(Type minv, Type maxv, double prb=0.1, double prc=0.1): 
		minval(minv), maxval(maxv), prob(prb), perc(prc) {}
	void apply(Vals& m) {
		if (uniform_random_real<double>(0.0, 1.0) < prob) {
			int n = (int)m.size();
			int nm = std::max(1, (int)(n*perc));
			std::vector<int> idcs;
			build_shuffled_indeces(idcs, n);
			for (int i=0; i<nm; ++i) {
				uniform_random(&m[idcs[i]], 1, minval, maxval);
			}
		}
	}
	std::string info() const { return "replace"; }
};

template <typename Type>
struct gaussian {
	using Vals = std::vector<Type>;
	Type minval, maxval;
	double prob;
	Type mean, stdev;
	gaussian(double prb=0.1, Type m=Type(0), Type s=Type(1)): 
		minval(std::numeric_limits<Type>::min()), maxval(std::numeric_limits<Type>::max()), prob(prb), mean(m), stdev(s) {}
	gaussian(Type minv, Type maxv, double prb=0.1, Type m=Type(0), Type s=Type(1)): 
		minval(minv), maxval(maxv), prob(prb), mean(m), stdev(s) {}
	void apply(Vals& m) {
		std::normal_distribution<Type> normal_dist(mean, stdev);		
		for (Type& v : m) {
			if (uniform_random_real<double>(0.0, 1.0) < prob) {
				v += normal_dist(umml::global_rng());
				if (v < minval) v = minval;
				if (v > maxval) v = maxval;
			}

		}
	}
	std::string info() const { return "gaussian"; }
};


}; // namespace values



////////////////////////////////////////////////////////////
// Multiset
//

namespace multiset {


template <typename Type=int, int Base=0>
struct replace {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "mutation::multiset::replace: Type must be integral.");
	#endif

	using Vals = std::vector<Type>;
	std::vector<int> L, N;
	bool sort;
	double prob, perc;
	replace(const std::vector<int> __L, const std::vector<int> __N, bool __sort=false,
			double __prob=0.1, double __perc=0.1): L(__L), N(__N), sort(__sort), prob(__prob), perc(__perc) {}
	void apply(Vals& m) {
		int f = 0;
		for (int s=0; s<(int)L.size(); ++s) {
			if (uniform_random_real<double>(0.0, 1.0) < prob) {
				int nm = std::max(1, (int)(L[s]*perc));
				std::vector<int> idcs;
				build_shuffled_indeces(idcs, L[s]);
				int l = f + L[s] - 1;
				typename Vals::iterator end = m.begin()+l+1;
				for (int i=0; i<nm; ++i) {
					// at most 5 tries to find a legal value for m[idcs[i]]
					for (int t=0; t<5; ++t) {						
						Type val = uniform_random_int<Type>(0+Base, N[s]-1+Base);
						if (std::find(m.begin()+f, end, val) == end) {
							m[f+idcs[i]] = val;
							break;
						}
					}
				}
			}
			if (sort) std::sort(m.begin()+f, m.begin()+f+L[s]);
			f += L[s];
		}
	}
	std::string info() const { return "replace"; }
};

template <typename Type=int>
struct swap {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "mutation::multiset::replace: Type must be integral.");
	#endif

	using Vals = std::vector<Type>;
	std::vector<int> L;
	bool sort;
	double prob, perc;
	
	swap(const std::vector<int>& __L, bool __sort=false, double prb=0.1, double prc=0.1): 
		L(__L), sort(__sort), prob(prb), perc(prc) {}
	void apply(Vals& m) {
		int f = 0;
		for (int s=0; s<(int)L.size(); ++s) {
			if (uniform_random_real<double>(0.0, 1.0) < prob) {
				int nm = std::max(1, (int)(L[s]*perc));
				std::vector<int> i1, i2;
				build_shuffled_indeces(i1, L[s]);
				build_shuffled_indeces(i2, L[s]);
				for (int i=0; i<nm; ++i)
					std::swap(m[f+i1[i]], m[f+i2[i]]);
			}
			if (sort) std::sort(m.begin()+f, m.begin()+f+L[s]);
			f += L[s];
		}
	}
	std::string info() const { return "swap"; }
};


}; // namespace multiset



////////////////////////////////////////////////////////////
// Permutations
//

namespace permut {


template <typename Type=int>
struct swap {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "mutation::permut::swap: Type must be integral.");
	#endif
	using Vals = std::vector<Type>;
	double prob, perc;
	swap(double prb=0.1, double prc=0.1): prob(prb), perc(prc) {}
	void apply(Vals& m) {
		if (uniform_random_real<double>(0.0, 1.0) < prob) {
			int n = (int)m.size();
			int nm = std::max(1, (int)(n*perc));
			std::vector<int> i1, i2;
			build_shuffled_indeces(i1, n);
			build_shuffled_indeces(i2, n);
			for (int i=0; i<nm; ++i) std::swap(m[i1[i]], m[i2[i]]);
		}
	}
	std::string info() const { return "swap"; }
};


}; // namespace permut


}; // namespace initializer
}; // namespace mml

#endif // UMML_MUTATION_INCLUDED
