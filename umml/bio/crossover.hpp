#ifndef UMML_CROSSOVER_INCLUDED
#define UMML_CROSSOVER_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Genetic Algorithms: crossovers

 FILE:     crossover.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2022-2024
 
 Namespace
 ~~~~~~~~~
 umml::crossover::values
 umml::crossover::subset
 umml::crossover::permut

 Notes
 ~~~~~
 binary encoding = bool value encoding (the default)
 value (binary) encoding and permutations crossovers between parents p1 and p2 to produce 
 children c1 and c2.:

 * onepoint crossover:
   xxxxxxxxxx      xxxyyyyyyy
	  ^        -->  
   yyyyyyyyyy      yyyxxxxxxx
  
 * twopoint crossover:
   xxxxxxxxxx      xxxyyyyxxx
	  ^   ^    -->  
   yyyyyyyyyy      yyyxxxxyyy
   
 * uniform crossover (value encoding only):
   xxxxxxxxxx      yyxxyxyxyx
	^^  ^ ^ ^  -->  
   yyyyyyyyyy      xxyyxyxyxy

 * rarw crossover (integer encoding) set of values in {1..N} and chromosome lenght is n (n <= N)
   as described in Radcliffe, N. J., & George, F. A. (1993, July). A Study in Set Recombination.

 Dependencies
 ~~~~~~~~~~~~
 umml rand
 STL string
 STL vector
 STL algorithm
  
 Usage example
 ~~~~~~~~~~~~~

*/


#include "../compiler.hpp"
#include "../rand.hpp"
#include <string>
#include <vector>
#include <algorithm>


namespace umml {
namespace crossover {


template <typename Type> 
struct null {
	using Vals = std::vector<Type>;
	null() {}
	void apply(const Vals& p1, const Vals& p2, Vals& c1) { c1=p1; }
	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) { c1=p1; c2=p2; }
	std::string info() const { return "null"; }
};


////////////////////////////////////////////////////////////
// Value encoding & Binary encoding (Type=bool)
//

namespace values {


// onepoint crossover
template <typename Type=bool> 
struct onepoint {
	using Vals = std::vector<Type>;
	int ofs;

	onepoint(int edge_offset=1): ofs(edge_offset) {}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		int n = (int)p1.size();
		int pt = uniform_random_int<int>(0+ofs, n-1-ofs);
		for (int i=0; i<pt; ++i) c1[i] = p1[i];
		for (int i=pt; i<n; ++i) c1[i] = p2[i];
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		int n = (int)p1.size();
		int pt = uniform_random_int<int>(0+ofs, n-1-ofs);
		// child 1
		for (int i=0; i<pt; ++i) c1[i] = p1[i];
		for (int i=pt; i<n; ++i) c1[i] = p2[i];
		// child 2
		for (int i=0; i<pt; ++i) c2[i] = p2[i];
		for (int i=pt; i<n; ++i) c2[i] = p1[i];
	}

	std::string info() const { return "onepoint"; }
};


// twopoint crossover
template <typename Type=bool> 
struct twopoint {
	using Vals = std::vector<Type>;
	int ofs;

	twopoint(int edge_offset=1): ofs(edge_offset) {}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		int pt[2];
		int n = (int)p1.size();
		uniform_random_ints(pt, 2, 0+ofs, n-1-ofs);
		if (pt[0] > pt[1]) std::swap(pt[0], pt[1]);
		for (int i=0; i<pt[0]; ++i) c1[i] = p1[i];
		for (int i=pt[0]; i<pt[1]; ++i) c1[i] = p2[i];
		for (int i=pt[1]; i<n; ++i) c1[i] = p1[i];
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		int pt[2];
		int n = (int)p1.size();
		uniform_random_ints(pt, 2, 0+ofs, n-1-ofs);
		if (pt[0] > pt[1]) std::swap(pt[0], pt[1]);
		// child 1
		for (int i=0; i<pt[0]; ++i) c1[i] = p1[i];
		for (int i=pt[0]; i<pt[1]; ++i) c1[i] = p2[i];
		for (int i=pt[1]; i<n; ++i) c1[i] = p1[i];
		// child 2
		for (int i=0; i<pt[0]; ++i) c2[i] = p2[i];
		for (int i=pt[0]; i<pt[1]; ++i) c2[i] = p1[i];
		for (int i=pt[1]; i<n; ++i) c2[i] = p2[i];
	}

	std::string info() const { return "twopoint"; }
};


// uniform crossover
template <typename Type=bool> 
struct uniform {
	using Vals = std::vector<Type>;
	double mix;

	uniform(double mixing=0.5): mix(mixing) {}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		int n = (int)p1.size();
		for (int i=0; i<n; ++i) {
			double coin = uniform_random_real<double>(0.0, 1.0);
			c1[i] = (coin < mix) ? p1[i] : p2[i];
		}
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		int n = (int)p1.size();
		for (int i=0; i<n; ++i) {
			double coin = uniform_random_real<double>(0.0, 1.0);
			if (coin < mix) { 
				c1[i] = p1[i];
				c2[i] = p2[i];
			} else {
				c1[i] = p2[i];
				c2[i] = p1[i];
			}
		}
	}

	std::string info() const { return "uniform"; }
};


}; // namespace values



////////////////////////////////////////////////////////////
// Subset
//

namespace subset {

// rar(w) Random Assorting Recombination
// Works with base=1 indexing
template <typename Type=int>
struct rarw {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "crossover::permut::rarw: Type must be integral.");
	#endif

	using Vals = std::vector<Type>;

	// child genome length
	enum {
		RARw_Any,
		RARw_ParentSize,
	};

	Type N;
	int w;

	rarw(Type __N, int __w=3): N(__N), w(__w) {}

	void produce_child(const Vals& p1, const Vals& p2, Vals& child, int child_size) {
		child.clear();
		std::vector<Type> pool;
		std::vector<Type> sel;
		// add w copies of common elements to the pool list
		for (Type val : p1) {
			if (val != 0 && std::find(p2.begin(), p2.end(), val) != p2.end())
				for (int i=0; i<w; ++i) pool.push_back(val);
		}
		// add w copies of barred elements to the pool list (-val denotes a barred item)
		for (Type barred=1; barred<=N; ++barred) {
			if (std::find(p1.begin(), p1.end(), barred) == p1.end() &&
				std::find(p2.begin(), p2.end(), barred) == p2.end()) {
				for (int i=0; i<w; ++i) pool.push_back(-barred);
			}
		}
		// add one copy of unique parent1 elements and barred counterpart
		for (Type val : p1) {
			if (val != 0 && std::find(p2.begin(), p2.end(), val) == p2.end()) {
				pool.push_back(val);
				pool.push_back(-val);
			}
		}
		// add one copy of unique parent2 elements and barred counterpart
		for (Type val : p2) {
			if (val != 0 && std::find(p1.begin(), p1.end(), val) == p1.end()) {
				pool.push_back(val);
				pool.push_back(-val);
			}
		}
		// select values
		for (int i=1; i<=(int)N; ++i) {
			if (pool.empty()) break;
			if (sel.size()==(size_t)N) break;
			int idx = uniform_random_int<int>(0, (int)pool.size()-1);
			Type val = pool[idx];
			std::remove(pool.begin(), pool.end(), val);
			if (std::find(sel.begin(), sel.end(), val) == sel.end() &&  
				std::find(sel.begin(), sel.end(),-val) == sel.end())
				sel.push_back(val);
		}
		// produce child
		size_t max_size = std::max(p1.size(), p2.size());
		for (size_t i=0; i<sel.size() && child.size()<max_size; ++i) 
			if (sel[i] > 0) child.push_back(sel[i]);
		while ((child_size != RARw_Any && child.size() < max_size) || (child.size() == 0)) {
			Type val = uniform_random_int<Type>(1, N);
			if (std::find(child.begin(), child.end(), val) == child.end()) child.push_back(val);
		}
		// sort child
		std::sort(child.begin(), child.end());
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		constexpr size_t n = 5;
		std::vector<Vals> children(n);
		for (size_t i=0; i<n; ++i) produce_child(p1, p2, children[i], RARw_ParentSize);
		c1 = children[uniform_random_int<int>(0,n-1)];
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		constexpr size_t n = 5;
		std::vector<Vals> children_any(n), children_psize(n);
		for (size_t i=0; i<n; ++i) {
			produce_child(p1, p2, children_any[i], RARw_Any);
			produce_child(p1, p2, children_psize[i], RARw_ParentSize);
		}
		c1 = children_any[uniform_random_int<int>(0,n-1)];
		c2 = children_psize[uniform_random_int<int>(0,n-1)];
	}

	std::string info() const { return "rar(w)"; }
};


template <>
struct rarw<bool> {
	using Vals = std::vector<bool>;
	rarw(int, int) {
		std::cerr << "crossover::values::rarw cannot be used with binary encoding (bool).\n";
		std::abort(); 
	}
	void apply(const Vals& p1, const Vals& p2, Vals& child) {}
	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {}
	std::string info() const { return "rar(w)"; }
};


}; // namespace subset



////////////////////////////////////////////////////////////
// Multiset
//

namespace multiset {

template <typename Type=int>
struct onepoint {
	using Vals = std::vector<Type>;
	std::vector<int> L;
	bool sort;
	int ofs;
	
	onepoint(const std::vector<int> __L, bool __sort=false, int edge_offset=1): 
		L(__L), sort(__sort), ofs(edge_offset) {}

	void produce_child(const Vals& p1, const Vals& p2, Vals& child, int f, int l, int pt) {
		int i;
		for (i=f; i<pt; ++i) child[i] = p1[i];
		typename Vals::iterator end = child.begin()+pt;
		for (int j=f; j<=l && i<=l; ++j)
			if (std::find(child.begin()+f, end, p2[j]) == end) child[i++] = p2[j];
		// sort child
		if (sort) std::sort(child.begin()+f, child.begin()+l+1);
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		int f = 0;
		for (int s=0; s<(int)L.size(); ++s) {
			int l = f + L[s] - 1;
			int pt = uniform_random_int<int>(f+ofs, l-ofs);
			produce_child(p1, p2, c1, f, l, pt);
			f += L[s];
		}
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		int f = 0;
		for (int s=0; s<(int)L.size(); ++s) {
			typename Vals::iterator end;
			int l = f + L[s] - 1;
			int pt = uniform_random_int<int>(f+ofs, l-ofs);
			produce_child(p1, p2, c1, f, l, pt);
			produce_child(p2, p1, c2, f, l, pt);
			f += L[s];
		}
	}

	std::string info() const { return "onepoint"; }
};

// uniform crossover
template <typename Type=int> 
struct uniform {
	using Vals = std::vector<Type>;
	std::vector<int> L;
	bool sort;
	double mix;

	uniform(const std::vector<int> __L, bool __sort=false, double __mix=0.5): L(__L), sort(__sort), mix(__mix) {}

	void produce_child(const Vals& p1, const Vals& p2, Vals& child, int f, int l) {
		int k = f;
		// create child
		for (int i=f; i<=l; ++i) {
			double coin = uniform_random_real<double>(0.0, 1.0);
			Type val = (coin < mix) ? p1[i] : p2[i];
			bool found = false;
			for (int j=f; j<k && !found; ++j) if (child[j]==val) found = true;
			if (!found) child[k++] = val;
		}
		// complete child
		while (k<=l) {
			for (int i=f; i<=l && k<=l; ++i) {
				bool found = false;
				Type val = p1[i];
				for (int j=f; j<k && !found; ++j) if (child[j]==val) found = true;
				if (!found) {
					child[k++] = val;
				} else {
					found = false;
					val = p2[i];
					for (int j=f; j<k && !found; ++j) if (child[j]==val) found = true;
					if (!found) {
						child[k++] = val;
					}
				}
			}
		}
		// sort child
		if (sort) std::sort(child.begin()+f, child.begin()+l+1);
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		int f = 0;
		for (int s=0; s<(int)L.size(); ++s) {
			int l = f + L[s] - 1;
			produce_child(p1, p2, c1, f, l);
			f += L[s];
		}
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		int f = 0;
		for (int s=0; s<(int)L.size(); ++s) {
			typename Vals::iterator end;
			int l = f + L[s] - 1;
			produce_child(p1, p2, c1, f, l);
			produce_child(p2, p1, c2, f, l);
			f += L[s];
		}
	}

	std::string info() const { return "uniform"; }
};


}; // namespace multiset



////////////////////////////////////////////////////////////
// Permutations
//

namespace permut {


template <typename Type=int>
struct onepoint {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "crossover::permut::onepoint: Type must be integral.");
	#endif
	using Vals = std::vector<Type>;
	int ofs;

	onepoint(int edge_offset=1): ofs(edge_offset) {}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		int i;
		int n = (int)p1.size();
		int pt = uniform_random_int<int>(0+ofs, n-1-ofs);
		for (i=0; i<pt; ++i) c1[i] = p1[i];
		typename Vals::iterator last = c1.begin()+pt;
		for (int j=0; j<n && i<n; ++j)
			if (std::find(c1.begin(), last, p2[j]) == last) c1[i++] = p2[j];
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		int i;
		int n = (int)p1.size();
		int pt = uniform_random_int<int>(0+ofs, n-1-ofs);
		// child 1
		for (i=0; i<pt; ++i) c1[i] = p1[i];
		typename Vals::iterator last = c1.begin()+pt;
		for (int j=0; j<n && i<n; ++j)
			if (std::find(c1.begin(), last, p2[j]) == last) c1[i++] = p2[j];
		// child 2
		for (i=0; i<pt; ++i) c2[i] = p2[i];
		last = c2.begin()+pt;
		for (int j=0; j<n && i<n; ++j)
			if (std::find(c2.begin(), last, p1[j]) == last) c2[i++] = p1[j];
	}

	std::string info() const { return "onepoint"; }
};


template <typename Type>
struct twopoint {
	#ifndef MML_MSVC
	static_assert(std::is_integral<Type>(), "crossover::permut::twopoint: Type must be integral.");
	#endif
	using Vals = std::vector<Type>;
	int ofs;

	twopoint(int edge_offset=1): ofs(edge_offset) {}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		int i, pt[2];
		int n = (int)p1.size();
		uniform_random_ints(pt, 2, 0+ofs, n-1-ofs);
		if (pt[0] > pt[1]) std::swap(pt[0], pt[1]);
		for (i=pt[0]; i<pt[1]; ++i) c1[i] = p1[i];
		typename Vals::iterator first = c1.begin()+pt[0];
		typename Vals::iterator last = c1.begin()+pt[1];
		i = 0;
		for (int j=0; j<n; ++j) {
			if (std::find(first, last, p2[j]) == last) {
				if (i >= pt[0] && i < pt[1]) i = pt[1];
				c1[i++] = p2[j];
			}
		}
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		int i, pt[2];
		int n = (int)p1.size();
		uniform_random_ints(pt, 2, 0+ofs, n-1-ofs);
		if (pt[0] > pt[1]) std::swap(pt[0], pt[1]);
		// child 1
		for (i=pt[0]; i<pt[1]; ++i) c1[i] = p1[i];
		typename Vals::iterator first = c1.begin()+pt[0];
		typename Vals::iterator last = c1.begin()+pt[1];
		i = 0;
		for (int j=0; j<n; ++j) {
			if (std::find(first, last, p2[j]) == last) {
				if (i >= pt[0] && i < pt[1]) i = pt[1];
				c1[i++] = p2[j];
			}
		}
		// child 2
		for (i=pt[0]; i<pt[1]; ++i) c2[i] = p2[i];
		first = c2.begin()+pt[0];
		last = c2.begin()+pt[1];
		i = 0;
		for (int j=0; j<n; ++j) {
			if (std::find(first, last, p1[j]) == last) {
				if (i >= pt[0] && i < pt[1]) i = pt[1];
				c2[i++] = p1[j];
			}
		}
	}

	std::string info() const { return "twopoint"; }
};


}; // namespace permut


}; // namespace crossover
}; // namespace umml

#endif // UMML_CROSSOVER_INCLUDED
