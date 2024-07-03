#ifndef UMML_RAND_INCLUDED
#define UMML_RAND_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Random number generator

 FILE:     rand.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2023-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 STL random (for seeding with random_device)
  
 Functions
 ~~~~~~~~~
 umml_seed_rng: seeds the global RNG
 uniform_random_ints : random integers from the global RNG in the range [min..max]
 uniform_random_reals: random reals from the global RNG in the range [min..max]
 build_shuffled_indeces: create and shuffle an indexing array with n indeces.
 
 Notes
 ~~~~~ 
 [1] PCG, A Family of Better Random Number Generators
 https://www.pcg-random.org/

 uniform_int_distribution and uniform_real_distribution are replacements
 for std counterparts.
 
 Usage
 ~~~~~

 Global RNG (_global_rng):
 umml_seed_rng(48) for a specific seed  or  umml_seed_rng() for a random seed
 std::vector<float> v(100);
 uniform_random_reals(v.data(), v.size(), -5.0, 5.0);       // all elements to random numbers
 uniform_random_reals(v.data(), v.size(), -5.0, 5.0, 0.25); // 25% of elements to random numbers
 


 Localized RNG:
 rng32 local_rng;
 std::vector<float> v(10);
 local_rng.random_uniform_reals(v.data(), v.size(), -5.0, 5.0);

 Can be used like std::mt19937:
 uniform_real_distribution dist(min, max);
 for (size_t i=0; i<v.size(); ++i) v[i] = dist(rng);
 std::shuffle(v.begin(), v.end(), rng);
*/


#include <random>
#include <vector>
#include <algorithm>


// set __USE_PCG32__ to 1 to use PCG rng or to 0 to use MT19937 rng
#define __USE_PCG32__ 1


namespace umml {


#if __USE_PCG32__ == 1

/*
 PCG32 generator
 ~~~~~~~~~~~~~~~
*/

// uniform distribution for integer values
template <typename Type>
struct uniform_int_distribution {
	uniform_int_distribution(Type __a, Type __b): _a(__a), _b(__b) {};
	Type operator()(uint32_t r) { 
		return (Type)((int)_a) + r % (((int)_b) - ((int)_a) + 1);
	}
	Type _a, _b;
};

// uniform distribution for real values
template <typename Type>
struct uniform_real_distribution {
	uniform_real_distribution(Type __a, Type __b): _a(__a), _b(__b) {};
	Type operator()(uint32_t r) { 
		return _a + ((((Type)r) / (Type)UINT32_MAX) * (_b - _a));;
	}
	Type _a, _b;
};


// PCG-32
struct rng32 {
	typedef uint32_t result_type;
	uint64_t _state;  
	uint64_t _incr;

	rng32(): _state(0x853c49e6748fea9bULL), _incr(0xda3e39cb94b95bdbULL) {}
	rng32(uint64_t __seed, uint64_t __seq) { seed(__seed, __seq); }
	rng32(uint32_t __seed) { seed(__seed, __seed+13); }

	void seed(uint64_t __seed, uint64_t __seq) {
		_state = 0U;
		_incr  = (__seq << 1u) | 1u;
		generate();
		_state += __seed;
		generate();
	}

	void random_seed() {
		std::random_device rd;
		uint32_t rn = (uint32_t)rd();
		seed(rn, rn ^ static_cast<uint32_t>((intptr_t)&_incr));
	}

	// returns a random 32bit integer in the range [0..UINT32_MAX-1]
	result_type generate() {
		uint64_t oldstate = _state;
		_state = oldstate * 6364136223846793005ULL + _incr;
		uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = (uint32_t)(oldstate >> 59u);
		return (result_type)((xorshifted >> rot) | (xorshifted << ((-rot) & 31)));
	}

	// interface for std algorithms (shuffle etc)
	static constexpr result_type min() { return 0; }
	static constexpr result_type max() { return UINT32_MAX; }
	result_type operator()() { return generate(); }
	operator result_type() { return generate(); }

	// random integer numbers from a uniform distribution in [minval, maxval]
	template <typename Type>
	void uniform_random_ints(Type* buf, int n, Type minval, Type maxval, float ratio=1.0f) {
		uniform_int_distribution<Type> dist(minval, maxval);
		for (int i=0; i<n; ++i) {
			if (ratio < 1.0f) {
				if ((Type)generate()/max() < ratio) *buf = dist(*this);
				++buf;
			} else {
				*buf++ = dist(*this);
			}
		}
	}

	// random floating point numbers from a uniform distribution in [minval, maxval]
	template <typename Type>
	void uniform_random_reals(Type* buf, int n, Type minval, Type maxval, float ratio=1.0f) {
		uniform_real_distribution<Type> dist(minval, maxval);
		for (int i=0; i<n; ++i) {
			if (ratio < 1.0f) {
				if ((Type)generate()/max() < ratio) *buf = dist(*this);
				++buf;
			} else {
				*buf++ = dist(*this);
			}
		}
	}
};


#else

/*
 MT19937 generator
 ~~~~~~~~~~~~~~~~~
*/

// uniform distribution for integer values
template <typename Type>
struct uniform_int_distribution {
	uniform_int_distribution(Type __a, Type __b): dist(__a, __b) {};
	Type operator()(uint32_t r) { return dist(r); }
	std::uniform_int_distribution<Type> dist;
};

// uniform distribution for real values
template <typename Type>
struct uniform_real_distribution {
	uniform_real_distribution(Type __a, Type __b): dist(__a, __b) {};
	Type operator()(uint32_t r) { return dist(r); }
	std::uniform_real_distribution<Type> dist;
};

struct rng32 {
	std::mt19937 g;
	typedef std::mt19937::result_type result_type;
	
	rng32() { g = std::mt19937(48); }
	rng32(uint64_t __seed, uint64_t __seq) { g = std::mt19937((uint32_t)__seed); }
	rng32(uint32_t __seed) { g = std::mt19937(__seed); }

	void seed(uint64_t __seed, uint64_t __seq) {
		g = std::mt19937((uint32_t)__seed);
	}

	void random_seed() {
		std::random_device rd;
		g = std::mt19937(rd());
	}

	// returns a random 32bit integer in the range [0..UINT32_MAX-1]
	result_type generate() { return g(); }

	// interface for std algorithms (shuffle etc)
	static constexpr result_type min() { return std::mt19937::min(); }
	static constexpr result_type max() { return std::mt19937::max(); }
	result_type operator()() { return g(); }
	operator result_type() { return g(); }

	// random integer numbers from a uniform distribution in [minval, maxval]
	template <typename Type>
	void uniform_random_ints(Type* buf, int n, Type minval, Type maxval, float ratio=1.0f) {
		std::uniform_int_distribution<Type> dist(minval, maxval);
		for (int i=0; i<n; ++i) {
			if (ratio < 1.0f) {
				if ((Type)generate()/max() < ratio) *buf = dist(*this);
				++buf;
			} else {
				*buf++ = dist(*this);
			}
		}
	}

	// random floating point numbers from a uniform distribution in [minval, maxval]
	template <typename Type>
	void uniform_random_reals(Type* buf, int n, Type minval, Type maxval, float ratio=1.0f) {
		std::uniform_real_distribution<Type> dist(minval, maxval);
		for (int i=0; i<n; ++i) {
			if (ratio < 1.0f) {
				if ((Type)generate()/max() < ratio) *buf = dist(*this);
				++buf;
			} else {
				*buf++ = dist(*this);
			}
		}
	}
};

#endif // __USE_PCG32__


// global uniform RNG
static rng32 _global_rng;


// access global RNG
rng32& global_rng() { return _global_rng; }


// seeds the RNG
void umml_seed_rng(uint32_t seed=(uint32_t)-1)
{
	if (seed==(uint32_t)-1) {
		_global_rng.random_seed();
	} else {
		_global_rng.seed(seed, seed+13);
	}
}


// random integer numbers from a uniform distribution using the global RNG
template <typename Type>
void uniform_random_ints(Type* buf, int n, Type minval, Type maxval, float ratio=1.0f) 
{ 
	_global_rng.uniform_random_ints(buf, n, minval, maxval, ratio);
}

// random real numbers from a uniform distribution using the global RNG
template <typename Type>
void uniform_random_reals(Type* buf, int n, Type minval, Type maxval, float ratio=1.0f) 
{
	_global_rng.uniform_random_reals(buf, n, minval, maxval, ratio);
}

template <typename Type>
void uniform_random(Type* buf, int n, Type minval, Type maxval, float ratio=1.0f) 
{
	_global_rng.uniform_random_ints(buf, n, minval, maxval, ratio);
}

// random reals if Type=float
template <>
void uniform_random<float>(float* buf, int n, float minval, float maxval, float ratio) 
{
	_global_rng.uniform_random_reals(buf, n, minval, maxval, ratio);
}

// random reals if Type=double
template <>
void uniform_random<double>(double* buf, int n, double minval, double maxval, float ratio) 
{
	_global_rng.uniform_random_reals(buf, n, minval, maxval, ratio);
}

template <typename Type>
Type uniform_random_int(Type minval, Type maxval) 
{
	Type result;
	_global_rng.uniform_random_ints(&result, 1, minval, maxval, 1.0f);
	return result;
}

template <typename Type>
Type uniform_random_real(Type minval, Type maxval) 
{
	Type result;
	_global_rng.uniform_random_reals(&result, 1, minval, maxval, 1.0f);
	return result;
}


// build_shuffled_indeces: Create and shuffle an indexing array using the RNG 'rg'
template <typename Type=int>
void build_shuffled_indeces(std::vector<Type>& idcs, int n, rng32& rg) 
{
	idcs.clear();
	idcs.reserve((size_t)n);
	for (int i=0; i<n; ++i) idcs.push_back((Type)i);
	std::shuffle(idcs.begin(), idcs.end(), rg);
}

// build_shuffled_indeces: Create and shuffle an indexing array using the global RNG
template <typename Type=int>
void build_shuffled_indeces(std::vector<Type>& idcs, int n) 
{
	build_shuffled_indeces(idcs, n, global_rng());
}


// shuffles between groups eg {1,2,3,4, 5,6,7,8, 9,10,11,12} -> {5,10,7,8, 9,2,11,12, 1,6,3,4}
template <typename Type=int>
void shuffle_between_groups(std::vector<Type>& v, int group_size, rng32& rg) 
{
	int ngroups = v.size() / group_size;
	for (int i=0; i<ngroups-1; ++i) {
		for (int j=0; j<group_size; ++j) {
			int idx1 = i*group_size + j;
			int rndi;
			rg.uniform_random_ints(&rndi, 1, i+1, ngroups-1);
			int idx2 = rndi*group_size + j;
			std::swap(v[idx1], v[idx2]);
		}
	}
}

// shuffles between groups using global RNG
template <typename Type=int>
void shuffle_between_groups(std::vector<Type>& v, int group_size) 
{
	shuffle_between_groups(v, group_size, global_rng()); 
}


};     // namespace umml

#endif // UMML_RAND_INCLUDED
