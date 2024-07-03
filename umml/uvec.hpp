#ifndef UMML_UVEC_INCLUDED
#define UMML_UVEC_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 Vector class for unified memory (heterogeneous computing), CPU or GPU

 FILE:     uvec.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2023-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 STL string (shape, format)
 
  
 Notes
 ~~~~~
 This is not a replacement for std::vector.


 Examples
 ~~~~~~~~

 * cpu v[10], gpu u[10], gpu y[10]
   uvec<> v(10), u(10, device::GPU), y(10, device::GPU); 

 * v=1, u=2
   v.set(1);
   u.set(2);

 * upload v to GPU
   v.to_gpu();

 * y = v + u
   y.add(v,u);

 * y = 2v - 3u
   y.add(v,u,2,-3);

 * y += 5u
   y.plus(u,5);
*/

#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>

#include "compiler.hpp"
#include "types.hpp"
#include "dev.hpp"
#include "func.hpp"
#include "utils.hpp"
#include "algo.hpp"
#include "rand.hpp"
#include "kernels_cpu.hpp"
#ifdef __USE_CUDA__
#include "cuda.hpp"
#endif
#ifdef __USE_OPENCL__
#include "ocl.hpp"
#endif


namespace umml {


template <typename Type>
class uvec;

template <typename Type>
class uv_ref;

template <typename Type>
class umat;

template <typename Type>
class um_ref;

template <typename Type>
class ucub;

template <typename Type>
class uc_ref;



////////////////////////////////////////////////////////////////////////////////////
// uvec_base class
//
// Does not do any allocations
// Methods that must be implemented in derived uv_ref and uvec:
// - force_device
// - force_padding
// - host_alloc
// - device_alloc
// - host_free
// - device_free
// - to_gpu
// - to_cpu
// - resize


// IMPLEMENT
// norm
// normalize





template <typename Type>
class uvec_base {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::uvec_base<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;

 protected:
	Type*  _mem;
	int    _nel;
	int    _xsize;
	int    _size;
	device _dev;
	bool   _padgpu;
	#ifdef __USE_CUDA__
	Type*  _dmem;
	#endif
	#ifdef __USE_OPENCL__
	cl::Buffer _dmem;
	#endif

 public:
	uvec_base() {
		_mem    = nullptr;
		_nel    = _xsize = _size = 0;
		_dev    = device::CPU;
		_padgpu = true;
		#ifdef __USE_CUDA__
		_dmem   = nullptr;
		#endif
		#ifdef __USE_OPENCL__
		_dmem   = NullBuffer;
		#endif
	}

	virtual ~uvec_base() {}

	virtual void force_device(device __dev) = 0;
	virtual void force_padding(bool __padgpu) = 0;

	// properties
	bool    empty() const { return _nel==0; }
	int     len() const { return _nel; }
	int     size() const { return _size; }
	dims4   dims() const { return { _nel,1,1,1 }; }
	int     xdim() const { return _nel; }
	int     xsize() const { return _xsize; }
	device  dev() const { return _dev; }
	Type*   mem() { return _mem; }
	const   Type* cmem() const { return (const Type*)_mem; }

	// to reference it directly as a matrix with one row or a cube with one slice and one row
	int     ydim() const { return 1; }
	int     zdim() const { return 1; }
	int     ysize() const { return 1; }
	int     zsize() const { return xsize(); }
	//int     zstride() const { return padx(); }

	#ifdef __USE_CUDA__
	Type*   dmem() { return _dmem; }
	const   Type* cdmem() const { return (const Type*)_dmem; }
	#endif

	#ifdef __USE_OPENCL__
	cl::Buffer& dmem() { return _dmem; }
	const cl::Buffer& cdmem() const { return _dmem; }
	#endif

	umem active_mem() {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		if (_dev==device::GPU) { return umem(_dmem); }
		#endif
		return umem(_mem);
	}
	const umem active_mem() const {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		if (_dev==device::GPU) { return umem(_dmem); }
		#endif
		return umem(_mem);
	}

	virtual void host_alloc() = 0;

	#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
	virtual void device_alloc() = 0;
	#endif

	virtual void host_free() = 0;

	#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
	virtual void device_free() = 0;
	#endif

	// CPU -> GPU
	virtual void to_gpu() = 0;

	// GPU -> CPU
	virtual void to_cpu() = 0;
	
	// to CPU or GPU
	void to_device(int __dev) {
		switch (__dev) {
			case device::CPU: to_cpu(); break;
			case device::GPU: to_gpu(); break;
		}
	}

	// resize active memory
	virtual void resize(int __n) = 0;


	// access an element (CPU memory only) for read/write 
	Type& operator()(int __x) { 
		assert(_dev==device::CPU && "Cannot use operator() for GPU memory.");
		return _mem[__x];
	}
	
	// access an element (CPU memory only) for read 
	const Type& operator()(int __x) const { 
		assert(_dev==device::CPU && "Cannot use operator() for GPU memory.");
		return _mem[__x];
	}

	// set the value of the x position
	void set_element(int __x, Type __value) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.set_device_element<Type>(_dmem, __x, &__value);
		#elif defined(__USE_OPENCL__)
		__ocl__.set_buffer_element<Type>(_dmem, __x, &__value);
		#endif
		return;
		}
		_mem[__x] = __value;
	}
	
	// return the value in x position
	Type get_element(int __x) const {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		return __cuda__.get_device_element<Type>(_dmem, __x);
		#elif defined(__USE_OPENCL__)
		return __ocl__.get_buffer_element<Type>(_dmem, __x);
		#endif
		}
		return _mem[__x];
	}

	// returns an element at index 'idx'
	Type sequential(int idx) const {
		return get_element(idx);
	}

	// clears the allocated memory in the cpu memory
	void zero_cpu() {
		if (_mem) std::memset(_mem, 0, _size*sizeof(Type));
	}

	// clears the allocated memory in the gpu memory
	void zero_gpu() {
		#if defined(__USE_CUDA__)
		if (_dmem) __cuda__.vec_set<Type>(_dmem, _size, Type(0));
		#elif defined(__USE_OPENCL__)
		if (_dmem != NullBuffer) __ocl__.vec_set<Type>(_dmem, _size, Type(0));
		#endif	
	}

	// clears the allocated memory
	void zero_active_device() {
		switch (_dev) {
			case device::CPU: zero_cpu(); break;
			case device::GPU: zero_gpu(); break;
		}
	}

	// reshape (shrink)
	void reshape(int __newlen) {
		assert(__newlen <= _size);
		_nel = __newlen;
	}

	// reshape as matrix
	um_ref<Type> reshape(int __ydim, int __xdim, int __xsize, int __ysize) {
		assert(__xsize*__ysize <= _size);
		return um_ref<Type>(active_mem(), _dev, __ydim, __xdim, __xsize, __ysize);
	}
	const um_ref<Type> reshape(int __ydim, int __xdim, int __xsize, int __ysize) const {
		assert(__xsize*__ysize <= _size);
		return um_ref<Type>(active_mem(), _dev, __ydim, __xdim, __xsize, __ysize);
	}

	// reshape as cube
	uc_ref<Type> reshape(int __zdim, int __ydim, int __xdim, int __xsize, int __ysize) {
		assert(__zdim*__xsize*__ysize <= _size);
		return uc_ref<Type>(active_mem(), _dev, __zdim, __ydim, __xdim, __xsize, __ysize, __xsize*__ysize);
	}
	const uc_ref<Type> reshape(int __zdim, int __ydim, int __xdim, int __xsize, int __ysize) const {
		assert(__zdim*__xsize*__ysize <= _size);
		return uc_ref<Type>(active_mem(), _dev, __zdim, __ydim, __xdim, __xsize, __ysize, __xsize*__ysize);
	}


	umem offset(int __x) {
		assert (__x < _size);

		#ifdef __USE_CUDA__ 
		if (_dev==device::GPU) { return umem(_dmem + __x); }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) {
		cl_buffer_region reg;
		reg.origin = __x*sizeof(Type);
		reg.size = (_size-__x)*sizeof(Type);
		cl::Buffer sub = _dmem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg);
		return umem(sub);
		}
		#endif

		return umem(_mem + __x);
	}

	umem offset(int __x) const {
		assert (__x < _size);

		#ifdef __USE_CUDA__ 
		if (_dev==device::GPU) { return umem(_dmem + __x); }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) {
		cl_buffer_region reg;
		reg.origin = __x*sizeof(Type);
		reg.size = (_size-__x)*sizeof(Type);
		cl::Buffer sub = static_cast<cl::Buffer>(_dmem).createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg);
		return umem(sub);
		}
		#endif

		return umem(_mem + __x);
	}


	// copy the values from __src to vector's memory at offset __offset.
	// __src must be in the same device memory as vector's active device.
	void load(int __offset, const umem& __src, int __n) {
		assert(__offset+__n <= len());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.copy<Type>(__src.get_cdmem(), _dmem, 0, __offset, __n);
		#elif defined(__USE_OPENCL__)
		__ocl__.copy<Type>(__src.get_cdmem(), _dmem, 0, __offset, __n);
		#endif
		return;
		}

		cpu_copy(__src.get_cmem(), _mem, 0, __offset, __n);
	}


	// copy the values from vector's memory at position __offset to __dst.
	// __dst must be in the same device memory as vector's active device.
	void store(int __offset, int __n, umem __dst) const {
		assert(__offset+__n <= len());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.copy<Type>(_dmem, __dst.get_dmem(), __offset, 0, __n);
		#elif defined(__USE_OPENCL__)
		__ocl__.copy<Type>(_dmem, __dst.get_dmem(), __offset, 0, __n);
		#endif
		return;
		}

		cpu_copy(_mem, __dst.get_mem(), __offset, 0, __n);
	}

	// copy elements from 'src', using 'idcs'
	template <template <typename> class Vector>
	void copy(const Vector<Type>& __src, int __src_ofs, int __n) {
		assert(_nel >= __src_ofs+__n);
		assert(_dev == __src.dev());
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.copy<Type>(__src.cdmem(), _dmem, __src_ofs, 0, __n);
		#elif defined(__USE_OPENCL__)
		__ocl__.copy<Type>(__src.cdmem(), _dmem, __src_ofs, 0, __n);
		#endif
		return;
		}

		cpu_copy(__src.cmem(), _mem, __src_ofs, 0, __n);
	}

	// copy elements from 'src', using 'idcs'
	template <template <typename> class Vector>
	void copy(const Vector<Type>& __src, const std::vector<int>& __idcs) {
		assert(_nel >= (int)__idcs.size());
		assert(_dev == __src.dev());
		for (int i=0; i<(int)__idcs.size(); ++i)
			set_element(i, __src.get_element(__idcs[i]));
	}

	// set all elements to random real values
	void random_reals(Type __min, Type __max, float __ratio=1.0f) {
		assert(_nel > 0);

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		assert(false && "not implemented yet");
		#elif defined(__USE_OPENCL__)
		assert(false && "not implemented yet");
		#endif
		return;
		}

		umml::uniform_random_reals(_mem, _nel, __min, __max, __ratio);
	}


	// set all elements to random integer values
	void random_ints(Type __min, Type __max, float __ratio=1.0f) {
		assert(_nel > 0);

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		assert(false && "not implemented yet");
		#elif defined(__USE_OPENCL__)
		assert(false && "not implemented yet");
		#endif
		return;
		}

		umml::uniform_random_ints(_mem, _nel, __min, __max, __ratio);
	}


	// set all elements to the given sequence
	void sequence(Type __start, Type __incr) {
		assert(_nel > 0);
		assert(_dev==device::CPU && "Cannot use `sequence` in GPU memory.");

		_mem[0] = __start;
		for (int i=1; i<_nel; ++i) _mem[i] = _mem[i-1] + __incr;
	}


	// v = α
	void set(Type __val) {
		assert(_nel > 0);

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_set<Type>(_dmem, _nel, __val);
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_set<Type>(_dmem, _nel, __val);
		#endif
		return;
		}

		cpu_vecset(_mem, _nel, __val);
	}

	// v = std::vector (CPU only)
	void set(const std::vector<Type>& __vals) {
		assert(_dev==device::CPU);
		assert(_nel <= (int)__vals.size());
		load(0, umem(&__vals[0]), _nel);
	}

	// v = "v1, v2, ..." (CPU only)
	void set(const std::string& vals) {
		assert(_dev==device::CPU);
		load(0, umem(&string_to_values<Type>(vals)[0]), _nel);
	}

	// v = αu
	void set(const uvec<Type>& __other, Type __alpha=Type(1)) {
		assert(len() == __other.len());
		assert(_dev==__other.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_set<Type>(_dmem, __other.cdmem(), _nel, __alpha);
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_set<Type>(_dmem, __other.cdmem(), _nel, __alpha);
		#endif
		return;
		}

		cpu_vecset(_mem, __other.cmem(), _nel, __alpha);
	}

	void set(const uv_ref<Type>& __other, Type __alpha=Type(1)) {
		assert(len() == __other.len());
		assert(_dev==__other.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_set<Type>(_dmem, __other.cdmem(), _nel, __alpha);
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_set<Type>(_dmem, __other.cdmem(), _nel, __alpha);
		#endif
		return;
		}

		cpu_vecset(_mem, __other.cmem(), _nel, __alpha);
	}


	// pads 'src' with zeros
	template <template <typename> class Vector>
	void zero_padded(const Vector<Type>& __src, int __xpad) {
		resize(__src.len()+2*__xpad);
		assert(_nel==__src.len()+2*__xpad);

		if (_dev==device::GPU) {
		zero_gpu();
		#if defined(__USE_CUDA__)
		//__cuda__.copy<Type>(__src.cdmem(), _dmem, __srcofs*_xsize, __ofs*_xsize, __n*_xsize);
		#elif defined(__USE_OPENCL__)
		//__ocl__.copy<Type>(__src.cdmem(), _dmem, __srcofs*_xsize, __ofs*_xsize, __n*_xsize);
		#endif
		return;
		}
		
		zero_cpu();
		for (int x=0; x<__src.len(); ++x) (*this)(x+__xpad) = __src(x);
	}

	
	// check if v == u
	template <template <typename> class Vector>
	bool equals(const Vector<Type>& __other, Type __tolerance=Type(1e-8)) const {
		assert(len() == __other.len());
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		assert(false && "vec equals not implemented");
		//return __cuda__vecequal(_dmem, __other.cdmem(), _nel, __tolerance);
		#elif defined(__USE_OPENCL__)
		assert(false && "vec equals not implemented");
		//return __ocl__vecequal(_dmem, __other.cdmem(), _nel, __tolerance);
		#endif
		return false;
		}
		
		return cpu_vecequal(_mem, __other.cmem(), _nel, __tolerance);
	}


	// v += α;
	void plus(Type __val) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_plus<Type>(_dmem, _nel, __val);
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_plus<Type>(_dmem, _nel, __val);
		#endif
		return;
		}

		cpu_vecplus(_mem, _nel, __val);
	}


	// v += α*u;
	template <template <typename> class Vector>
	void plus(const Vector<Type>& __other, Type __a=Type(1)) {
		assert(len() == __other.len());
		assert(_dev==__other.dev());
	
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.axpy<Type>(__a, __other.cdmem(), _nel, _dmem);
		#elif defined(__USE_OPENCL__)
		__ocl__.axpy<Type>(__a, __other.cdmem(), _nel, _dmem);
		#endif
		return;
		}

		cpu_axpy(__a, __other.cmem(), _nel, _mem);
	}


	// v = α*u1 + β*u2
	template <template <typename> class Vector1, template <typename> class Vector2>
	void add(const Vector1<Type>& __u1, const Vector2<Type>& __u2, Type __a=Type(1), Type __b=Type(1)) {
		assert(len()==__u1.len() && len()==__u2.len());
		assert(_dev==__u1.dev() && _dev==__u2.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.zaxpby<Type>(__a, __u1.cdmem(), _nel, __b, __u2.cdmem(), _dmem);
		#elif defined(__USE_OPENCL__)
		__ocl__.zaxpby<Type>(__a, __u1.cdmem(), _nel, __b, __u2.cdmem(), _dmem);
		#endif
		return;
		}

		cpu_zaxpby(__a, __u1.cmem(), _nel, __b, __u2.cmem(), _mem);
	}


	// v *= α
	void mul(Type __alpha) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_set<Type>(_dmem, _dmem, _nel, __alpha);
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_set<Type>(_dmem, _dmem, _nel, __alpha);
		#endif
		return;
		}

		cpu_vecset(_mem, _mem, _nel, __alpha);
	}


	// v = α*u;
	template <template <typename> class Vector>
	void mul(const Vector<Type>& __other, Type __val) {
		assert(len()==__other.len());
		assert(_dev==__other.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_set<Type>(_dmem, __other.cdmem(), _nel, __val);
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_set<Type>(_dmem, __other.cdmem(), _nel, __val);
		#endif
		return;
		}

		cpu_vecset(_mem, __other.cmem(), _nel, __val);
	}


	// reciprocal v = α*1/v
	template <template <typename> class Vector>
	void reciprocal(const Vector<Type>& __v, Type __alpha=Type(1)) {
		assert(len()==__v.len());
		assert(_dev==__v.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.reciprocal<Type>(__alpha, __v.cdmem(), _nel, _dmem);
		#elif defined(__USE_OPENCL__)
		__ocl__.reciprocal<Type>(__alpha, __v.cdmem(), _nel, _dmem);
		#endif
		return;
		}

		cpu_reciprocal(__alpha, __v.cmem(), _nel, _mem);
	}


	// v = u1*u2 (hadamard)
	template <template <typename> class Vector1, template <typename> class Vector2>
	void prod(const Vector1<Type>& __u1, const Vector2<Type>& __u2) {
		assert(len()==__u1.len() && len()==__u2.len());
		assert(_dev==__u1.dev() && _dev==__u2.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.hadamard<Type>(__u1.cdmem(), __u2.cdmem(), _dmem, _nel);
		#elif defined(__USE_OPENCL__)
		__ocl__.hadamard<Type>(__u1.cdmem(), __u2.cdmem(), _dmem, _nel);
		#endif
		return;
		}

		cpu_hadamard(__u1.cmem(), __u2.cmem(), _mem, _nel);
	}


	// v = v^2
	template <template <typename> class Vector>
	void squared(const Vector<Type>& __other) {
		assert(_nel == __other.len());
		assert(_dev==__other.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_squared<Type>(_dmem, __other.cdmem(), _nel);
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_squared<Type>(_dmem, __other.cdmem(), _nel);
		#endif
		return;
		}

		cpu_vec_squared(_mem, __other.cmem(), _nel);
	}


	// v.u
	template <template <typename> class Vector>
	Type dot(const Vector<Type>& __other) const {
		assert(_nel == __other.len());
		assert(_dev==__other.dev());

		if (_dev==device::GPU) {
		Type dp = 0;
		#if defined(__USE_CUDA__)
		Type* d_dp = __cuda__.alloc<Type>(1);
		__cuda__.dot<Type>(Type(1), _dmem, _nel, Type(1), __other.cdmem(), d_dp);
		__cuda__.to_cpu<Type>(d_dp, &dp, 1);
		CUDA_CHECK(cudaFree(d_dp));
		
		#elif defined(__USE_OPENCL__)
		cl::Buffer d_dp = __ocl__.alloc<Type>(1);
		__ocl__.dot<Type>(Type(1), _dmem, _nel, Type(1), __other.cdmem(), d_dp);
		__ocl__.to_cpu<Type>(d_dp, &dp, 1);
		#endif
		return dp;
		}

		return cpu_dot(__other.cmem(), _nel, _mem);
	}


	// v(m) = A(m,n) * u(n)
	template <template <typename> class Matrix, template <typename> class Vector>
	void mul(const Matrix<Type>& __A, const Vector<Type>& __u) {
		assert(_dev==__A.dev() && _dev==__u.dev());
		assert(__A.xdim() == __u.len());
		resize(__A.ydim());
		assert(len() == __A.ydim());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		//__cuda__.gemv1<Type>(Type(1), __A.cdmem(), __A.ydim(), __A.xdim(), __A.xsize(), __u.cdmem(), Type(0), _dmem);
		__cuda__.gemv2<Type>(Type(1), __A.cdmem(), __A.ydim(), __A.xdim(), __A.xsize(), __u.cdmem(), Type(0), _dmem);
		
		#elif defined(__USE_OPENCL__)
		//__ocl__.gemv1<Type>(Type(1), __A.cdmem(), __A.ydim(), __A.xdim(), __A.xsize(), __u.cdmem(), Type(0), _dmem);
		__ocl__.gemv2<Type>(Type(1), __A.cdmem(), __A.ydim(), __A.xdim(), __A.xsize(), __u.cdmem(), Type(0), _dmem);
		#endif
		return;
		}

		cpu_gemv(Type(1), __A.cmem(), __A.ydim(), __A.xdim(), __A.xsize(), __u.cmem(), Type(0), _mem);
	}


	// apply the function 'f' to all elements of the vector
	void apply_function(int __f) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_func<Type>(_dmem, _nel, __f, Type(0), nullptr);
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_func<Type>(_dmem, _nel, __f, Type(0), NullBuffer);
		#endif
		return;
		}
		cpu_apply_function1d(_mem, _nel, __f);
	}


	// this = f(this + αx)
	template <template <typename> class Vector>
	void apply_function(int __f, Type __alpha, const Vector<Type>& __x) {
		assert(_dev==__x.dev());
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.vec_func<Type>(_dmem, _nel, __f, __alpha, __x.cdmem());
		#elif defined(__USE_OPENCL__)
		__ocl__.vec_func<Type>(_dmem, _nel, __f, __alpha, __x.cdmem());
		#endif
		return;
		}
		cpu_apply_function1d(_mem, _nel, __f, __alpha, __x.cmem());
	}


	// magnitude (euclidean norm)
	Type magnitude() const {
		return std::sqrt(sum_squared());
	}

	// inplace normalization:  y = y / sqrt(Σ(yi*yi))
	Type normalize() {
		Type s = magnitude();
		mul(Type(1.0)/s);
		return s;
	}


	// find the first occurrence of 'val' in the vector
	int find_first(Type val, int start=0) const {
		for (int i=start; i<_nel; ++i) if (get_element(i)==val) return i;
		return -1;
	}

	// find the last occurrence of 'val' in the vector
	int find_last(Type val, int end=0) const {
		for (int i=_nel-1; i>=end; --i) if (get_element(i)==val) return i;
		return -1;
	}
	
	// find a random occurrence of 'val' in the vector
	int find_random(Type val) const {
		uniform_int_distribution<int> dist(0, _nel-1);
		int k = dist(umml::global_rng());
		int pos = find_first(val, k);
		if (pos >= 0) return pos;
		pos = find_last(val, k);
		if (pos >= 0) return pos;
		return -1;
	}

	
	// [4,-3,10,2,5] -> 2
	int argmax() const {
		int imax = 0;
		
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		int* d_pos = __cuda__.alloc<int>(1);
		__cuda__.argmax<Type>(_dmem, _nel, d_pos);
		__cuda__.to_cpu<int>(d_pos, &imax, 1);
		CUDA_CHECK(cudaFree(d_pos));
		
		#elif defined(__USE_OPENCL__)
		cl::Buffer d_pos = __ocl__.alloc<int>(1);
		__ocl__.argmax<Type>(_dmem, _nel, d_pos);
		__ocl__.to_cpu<int>(d_pos, &imax, 1);
		#endif
		return imax;
		}

		for (int i=1; i<_nel; ++i) if (_mem[i] > _mem[imax]) imax = i;
		return imax;
	}

	Type maximum() const {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		return get_device_element(_dmem, argmax());

		#elif defined(__USE_OPENCL__)
		return get_buffer_element(_dmem, argmax());
		#endif
		}

		return _mem[argmax()];
	}

	// [4,-3,10,2,5] -> 1
	int argmin() const {
		int imin = 0;
		
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		int* d_pos = __cuda__.alloc<int>(1);
		__cuda__.argmin<Type>(_dmem, _nel, d_pos);
		__cuda__.to_cpu<int>(d_pos, &imin, 1);
		CUDA_CHECK(cudaFree(d_pos));
		
		#elif defined(__USE_OPENCL__)
		cl::Buffer d_pos = __ocl__.alloc<int>(1);
		__ocl__.argmin<Type>(_dmem, _nel, d_pos);
		__ocl__.to_cpu<int>(d_pos, &imin, 1);
		#endif
		return imin;
		}

		for (int i=1; i<_nel; ++i) if (_mem[i] < _mem[imin]) imin = i;
		return imin;
	}

	Type minimum() const {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		return get_device_element(_dmem, argmin());

		#elif defined(__USE_OPENCL__)
		return get_buffer_element(_dmem, argmin());
		#endif
		}

		return _mem[argmin()];
	}


	// [4,-3,10,2,5] -> [0,0,1,0,0]
	template <typename T, template <typename> class Vector>
	void argmaxto1hot(const Vector<T>& __x) {
		assert(_dev==__x.dev());
		assert(len()==__x.len());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		int pos;
		__cuda__.fill<Type>(_dmem, Type(0), 0, _size);
		int* d_pos = __cuda__.alloc<int>(1);
		__cuda__.argmax<Type>(__x.cdmem(), __x.len(), d_pos);
		__cuda__.to_cpu<int>(d_pos, &pos, 1);
		__cuda__.set_device_element<Type>(_dmem, pos, Type(1));
		CUDA_CHECK(cudaFree(d_pos));
		
		#elif defined(__USE_OPENCL__)
		int pos;
		__ocl__.fill<Type>(_dmem, Type(0), 0, _size);
		cl::Buffer d_pos = __ocl__.alloc<int>(1);
		__ocl__.argmax<Type>(__x.cdmem(), __x.len(), d_pos);
		__ocl__.to_cpu<int>(d_pos, &pos, 1);
		__ocl__.set_buffer_element<Type>(_dmem, pos, Type(1));
		#endif
		return;
		}

		cpu_vec_argmaxto1hot(_mem, _nel, __x.cmem());
	}


	// count of v[i]==val
	int count(Type __val, Type __tolerance=Type(1e-8)) const {
		int n = 0;
	/*
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		int* d_n = __cuda__.alloc<int>(1);
		__cuda__.count_equal<Type>(_dmem, _nel, __x.cdmem(), d_n);
		__cuda__.to_cpu<int>(d_n, &n, 1);
		CUDA_CHECK(cudaFree(d_n));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_n = __ocl__.alloc<int>(1);
		__ocl__.count_equal<Type>(_dmem, _nel, __x.cdmem(), d_n);
		__ocl__.to_cpu<int>(d_n, &n, 1);
		#endif
		return n;
		}
		*/

		for (int i=0; i<_nel; ++i) if (std::abs(_mem[i]-__val) <= __tolerance) n++;
		return n;
	}


	// count v[i]==x[i]
	template <template <typename> class Vector>
	int count_equal(const Vector<Type>& __x) const {
		assert(_dev==__x.dev());
		assert(len()==__x.len());
		int n = 0;

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		int* d_n = __cuda__.alloc<int>(1);
		__cuda__.count_equal<Type>(_dmem, _nel, __x.cdmem(), d_n);
		__cuda__.to_cpu<int>(d_n, &n, 1);
		CUDA_CHECK(cudaFree(d_n));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_n = __ocl__.alloc<int>(1);
		__ocl__.count_equal<Type>(_dmem, _nel, __x.cdmem(), d_n);
		__ocl__.to_cpu<int>(d_n, &n, 1);
		#endif
		return n;
		}

		for (int i=0; i<_nel; ++i) if (_mem[i]==__x(i)) ++n;
		return n;
	}


	// euclidean distance (squared) between two vectors
	template <template <typename> class Vector>
	Type distance_squared(const Vector<Type>& __x) const {
		assert(_dev==__x.dev());
		assert(len()==__x.len());

		if (_dev==device::GPU) {
		Type acc = 0;
		#if defined(__USE_CUDA__)
		Type* d_acc = __cuda__.alloc<Type>(1);
		__cuda__.dist_squared<Type>(_dmem, _nel, __x.cdmem(), d_acc);
		__cuda__.to_cpu<Type>(d_acc, &acc, 1);
		CUDA_CHECK(cudaFree(d_acc));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_acc = __ocl__.alloc<Type>(1);
		__ocl__.dist_squared<Type>(_dmem, _nel, __x.cdmem(), d_acc);
		__ocl__.to_cpu<Type>(d_acc, &acc, 1);
		#endif
		return acc;
		}

		return cpu_distance_squared(_mem, _nel, __x.cmem());
	}


	// manhattan distance between two vectors
	template <template <typename> class Vector>
	Type manhattan_distance(const Vector<Type>& __x) const {
		assert(_dev==__x.dev());
		assert(len()==__x.len());

		if (_dev==device::GPU) {
		Type acc = 0;
		#if defined(__USE_CUDA__)
		Type* d_acc = __cuda__.alloc<Type>(1);
		__cuda__.manhattan<Type>(_dmem, _nel, __x.cdmem(), d_acc);
		__cuda__.to_cpu<Type>(d_acc, &acc, 1);
		CUDA_CHECK(cudaFree(d_acc));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_acc = __ocl__.alloc<Type>(1);
		__ocl__.manhattan<Type>(_dmem, _nel, __x.cdmem(), d_acc);
		__ocl__.to_cpu<Type>(d_acc, &acc, 1);
		#endif
		return acc;
		}

		return cpu_manhattan(_mem, _nel, __x.cmem());
	}


	// result = Σxi
	Type sum() const {
		Type s = 0;
		
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		Type* d_sum = __cuda__.alloc<Type>(1);
		__cuda__.sve<Type>(_dmem, _nel, d_sum);
		__cuda__.to_cpu<Type>(d_sum, &s, 1);
		CUDA_CHECK(cudaFree(d_sum));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_sum = __ocl__.alloc<Type>(1);
		__ocl__.sve<Type>(_dmem, _nel, d_sum);
		__ocl__.to_cpu<Type>(d_sum, &s, 1);
		#endif
		return s;
		}

		cpu_sum(_mem, _nel, &s);
		return s;
	}


	// result = Σ(xi+α)^2
	Type sum_squared(Type alpha=Type(0)) const {
		Type s = 0;
		
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		Type* d_sum = __cuda__.alloc<Type>(1);
		__cuda__.sve2<Type>(_dmem, _nel, alpha, d_sum);
		__cuda__.to_cpu<Type>(d_sum, &s, 1);
		CUDA_CHECK(cudaFree(d_sum));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_sum = __ocl__.alloc<Type>(1);
		__ocl__.sve2<Type>(_dmem, _nel, alpha, d_sum);
		__ocl__.to_cpu<Type>(d_sum, &s, 1);
		#endif
		return s;
		}

		cpu_sum2(_mem, _nel, alpha, &s);
		return s;
	}


	std::string shape() const { 
		std::stringstream ss;
		ss << "(" << _nel << ")";
		if (_size != _nel) ss << "[" << _size << "]";
		return ss.str(); 
	}

	std::string bytes() const { 
		return memory_footprint(_nel*sizeof(Type));
	}

	std::string format(size_t __decimals=0, size_t __padding=0, char __sep=' ', int __n=0) const {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "format() only works with CPU memory.");
		#endif

		std::stringstream ss;
		ss << std::fixed;
		ss << std::setprecision(__decimals);
		int start = 0;
		int n = __n;
		if (n==0) {
			n = _nel;
		} else if (n < 0) {
			n = -n;
			start = _nel - n - 1;
		}
		for (int i=start; i<n; ++i) {
			ss << std::setw(__padding) << _mem[i];
			if (i != n-1) ss << __sep;
		}
		return ss.str();
	}
};





// --------------------------------------------------------------------------------------
// uv_ref
// --------------------------------------------------------------------------------------

template <typename Type=float>
class uv_ref: public uvec_base<Type> {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::uv_ref<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;
 using uvec_base<Type>::_mem;
 using uvec_base<Type>::_nel;
 using uvec_base<Type>::_xsize;
 using uvec_base<Type>::_size;
 using uvec_base<Type>::_dev;
 using uvec_base<Type>::_padgpu;
 #if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
 using uvec_base<Type>::_dmem;
 #endif

 public:
 	uv_ref() = delete;
 	uv_ref(device __dev): uvec_base<Type>() { _dev = __dev; }

	uv_ref(umem __ref, device __dev, int __n, int __size) {
		_nel  = __n;
		_size = _xsize = __size;
		_dev  = __dev;
		_mem  = nullptr;

		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { _dmem = (Type*)__ref.get_mem(); return; }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) { _dmem = (cl::Buffer)__ref.get_dmem(); return; }
		#endif

		_mem = (Type*)__ref.get_mem();
	}

	uv_ref(const std::vector<Type>& __stdvec, int __start=0, int __n=0): 
		uv_ref(umem(&__stdvec[__start]), device::CPU, __n > 0 ? __n : (int)__stdvec.size(), __n > 0 ? __n : (int)__stdvec.size()) {}

	void   force_device(device __dev) override {}
	void   force_padding(bool __padgpu) override {}
	void   host_alloc() override {}

	#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
	void   device_alloc() override {}
	#endif

	void   host_free() override {}

	#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
	void   device_free() override {}
	#endif

	void   to_gpu() override {}
	void   to_cpu() override {}

	void   resize(int __n) override {}
};





// --------------------------------------------------------------------------------------
// uvec
// --------------------------------------------------------------------------------------

template <typename Type=float>
class uvec: public uvec_base<Type> {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::uvec<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;
 using uvec_base<Type>::_mem;
 using uvec_base<Type>::_nel;
 using uvec_base<Type>::_xsize;
 using uvec_base<Type>::_size;
 using uvec_base<Type>::_dev;
 using uvec_base<Type>::_padgpu;
 #if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
 using uvec_base<Type>::_dmem;
 #endif

 public:
	uvec(device __dev=device::CPU): uvec_base<Type>() { _dev = __dev; }

	uvec(int __n, device __dev=device::CPU, bool __padgpu=true) {
		_nel    = __n;
		_dev    = __dev;
		_padgpu = __padgpu;
		_mem    = nullptr;

		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		#ifdef __USE_CUDA__
		_dmem   = nullptr;
		#else
		_dmem   = NullBuffer;
		#endif
		_size = _xsize = (_padgpu ? DIMPAD(_nel) : _nel);
		if (_dev==device::CPU) host_alloc();
		else device_alloc();
		this->zero_active_device();
		return;
		#endif

		_size = _xsize = _nel;
		host_alloc();
	}

	// copy constructor (CPU only or empty, to avoid accidental use for GPU memory)
	uvec(const uvec& __other): uvec() {
		if (__other.dev()==device::CPU) {
			_dev = __other.dev();
			_padgpu = __other._padgpu;
			resize(__other.len());
			int copysize = std::min(_size, __other._size);
			if (_mem) std::memcpy(_mem, __other.cmem(), copysize*sizeof(Type));
			#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
			device_free();
			#endif
		} else {
			assert(__other.empty() && "uvec<Type> copy constructor allowed only for empty vectors");
			_dev = __other.dev();
			_padgpu = __other._padgpu;
		}
	}

	// move constructor
	uvec(uvec&& __tmp) {
		_mem = std::move(__tmp._mem);
		_nel = std::move(__tmp._nel);
		_xsize = std::move(__tmp._xsize);
		_size = std::move(__tmp._size);
		_dev = std::move(__tmp._dev);
		_padgpu = std::move(__tmp._padgpu);
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		_dmem = std::move(__tmp._dmem);
		#endif
		// important, destructor WILL be called for __tmp
		__tmp._mem = nullptr;
		#if defined(__USE_CUDA__)
		__tmp._dmem = nullptr;
		#endif
	}

	// copy from std::vector constructor (CPU only)
	uvec(const std::vector<Type>& __other): uvec() {
		assert(_dev==device::CPU);
		this->resize(__other.size());
		this->set(__other);
	}

	~uvec() override {
		host_free();
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		device_free();
		#endif
	}

	// copy assignment (CPU only)
	uvec& operator =(const uvec& __other) {
		if (__other.dev()==device::CPU) {
			_dev = __other.dev();
			_padgpu = __other._padgpu;
			resize(__other.len());
			int copysize = std::min(_size, __other._size);
			if (_mem) std::memcpy(_mem, __other.cmem(), copysize*sizeof(Type));
			#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
			device_free();
			#endif
		} else {
			assert(__other.empty() && "uvec<Type> copy constructor (GPU) allowed only for empty vectors");
			_dev = __other.dev();
			_padgpu = __other._padgpu;
		}
 		return *this;
	}

	// move assignment (CPU and GPU)
	uvec& operator =(uvec&& __tmp) {
		_mem = std::move(__tmp._mem);
		_nel = std::move(__tmp._nel);
		_xsize = std::move(__tmp._xsize);
		_size = std::move(__tmp._size);
		_dev = std::move(__tmp._dev);
		_padgpu = std::move(__tmp._padgpu);
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		_dmem = std::move(__tmp._dmem);
		#endif
		// important, destructor WILL be called for __tmp
		__tmp._mem = nullptr;
		#if defined(__USE_CUDA__)
		__tmp._dmem = nullptr;
		#endif
		#if defined(__USE_OPENCL__)
		__tmp._dmem = NullBuffer;
		#endif
 		return *this;
	}

	// useful when constructed with the default constructor
	void force_device(device __dev) override { _dev = __dev; }
	void force_padding(bool __padgpu) override { _padgpu = __padgpu; }

	void host_alloc() override {
		_mem = new Type [_size];
	}

	#ifdef __USE_CUDA__
	void device_alloc() override {
		CUDA_CHECK(cudaMalloc((void**)&_dmem, _size*sizeof(Type)));
	}
	#endif
	#ifdef __USE_OPENCL__
	void device_alloc() override {
		_dmem = __ocl__.alloc<Type>(_size);
	}
	#endif

	void host_free() override {
		if (_mem) {
			delete[] _mem;
			_mem = nullptr;
		}
	}

	#ifdef __USE_CUDA__
	void device_free() override {
		if (_dmem) {
			CUDA_CHECK(cudaFree(_dmem));
			_dmem = nullptr;
		}
	}
	#endif
	#ifdef __USE_OPENCL__
	void device_free() override {
		_dmem = NullBuffer;
	}
	#endif


	// CPU -> GPU
	void to_gpu() override {
		#ifdef __USE_CUDA__
		if (_dev==device::GPU) return;
		_dev = device::GPU;
		if (_size==0) return;
		assert(_mem);
		// copy host vectors to device
		if (!_dmem) device_alloc();
		assert(_dmem);
		CUDA_CHECK(cudaMemcpy(_dmem, _mem, _size*sizeof(Type), cudaMemcpyHostToDevice));
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) return;
		_dev = device::GPU;
		if (_size==0) return;
		assert(_mem);
		// copy host vectors to device
		if (_dmem==NullBuffer) device_alloc();
		__ocl__.to_gpu<Type>(_dmem, _mem, _size);
		#endif
	}

	// GPU -> CPU
	void to_cpu() override {
		#ifdef __USE_CUDA__
		if (_dev==device::CPU) return;
		_dev = device::CPU;
		if (_size==0) return;
		assert(_dmem);
		if (!_mem) host_alloc();
		assert(_mem);
		// copy device vectors to host
		CUDA_CHECK(cudaMemcpy(_mem, _dmem, _size*sizeof(Type), cudaMemcpyDeviceToHost));
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::CPU) return;
		_dev = device::CPU;
		if (_size==0) return;
		assert(_dmem != NullBuffer);
		if (!_mem) host_alloc();
		assert(_mem);
		// copy device vectors to host
		__ocl__.to_cpu<Type>(_dmem, _mem, _size);
		#endif
	}


	void resize(int __n) override {
		_nel = __n;

		// GPU or padded CPU
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		_xsize = (_padgpu ? DIMPAD(_nel) : _nel);
		int new_size = _xsize;
		if (new_size <= _size) {
			if (new_size < _size) this->zero_active_device();
			return;
		}
		_size = new_size;
		host_free();
		device_free();
		if (_dev==device::CPU) host_alloc();
		else device_alloc();
		this->zero_active_device();
		return;
		#endif

		// CPU
		if (_nel <= _size) return;
		_size = _xsize = _nel;
		host_free();
		host_alloc();
	}

	template <class Vector>
	void resize_like(const Vector& __other) {
		resize(__other.len());
	}
};


};     // namespace umml

#endif // UMML_UVEC_INCLUDED
