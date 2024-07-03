#ifndef UMML_UMAT_INCLUDED
#define UMML_UMAT_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 Matrix class for unified memory (heterogeneous computing), CPU or GPU

 FILE:     umat.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2023-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 STL string (shape, format)
*/

#include "uvec.hpp"


namespace umml {


enum {
	Axis0 = 0,
	AxisX = Axis0,
	Axis1 = 1,
	AxisY = Axis1,
};

template <typename Type>
std::string format2d(const Type* __mem, int __ydim, int __xdim, int __xsize, 
					 int __decimals, int __padding, int __maxy, char __sep); 


////////////////////////////////////////////////////////////////////////////////////
// umat_base class
//
// Does not do any allocations
// Methods that must be implemented in derived um_ref and umat:
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
// random
// argmaxto1hot

template <typename Type>
class umat_base {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::umat_base<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;

 protected:
	Type*  _mem;
	int    _ydim, _xdim;
	int    _ysize, _xsize;
	int    _nel, _size;
	bool   _padgpu;
	device _dev;
	#ifdef __USE_CUDA__
	Type*  _dmem;
	#endif
	#ifdef __USE_OPENCL__
	cl::Buffer _dmem;
	#endif

 public:
	umat_base() {
		_mem    = nullptr;
		_ydim   = _xdim = 0;
		_ysize  = _xsize = 0;
		_nel    = _size = 0;
		_dev    = device::CPU;
		_padgpu = true;
		#ifdef __USE_CUDA__
		_dmem   = nullptr;
		#endif
		#ifdef __USE_OPENCL__
		_dmem   = NullBuffer;
		#endif
	}

	virtual ~umat_base() {}

	virtual void force_device(device __dev) = 0;
	virtual void force_padding(bool __padgpu) = 0;

	// properties
	bool   empty() const { return _nel==0; }
	int    len() const { return _nel; }
	int    size() const { return _size; }
	dims4  dims() const { return { _xdim,_ydim,1,1 }; }
	int    ydim() const { return _ydim; }
	int    xdim() const { return _xdim; }
	int    xsize() const { return _xsize; }
	int    ysize() const { return _ysize; }
	int    xpadding(int __x) const { return (_padgpu ? DIMPAD(__x) : __x); }
	int    ypadding(int __y) const { return (_padgpu ? DIMPAD(__y) : __y); }

	// to reference it directly as a cube
	int    zdim() const { return 1; }
	int    zsize() const { return _ysize*_xsize; }

	device dev() const { return _dev; }
	Type*  mem() { return _mem; }
	const  Type* cmem() const { return (const Type*)_mem; }

	#ifdef __USE_CUDA__
	Type*  dmem() { return _dmem; }
	const  Type* cdmem() const { return (const Type*)_dmem; }
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
	virtual void resize(int __ydim, int __xdim) = 0;


	// access an element (CPU memory only) for read/write 
	Type& operator()(int __y, int __x) { 
		assert(_dev==device::CPU && "Cannot use operator() for GPU memory.");
		return _mem[__y*_xsize + __x]; 
	}
	
	// access an element (CPU memory only) for read/write 
	const Type& operator()(int __y, int __x) const { 
		assert(_dev==device::CPU && "Cannot use operator() for GPU memory.");
		return _mem[__y*_xsize + __x];
	}


	// set the value of the (y,x) position
	void set_element(int __y, int __x, Type __value) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.set_device_element<Type>(_dmem, __y*_xsize+__x, &__value);
		#elif defined(__USE_OPENCL__)
		__ocl__.set_buffer_element<Type>(_dmem, __y*_xsize+__x, &__value);
		#endif
		return;
		}
		_mem[__y*_xsize+__x] = __value;
	}
	
	// return the value in the (y,x) position
	Type get_element(int __y, int __x) const {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		return __cuda__.get_device_element<Type>(_dmem, __y*_xsize+__x);
		#elif defined(__USE_OPENCL__)
		return __ocl__.get_buffer_element<Type>(_dmem, __y*_xsize+__x);
		#endif
		}
		return _mem[__y*_xsize+__x];
	}

	// returns an element at index 'idx' (like if matrix was a vector)
	Type sequential(int idx) const {
		return get_element(idx/xdim(), idx%xdim());
	}


	// clears the allocated memory in the cpu memory
	void zero_cpu() {
		if (_mem) std::memset(_mem, 0, _size*sizeof(Type));
	}

	// clears the allocated memory in the gpu memory
	void zero_gpu() {
		#ifdef __USE_CUDA__
		if (_dmem) __cuda__.vec_set<Type>(_dmem, _size, Type(0));
		#endif
		#ifdef __USE_OPENCL__
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


	// reshape
	void reshape(int __newydim, int __newxdim) {
		assert(__newydim*__newxdim <= _size);
		_ydim = __newydim;
		_xdim = __newxdim;
	}

	// every row is one slice in a cube
	uc_ref<Type> reshape3d(int __zdim, int __ydim, int __xdim) {
		assert(__zdim*__ydim*__xdim == _xdim);
		return uc_ref<Type>(active_mem(), _dev, __zdim, __ydim, __xdim, __xdim, __ydim, _xsize);
	}
	uc_ref<Type> reshape3d(int __zdim, int __ydim, int __xdim) const {
		assert(__zdim*__ydim*__xdim == _xdim);
		return uc_ref<Type>(active_mem(), _dev, __zdim, __ydim, __xdim, __xdim, __ydim, _xsize);
	}


	// returns a single row of the matrix as a umemory<Type>
	umem row_offset(int __y) {
		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { return umem(&_dmem[__y * _xsize]); }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) {
		cl_buffer_region reg;
		reg.origin = __y*_xsize*sizeof(Type);
		reg.size = _xsize*sizeof(Type);
		cl::Buffer sub = _dmem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg);
		return umem(sub);
		}
		#endif

		return umem(&_mem[__y * _xsize]);
	}

	umem row_offset(int __y) const {
		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { return umem(&_dmem[__y * _xsize]); }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) {
		cl_buffer_region reg;
		reg.origin = __y*_xsize*sizeof(Type);
		reg.size = _xsize*sizeof(Type);
		cl::Buffer sub = static_cast<cl::Buffer>(_dmem).createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg);
		return umem(sub);
		}
		#endif

		return umem(&_mem[__y * _xsize]);
	}

	void set_row(int __y, const umem& __vals, int __n) {
		assert(__y < _ydim && __n <= _xsize);

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.copy<Type>(__vals.get_cdmem(), _dmem, 0, __y*_xsize, __n);
		#elif defined(__USE_OPENCL__)
		__ocl__.copy<Type>(__vals.get_cdmem(), _dmem, 0, __y*_xsize, __n);
		#endif
		return;
		}

		cpu_copy(__vals.get_cmem(), _mem, 0, __y*_xsize, __n);
	}

	uv_ref<Type> row(int __y) {
		return uv_ref<Type>(row_offset(__y), _dev, _xdim, _xsize);
	}

	const uv_ref<Type> row(int __y) const {
		return uv_ref<Type>(row_offset(__y), _dev, _xdim, _xsize);
	}


	// copies 'n' cols from 'src', starting at col offset 'srcofs', into this matrix at 'ofs' col offset
	template <template <typename> class Matrix>
	void copy_cols(const Matrix<Type>& __src, int __srcofs, int __n, int __ofs=0) {
		assert(__src.ydim() == _ydim);
		assert(__srcofs+__n <= __src.xdim());
		assert(__ofs+__n <= _xdim);
		assert(__src.dev() == _dev);

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.copy2d<Type>(__src.cdmem(), __src.xsize(), _dmem, _xsize, 0, __srcofs, 0, __ofs, __src.ydim(), __n);
		#elif defined(__USE_OPENCL__)
		__ocl__.copy2d<Type>(__src.cdmem(), __src.xsize(), _dmem, _xsize, 0, __srcofs, 0, __ofs, __src.ydim(), __n);
		#endif
		return;
		}
		
		cpu_copy2d(__src.cmem(), __src.xsize(), _mem, _xsize, 0, __srcofs, 0, __ofs, __src.ydim(), __n);
	}

	// copies 'n' rows from 'src', starting at row offset 'srcofs', into this matrix at 'ofs' row offset
	template <template <typename> class Matrix>
	void copy_rows(const Matrix<Type>& __src, int __srcofs, int __n, int __ofs=0) {
		assert(__src.xdim() == _xdim);
		assert(__srcofs+__n <= __src.ydim());
		assert(__ofs+__n <= _ydim);
		assert(__src.dev() == _dev);
		if (__src.xsize()==_xsize) {
			if (_dev==device::GPU) {
			#if defined(__USE_CUDA__)
			__cuda__.copy<Type>(__src.cdmem(), _dmem, __srcofs*_xsize, __ofs*_xsize, __n*_xsize);
			#elif defined(__USE_OPENCL__)
			__ocl__.copy<Type>(__src.cdmem(), _dmem, __srcofs*_xsize, __ofs*_xsize, __n*_xsize);
			#endif
			return;
			}
			cpu_copy(__src.cmem(), _mem, __srcofs*_xsize, __ofs*_xsize, __n*_xsize);
		} else {
			for (int i=0; i<__n; ++i)
				set_row(__ofs+i, __src.row_offset(__srcofs+i), _xdim);
		}
	}

	// copies all rows from 'src', using 'idcs'
	template <template <typename> class Matrix>
	void copy_rows(const Matrix<Type>& __src, const std::vector<int>& __idcs) {
		assert(_xdim == __src.xdim());
		assert(_ydim >= (int)__idcs.size());
		assert(_dev == __src.dev());
		for (int i=0; i<(int)__idcs.size(); ++i)
			set_row(i, __src.row_offset(__idcs[i]), _xdim);
	}

	// flattens the matrix by collapsing its dimensions to 1D, in row-major order
	void flatten(uvec<Type>& __storage, bool __preserve_padding=false) const {
		if (__preserve_padding) assert(__storage.xsize() >= _size);
		else assert(__storage.xsize() >= _nel);
		int nrows = (__preserve_padding ? _ysize : _ydim);
		int xsize = (__preserve_padding ? _xsize : _xdim);
		int ofs = 0;
		for (int i=0; i<nrows; ++i) {
			if (_dev==device::GPU) __storage.load(ofs, row_offset(i).get_cdmem(), xsize);
			else __storage.load(ofs, row_offset(i).get_cmem(), xsize);
			ofs += xsize;
		}
	}
	
	// copy a flattened vector back to 2D
	void inflate(const uvec<Type>& __storage, bool __padding_preserved=false) {
		if (__padding_preserved) assert(__storage.len() <= _size);
		assert(__storage.len() <= _nel);
		int ofs = 0;
		for (int i=0; i<_ydim; ++i) {
			if (_dev==device::GPU) __storage.store(ofs, _xdim, row_offset(i).get_dmem());
			else __storage.store(ofs, _xdim, row_offset(i).get_mem());
			ofs += (__padding_preserved ? _xsize : _xdim);
		}
	}

	// copy 2d rect from 'src' to this matrix
	template <template <typename> class Matrix>
	void copy2d(const Matrix<Type>& __src, int __sy, int __sx, int __dy, int __dx, int __ylen, int __xlen) {
		assert(_ydim >= __dy+__ylen && _xdim >= __dx+__xlen);

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.copy2d<Type>(__src.cdmem(), __src.xsize(), _dmem, _xsize, __sy, __sx, __dy, __dx, __ylen, __xlen);
		#elif defined(__USE_OPENCL__)
		__ocl__.copy2d<Type>(__src.cdmem(), __src.xsize(), _dmem, _xsize, __sy, __sx, __dy, __dx, __ylen, __xlen);
		#endif
		return;
		}
		
		cpu_copy2d(__src.cmem(), __src.xsize(), _mem, _xsize, __sy, __sx, __dy, __dx, __ylen, __xlen);
	}

	// pads 'src' with zeros
	template <template <typename> class Matrix>
	void zero_padded(const Matrix<Type>& __src, int __ypad, int __xpad, bool __zero=true) {
		resize(__src.ydim()+2*__ypad, __src.xdim()+2*__xpad);
		assert(_ydim==__src.ydim()+2*__ypad && _xdim==__src.xdim()+2*__xpad);

		if (_dev==device::GPU) {
		if (__zero) zero_gpu();
		#if defined(__USE_CUDA__)
		__cuda__.copy2d<Type>(__src.cdmem(), __src.xsize(), _dmem, _xsize, 0, 0, __ypad, __xpad, __src.ydim(), __src.xdim());
		#elif defined(__USE_OPENCL__)
		__ocl__.copy2d<Type>(__src.cdmem(), __src.xsize(), _dmem, _xsize, 0, 0, __ypad, __xpad, __src.ydim(), __src.xdim());
		#endif
		return;
		}
		
		if (__zero) zero_cpu();
		cpu_copy2d(__src.cmem(), __src.xsize(), _mem, _xsize, 0, 0, __ypad, __xpad, __src.ydim(), __src.xdim());
		/*
		for (int y=0; y<__src.ydim(); ++y)
		for (int x=0; x<__src.xdim(); ++x) (*this)(y+__ypad, x+__xpad) = __src(y,x);
		*/
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

		float ratio = (__ratio != 1.0f ? __ratio / _ydim : 1.0f);
		for (int i=0; i<_ydim; ++i) umml::uniform_random_reals(&_mem[i*_xsize], _xdim, __min, __max, ratio);
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

		float ratio = (__ratio != 1.0f ? __ratio / _ydim : 1.0f);
		for (int i=0; i<_ydim; ++i) umml::uniform_random_ints(&_mem[i*_xsize], _xdim, __min, __max, ratio);
	}

	// set all elements to the given sequence
	void sequence(Type __start, Type __incr) {
		assert(_nel > 0);
		assert(_dev==device::CPU && "Cannot use `sequence` in GPU memory.");

		Type val = __start - __incr;
		for (int i=0; i<_ydim; ++i) 
		for (int j=0; j<_xdim; ++j) _mem[i] = (val += __incr);
	}

	// M = α
	void set(Type __val) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.mat_set<Type>(_dmem, _ydim, _xdim, _xsize, __val);
		#elif defined(__USE_OPENCL__)
		__ocl__.mat_set<Type>(_dmem, _ydim, _xdim, _xsize, __val);
		#endif
		return;
		}
		cpu_matset(_mem, _ydim, _xdim, _xsize, __val);
	}

	// M = memory in the same device
	void set(const umem& __vals, int __vpitch) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.mat_set<Type>(_dmem, _ydim, _xdim, _xsize, Type(1), __vals.get_cdmem(), __vpitch);
		#elif defined(__USE_OPENCL__)
		__ocl__.mat_set<Type>(_dmem, _ydim, _xdim, _xsize, Type(1), __vals.get_cdmem(), __vpitch);
		#endif
		return;
		}
		cpu_matset(_mem, _ydim, _xdim, _xsize, __vals.get_cmem(), __vpitch);
	}

	// M = std::vector (CPU only)
	void set(const std::vector<Type>& vals) {
		assert(_dev==device::CPU);
		set(umem(&vals[0]), _xdim);
	}

	// M = "v1, v2m ..." (CPU only)
	void set(const std::string& vals) {
		assert(_dev==device::CPU);
		set(umem(&string_to_values<Type>(vals)[0]), _xdim);
	}

	// M = A
	template <template <typename> class Matrix>
	void set(const Matrix<Type>& __other) {
		assert(_ydim==__other.ydim() && _xdim==__other.xdim());
		assert(_dev==__other.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.mat_set<Type>(_dmem, _ydim, _xdim, _xsize, Type(1), __other.cdmem(), __other.xsize());
		
		#elif defined(__USE_OPENCL__)
		__ocl__.mat_set<Type>(_dmem, _ydim, _xdim, _xsize, Type(1), __other.cdmem(), __other.xsize());
		#endif
		return;
		}

		cpu_matset(_mem, _ydim, _xdim, _xsize, __other.cmem(), __other.xsize());
	}


	// M += α
	void plus(Type __val) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.mat_plus<Type>(_mem, _ydim, _xdim, _xsize, __val);

		#elif defined(__USE_OPENCL__)
		__ocl__.mat_plus<Type>(_dmem, _ydim, _xdim, _xsize, __val);
		#endif
		return;
		}

		cpu_matplus(_mem, _ydim, _xdim, _xsize, __val); 
	}


	// M += α*A
	template <template <typename> class Matrix>
	void plus(const Matrix<Type>& __other, Type __val=Type(1)) {
		assert(_ydim==__other.ydim() && _xdim==__other.xdim());
		assert(_dev==__other.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.axpy<Type>(__val, __other.cdmem(), _ydim, _xdim, _xsize, _dmem, _xsize);

		#elif defined(__USE_OPENCL__)
		__ocl__.axpy<Type>(__val, __other.cdmem(), _ydim, _xdim, _xsize, _dmem, _xsize);
		#endif
		return;
		}

		cpu_axpy(__val, __other.cmem(), _ydim, _xdim, _xsize, _mem, _xsize);
	}


	// M(m,n) += α*v(n)
	template <template <typename> class Vector>
	void plus_vector(const Vector<Type>& __v, int __axis=AxisX, Type __alpha=Type(1)) {
		assert(__v.len() == (__axis==AxisX ? _xdim:_ydim));
		assert(_dev == __v.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.mplusv<Type>(_dmem, _ydim, _xdim, _xsize, __alpha, __v.cdmem(), __axis);
		
		#elif defined(__USE_OPENCL__)
		__ocl__.mplusv<Type>(_dmem, _ydim, _xdim, _xsize, __alpha, __v.cdmem(), __axis);
		#endif
		return;
		}

		cpu_mplusv(_mem, _ydim, _xdim, _xsize, __alpha, __v.cmem(), __axis); 
	}

	// M = α*M1 + β*M2
	template <template <typename> class Matrix1, template <typename> class Matrix2>
	void add(const Matrix1<Type>& __m1, const Matrix2<Type>& __m2, Type __a=Type(1), Type __b=Type(1)) {
		assert(len()==__m1.len() && __m1.len()==__m2.len());
		assert(_dev==__m1.dev() && _dev==__m2.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.zaxpby<Type>
			(__a, __m1.cdmem(), __m1.ydim(), __m1.xdim(), __m1.xsize(), __b, __m2.cdmem(), __m2.xsize(), _dmem, _xsize);

		#elif defined(__USE_OPENCL__)
		__ocl__.zaxpby<Type>
			(__a, __m1.cdmem(), __m1.ydim(), __m1.xdim(), __m1.xsize(), __b, __m2.cdmem(), __m2.xsize(), _dmem, _xsize);
		#endif
		return;
		}

		cpu_zaxpby(__a, __m1.cmem(), __m1.ydim(), __m1.xdim(), __m1.xsize(), __b, __m2.cmem(), __m2.xsize(), _mem, _xsize);
	}

	// M *= α
	void mul(Type __val) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.mat_mul<Type>(_mem, _ydim, _xdim, _xsize, __val);

		#elif defined(__USE_OPENCL__)
		__ocl__.mat_mul<Type>(_dmem, _ydim, _xdim, _xsize, __val);
		#endif
		return;
		}

		cpu_matmul(_mem, _ydim, _xdim, _xsize, __val); 
	}


	// C(m,p) = A(m,n) . B(n,p)   A cols must equal to B rows, C must be zeroed 
	template <template <typename> class Matrix1, template <typename> class Matrix2>
	void mul(const Matrix1<Type>& __a, const Matrix2<Type>& __b) {
		assert(_dev==__a.dev() && _dev==__b.dev());
		assert(__a.xdim() == __b.ydim());
		resize(__a.ydim(), __b.xdim());
		assert(ydim() == __a.ydim() && xdim() == __b.xdim());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.gemm<Type>(__a.cdmem(), __b.cdmem(), _dmem, __a.ydim(), __a.xdim(), __b.xdim(), __a.xsize(), __b.xsize(), _xsize);

		#elif defined(__USE_OPENCL__)
		__ocl__.gemm<Type>(__a.cdmem(), __b.cdmem(), _dmem, __a.ydim(), __a.xdim(), __b.xdim(), __a.xsize(), __b.xsize(), _xsize);
		#endif
		return;
		}

		zero_cpu();
		cpu_gemm(__a.cmem(), __b.cmem(), _mem,  __a.ydim(), __a.xdim(), __b.xdim(), __a.xsize(), __b.xsize(), _xsize);
	}


	// reciprocal M = α*1/M
	template <template <typename> class Matrix>
	void reciprocal(const Matrix<Type>& __a, Type __alpha=Type(1)) {
		assert(len()==__a.len());
		assert(_dev==__a.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.reciprocal<Type>(__alpha, __a.cdmem(), _nel, _dmem);
		#elif defined(__USE_OPENCL__)
		__ocl__.reciprocal<Type>(__alpha, __a.cdmem(), _nel, _dmem);
		#endif
		return;
		}

		cpu_reciprocal(__alpha, __a.cmem(), _nel, _mem);
	}


	// Hadamard product (element-wise matrix product) C(m,n) = A(m,n) * B(m,n)
	template <template <typename> class Matrix1, template <typename> class Matrix2>
	void prod(const Matrix1<Type>& __a, const Matrix2<Type>& __b) {
		assert(_dev == __a.dev() && _dev == __b.dev());
		assert(_xdim == __a.xdim() && _xdim == __b.xdim());
		assert(_ydim == __a.ydim() && _ydim == __b.ydim());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.hadamard<Type>(__a.cdmem(), __b.cdmem(), _dmem, _ydim, _xdim, _xsize);

		#elif defined(__USE_OPENCL__)
		__ocl__.hadamard<Type>(__a.cdmem(), __b.cdmem(), _dmem, _ydim, _xdim, _xsize);
		#endif
		return;
		}

		cpu_hadamard(__a.cmem(), __b.cmem(), _mem,  _ydim, _xdim, _xsize);
	}

	// Hadamard product (element-wise matrix-vector product) C(m,n) *= v(m|n)
	template <template <typename> class Vector>
	void prod(const Vector<Type>& __v, int __axis=AxisX) {
		assert(_dev == __v.dev());
		assert(__v.len() == (__axis==AxisX ? _xdim : _ydim));

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.prod_vec<Type>(_dmem, _ydim, _xdim, _xsize, __v.cdmem(), __axis);

		#elif defined(__USE_OPENCL__)
		__ocl__.prod_vec<Type>(_dmem, _ydim, _xdim, _xsize, __v.cdmem(), __axis);
		#endif
		return;
		}

		cpu_mprodv(_mem, _ydim, _xdim, _xsize, __v.cmem(), __axis);
	}

	// Matrix/vector (used in softmax)  C(m,n) /= v(m|n)
	template <template <typename> class Vector>
	void divide(const Vector<Type>& __v, int __axis=AxisX) {
		assert(_dev == __v.dev());
		assert(__v.len() == (__axis==AxisX ? _xdim : _ydim));

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.div_vec<Type>(_dmem, _ydim, _xdim, _xsize, __v.cdmem(), __axis);

		#elif defined(__USE_OPENCL__)
		__ocl__.div_vec<Type>(_dmem, _ydim, _xdim, _xsize, __v.cdmem(), __axis);
		#endif
		return;
		}

		cpu_mdivv(_mem, _ydim, _xdim, _xsize, __v.cmem(), __axis);
	}


	template <template <typename> class Matrix>
	void gram(const Matrix<Type>& __a) {
		assert(_dev==__a.dev());
		resize(__a.xdim(), __a.xdim());
		assert(ydim()==__a.xdim());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.gram<Type>(__a.cdmem(), __a.ydim(), __a.xdim(), __a.xsize(), _dmem, _xsize);

		#elif defined(__USE_OPENCL__)
		__ocl__.gram<Type>(__a.cdmem(), __a.ydim(), __a.xdim(), __a.xsize(), _dmem, _xsize);
		#endif
		return;
		}

		zero_cpu();
		cpu_gram(__a.cmem(), __a.ydim(), __a.xdim(), __a.xsize(), _mem, _xsize);
	}


	template <template <typename> class Matrix>
	void transpose(const Matrix<Type>& __a) {
		assert(_dev==__a.dev());
		resize(__a.xdim(), __a.ydim());
		assert(ydim()==__a.xdim() && xdim()==__a.ydim());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.gemt<Type>(__a.cdmem(), __a.ydim(), __a.xdim(), __a.xsize(), _dmem, _xsize);

		#elif defined(__USE_OPENCL__)
		__ocl__.gemt<Type>(__a.cdmem(), __a.ydim(), __a.xdim(), __a.xsize(), _dmem, _xsize);
		#endif
		return;
		}

		cpu_gemt(__a.cmem(), __a.ydim(), __a.xdim(), __a.xsize(), _mem, _xsize);
	}


	// v(m) * u(n) = C(m,n)
	template <template <typename> class Vector1, template <typename> class Vector2>
	void outer(const Vector1<Type>& __v, const Vector2<Type>& __u) {
		assert(_dev==__v.dev() && _dev==__u.dev());
		resize(__v.len(), __u.len());
		assert(ydim()==__v.len() && xdim()==__u.len());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.outer<Type>(_dmem, _xsize, __v.cdmem(), _ydim, __u.cdmem(), _xdim);
		
		#elif defined(__USE_OPENCL__)
		__ocl__.outer<Type>(_dmem, _xsize, __v.cdmem(), _ydim, __u.cdmem(), _xdim);
		#endif
		return;
		}

		cpu_outer(_mem, _xsize, __v.cmem(), _ydim, __u.cmem(), _xdim);
	}


	// apply the function 'f' to all elements of the matrix
	void apply_function(int __f) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.mat_func<Type>(_dmem, _ydim, _xdim, _xsize, __f, Type(1), Type(0), nullptr);
		#elif defined(__USE_OPENCL__)
		__ocl__.mat_func<Type>(_dmem, _ydim, _xdim, _xsize, __f, Type(1), Type(0), NullBuffer);
		#endif
		return;
		}
		cpu_apply_function2d(_mem, _ydim, _xdim, _xsize, __f);
	}

	// this = f(beta*(this + αx))
	template <template <typename> class Vector>
	void apply_function(int __f, Type __beta, Type __alpha, const Vector<Type>& __x, int __axis=AxisX) {
		assert(__x.len() == (__axis==AxisX ? _xdim : _ydim));
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.mat_func<Type>(_dmem, _ydim, _xdim, _xsize, __f, __beta, __alpha, __x.cdmem(), __axis);
		#elif defined(__USE_OPENCL__)
		__ocl__.mat_func<Type>(_dmem, _ydim, _xdim, _xsize, __f, __beta, __alpha, __x.cdmem(), __axis);
		#endif
		return;
		}
		cpu_apply_function2d(_mem, _ydim, _xdim, _xsize, __f, __beta, __alpha, __x.cmem(), __axis);
	}


	template <typename T, template <typename> class Matrix>
	void argmaxto1hot(const Matrix<T>& __A) {
		assert(ydim() == __A.ydim() && xdim() == __A.xdim());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		int* d_idcs = __cuda__.alloc<int>(_ydim);
		int* h_idcs = new int [_ydim];
		__cuda__.fill<Type>(_dmem, Type(0), 0, _size);
		__cuda__.argmax<T>(__A.cdmem(), __A.ydim(), __A.xdim(), __A.xsize(), d_idcs);
		__cuda__.to_cpu<int>(d_idcs, h_idcs, _ydim);
		for (int i=0; i<_ydim; ++i) 
			__cuda__.set_device_element<Type>(_dmem, i*_xsize+h_idcs[i], Type(1));
		CUDA_CHECK(cudaFree(d_idcs));
		delete[] h_idcs;

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_idcs = __ocl__.alloc<int>(_ydim);
		int* h_idcs = new int [_ydim];
		__ocl__.fill<Type>(_dmem, Type(0), 0, _size);
		__ocl__.argmax<T>(__A.cdmem(), __A.ydim(), __A.xdim(), __A.xsize(), d_idcs);
		__ocl__.to_cpu<int>(d_idcs, h_idcs, _ydim);
		for (int i=0; i<_ydim; ++i)
			__ocl__.set_buffer_element<Type>(_dmem, i*_xsize+h_idcs[i], Type(1));
		delete[] h_idcs;
		#endif
		return;
		}

		cpu_mat_argmaxto1hot(_mem, _ydim, _xdim, _xsize, __A.cmem(), __A.xsize());
	}


	// count m[i,j]==a[i,j]
	template <template <typename> class Matrix>
	int count_equal(const Matrix<Type>& __A, Type novalue=Type(0)) const {
		assert(_dev==__A.dev());
		assert(_ydim==__A.ydim() && _xdim==__A.xdim());
		int n = 0;

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		int* d_n = __cuda__.alloc<int>(1);
		__cuda__.count_equal<Type>(_dmem, _ydim, _xdim, _xsize, __A.cdmem(), __A.xsize(), (Type)novalue, d_n);
		__cuda__.to_cpu<int>(d_n, &n, 1);
		CUDA_CHECK(cudaFree(d_n));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_n = __ocl__.alloc<int>(1);
		__ocl__.count_equal<Type>(_dmem, _ydim, _xdim, _xsize, __A.cdmem(), __A.xsize(), (Type)novalue, d_n);
		__ocl__.to_cpu<int>(d_n, &n, 1);
		#endif
		return n;
		}

		for (int i=0; i<_ydim; ++i) 
		for (int j=0; j<_xdim; ++j) if ((*this)(i,j) != (Type)novalue && (*this)(i,j)==__A(i,j)) ++n;
		return n;
	}


	// euclidean distance between two matrices
	template <template <typename> class Matrix>
	Type distance_squared(const Matrix<Type>& __A) const {
		assert(_dev==__A.dev());
		assert(_ydim==__A.ydim() && _xdim==__A.xdim());

		if (_dev==device::GPU) {
		Type acc = 0;
		#if defined(__USE_CUDA__)
		Type* d_acc = __cuda__.alloc<Type>(1);
		__cuda__.dist_squared<Type>(_dmem, _ydim, _xdim, _xsize, __A.cdmem(), __A.xsize(), d_acc);
		__cuda__.to_cpu<Type>(d_acc, &acc, 1);
		CUDA_CHECK(cudaFree(d_acc));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_acc = __ocl__.alloc<Type>(1);
		__ocl__.dist_squared<Type>(_dmem, _ydim, _xdim, _xsize, __A.cdmem(), __A.xsize(), d_acc);
		__ocl__.to_cpu<Type>(d_acc, &acc, 1);
		#endif
		return acc;
		}

		return cpu_distance_squared(_mem, _ydim, _xdim, _xsize, __A.cmem(), __A.xsize());
	}


	// manhattan distance between two matrices
	template <template <typename> class Matrix>
	Type manhattan_distance(const Matrix<Type>& __A) const {
		assert(_dev==__A.dev());
		assert(_ydim==__A.ydim() && _xdim==__A.xdim());

		if (_dev==device::GPU) {
		Type acc = 0;
		#if defined(__USE_CUDA__)
		Type* d_acc = __cuda__.alloc<Type>(1);
		__cuda__.manhattan<Type>(_dmem, _ydim, _xdim, _xsize, __A.cdmem(), __A.xsize(), d_acc);
		__cuda__.to_cpu<Type>(d_acc, &acc, 1);
		CUDA_CHECK(cudaFree(d_acc));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_acc = __ocl__.alloc<Type>(1);
		__ocl__.manhattan<Type>(_dmem, _ydim, _xdim, _xsize, __A.cdmem(), __A.xsize(), d_acc);
		__ocl__.to_cpu<Type>(d_acc, &acc, 1);
		#endif
		return acc;
		}

		return cpu_manhattan(_mem, _ydim, _xdim, _xsize, __A.cmem(), __A.xsize());
	}

	// maximum element
	Type maximum() const {
		uvec<Type> rows(ydim(), device());
		reduce_max(rows, AxisY);
		return rows.maximum();
	}

	// minimum element
	Type minimum() const {
		uvec<Type> rows(ydim(), device());
		reduce_min(rows, AxisY);
		return rows.minimum();
	}

	// max of rows (axis=1) or max of cols (axis=0)
	template <template <typename> class Vector>
	void reduce_max(Vector<Type>& __output, int __axis) const {
		assert(_dev == __output.dev());
		assert(__output.len() == (__axis==AxisX ? _xdim : _ydim));

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.reduce_max2d<Type>(_dmem, _ydim, _xdim, _xsize, __output.dmem(), __axis);
		#elif defined(__USE_OPENCL__)
		__ocl__.reduce_max2d<Type>(_dmem, _ydim, _xdim, _xsize, __output.dmem(), __axis);
		#endif
		return;
		}

		cpu_reduce_max2d(_mem, _ydim, _xdim, _xsize, __output.mem(), __axis);
	}

	// max of rows (axis=1) or max of cols (axis=0)
	template <template <typename> class Vector>
	void reduce_min(Vector<Type>& __output, int __axis) const {
		assert(_dev == __output.dev());
		assert(__output.len() == (__axis==AxisX ? _xdim : _ydim));

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.reduce_min2d<Type>(_dmem, _ydim, _xdim, _xsize, __output.dmem(), __axis);
		#elif defined(__USE_OPENCL__)
		__ocl__.reduce_min2d<Type>(_dmem, _ydim, _xdim, _xsize, __output.dmem(), __axis);
		#endif
		return;
		}

		cpu_reduce_min2d(_mem, _ydim, _xdim, _xsize, __output.mem(), __axis);
	}


	// sum of rows (axis=1) or sum of cols (axis=0)
	template <template <typename> class Vector>
	void reduce_sum(Vector<Type>& __output, int __axis) const {
		assert(_dev == __output.dev());
		assert(__output.len() == (__axis==AxisX ? _xdim : _ydim));

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.reduce_sum2d<Type>(_dmem, _ydim, _xdim, _xsize, __output.dmem(), __axis);
		#elif defined(__USE_OPENCL__)
		__ocl__.reduce_sum2d<Type>(_dmem, _ydim, _xdim, _xsize, __output.dmem(), __axis);
		#endif
		return;
		}

		cpu_reduce_sum2d(_mem, _ydim, _xdim, _xsize, __output.mem(), __axis);
	}


	// s = Σm(i,j)
	Type sum() const {
		Type s = 0;

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		Type* d_sum = __cuda__.alloc<Type>(1);
		//__cuda__.sve<Type>(_dmem, _ydim*_xsize, d_sum);
		__cuda__.sme<Type>(_dmem, _ydim, _xdim, _xsize, d_sum);
		__cuda__.to_cpu<Type>(d_sum, &s, 1);
		CUDA_CHECK(cudaFree(d_sum));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_sum = __ocl__.alloc<Type>(1);
		//__ocl__.sve<Type>(_dmem, _ydim*_xsize, d_sum);
		__ocl__.sme<Type>(_dmem, _ydim, _xdim, _xsize, d_sum);
		__ocl__.to_cpu<Type>(d_sum, &s, 1);
		#endif
		return s;
		}

		cpu_sum(_mem, _ydim, _xdim, _xsize, &s);
		return s;
	}


	// s = Σ(m(i,j)+α)^2
	Type sum_squared(Type alpha=Type(0)) const {
		Type s = 0;

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		Type* d_sum = __cuda__.alloc<Type>(1);
		//__cuda__.sve<Type>(_dmem, _ydim*_xsize, d_sum);
		__cuda__.sme2<Type>(_dmem, _ydim, _xdim, _xsize, alpha, d_sum);
		__cuda__.to_cpu<Type>(d_sum, &s, 1);
		CUDA_CHECK(cudaFree(d_sum));

		#elif defined(__USE_OPENCL__)
		cl::Buffer d_sum = __ocl__.alloc<Type>(1);
		__ocl__.sme2<Type>(_dmem, _ydim, _xdim, _xsize, alpha, d_sum);
		__ocl__.to_cpu<Type>(d_sum, &s, 1);
		#endif
		return s;
		}

		cpu_sum2(_mem, _ydim, _xdim, _xsize, alpha, &s);
		return s;
	}


	std::string shape() const { 
		std::stringstream ss;
		ss << "(" << _ydim << "," << _xdim << ")";
		if (_ysize != _ydim || _xsize != _xdim) ss << "[" << _ysize << "," << _xsize << "]";
		return ss.str();
	}

	std::string bytes() const { 
		return memory_footprint(_nel*sizeof(Type));
	}

	std::string format(int __decimals=0, int __padding=0, int __maxrows=0, char __sep=' ') const {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "format() only works for CPU memory.");
		#endif
		return format2d(_mem, _ydim, _xdim, _xsize, __decimals, __padding, __maxrows, __sep);
	}

	std::string full_format(int __decimals=0, int __padding=0, int __maxrows=0, char __sep=' ') const {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "format() only works for CPU memory.");
		#endif
		return format2d(_mem, _ysize, _xsize, _xsize, __decimals, __padding, __maxrows, __sep);
	}

};


template <typename Type>
std::string format2d(const Type* __mem, int __ydim, int __xdim, int __padx, 
					 int __decimals, int __padding, int __maxy, char __sep) 
{
	std::stringstream ss;
	ss << std::fixed;
	ss << std::setprecision(__decimals);
	int start=0, end=__maxy;
	if (end < 0) { 
		start = __ydim + __maxy;
		end = __ydim;
	}
	if (end==0) end = __ydim;
	for (int i=start; i<end; ++i) {
		for (int j=0; j<__xdim; ++j) {
			Type value = __mem[i*__padx + j];
			ss << std::setw(__padding) << value;
			if (j != __xdim-1) ss << __sep;
		}
		if (i != end-1) ss << "\n";
	}
	return ss.str();
}




// --------------------------------------------------------------------------------------
// um_ref
// --------------------------------------------------------------------------------------

template <typename Type=float>
class um_ref: public umat_base<Type> {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "mml::um_ref<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;
 using umat_base<Type>::_mem;
 using umat_base<Type>::_nel;
 using umat_base<Type>::_size;
 using umat_base<Type>::_xdim;
 using umat_base<Type>::_ydim;
 using umat_base<Type>::_xsize;
 using umat_base<Type>::_ysize;
 using umat_base<Type>::_dev;
 using umat_base<Type>::_padgpu;
 #if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
 using umat_base<Type>::_dmem;
 #endif

 public:
 	um_ref() = delete;
 	um_ref(device __dev): umat_base<Type>() { _dev = __dev; }

	um_ref(umem __ref, device __dev, int __ydim, int __xdim, int __padx, int __pady) {
		_ydim  = __ydim;
		_xdim  = __xdim;
		_nel   = _ydim * _xdim;
		_xsize = __padx;
		_ysize = __pady;
		_size  = _ysize * _xsize;
		_dev   = __dev;

		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { _dmem = (Type*)__ref.get_dmem(); return; }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) { _dmem = __ref.get_dmem(); return; }
		#endif

		_mem = (Type*)__ref.get_mem();
	}

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
	void   resize(int __ydim, int __xdim) {}
};




// --------------------------------------------------------------------------------------
// umat
// --------------------------------------------------------------------------------------

template <typename Type=float>
class umat: public umat_base<Type> {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "mml::umat<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;
 using umat_base<Type>::_mem;
 using umat_base<Type>::_nel;
 using umat_base<Type>::_size;
 using umat_base<Type>::_xdim;
 using umat_base<Type>::_ydim;
 using umat_base<Type>::_xsize;
 using umat_base<Type>::_ysize;
 using umat_base<Type>::_dev;
 using umat_base<Type>::_padgpu;
 #if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
 using umat_base<Type>::_dmem;
 #endif

 public:
 	umat(device __dev=device::CPU): umat_base<Type>() { _dev = __dev; }

	umat(int __ydim, int __xdim, device __dev=device::CPU, bool __padgpu=true) {
		_ydim   = __ydim;
		_xdim   = __xdim;
		_nel    = _ydim * _xdim;
		_dev    = __dev;
		_mem    = nullptr;

		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		#ifdef __USE_CUDA__
		_dmem   = nullptr;
		#else
		_dmem   = NullBuffer;
		#endif
		_padgpu = __padgpu;
		_xsize  = _padgpu ? DIMPAD(_xdim) : _xdim;
		_ysize  = _padgpu ? DIMPAD(_ydim) : _ydim;
		_size   = _ysize * _xsize;
		if (_dev==device::CPU) host_alloc();
		else device_alloc();
		this->zero_active_device();
		return;
		#endif

		_xsize = _xdim;
		_ysize = _ydim;
		_size  = _nel;
		host_alloc();
	}

	// copy constructor (CPU only or empty, to avoid accidental use for GPU memory)
	umat(const umat& __other): umat() {
		if (__other.dev()==device::CPU) {
			_dev = __other.dev();
			_padgpu = __other._padgpu;
			resize(__other.ydim(), __other.xdim());
			int copysize = std::min(_size, __other._size);
			if (_mem) std::memcpy(_mem, __other.cmem(), copysize*sizeof(Type));
			#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
			device_free();
			#endif
		} else {
			assert(__other.empty() && "umat<Type> copy constructor allowed only for empty matrices");
			_dev = __other.dev();
			_padgpu = __other._padgpu;
		}
	}
	
	// move constructor
	umat(umat&& __tmp) {
		_mem    = std::move(__tmp._mem);
		_nel    = std::move(__tmp._nel);
		_size   = std::move(__tmp._size);
		_xdim   = std::move(__tmp._xdim);
		_ydim   = std::move(__tmp._ydim);
		_xsize  = std::move(__tmp._xsize);
		_ysize  = std::move(__tmp._ysize);
		_dev    = std::move(__tmp._dev);
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

	~umat() override {
		host_free();
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		device_free();
		#endif
	}

	// copy assignment (CPU only)
	umat& operator =(const umat& __other) {
		if (__other.dev()==device::CPU) {
			_dev = __other.dev();
			_padgpu = __other._padgpu;
			resize(__other.ydim(), __other.xdim());
			int copysize = std::min(_size, __other._size);
			if (_mem) std::memcpy(_mem, __other.cmem(), copysize*sizeof(Type));
			#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
			device_free();
			#endif
		} else {
			assert(__other.empty() && "umat<Type> copy constructor (GPU) allowed only for empty matrices");
			_dev = __other.dev();
			_padgpu = __other._padgpu;
		}
 		return *this;
	}
	
	// move assignment (CPU and GPU)
	umat& operator =(umat&& __tmp) {
		_mem    = std::move(__tmp._mem);
		_nel    = std::move(__tmp._nel);
		_size   = std::move(__tmp._size);
		_xdim   = std::move(__tmp._xdim);
		_ydim   = std::move(__tmp._ydim);
		_xsize  = std::move(__tmp._xsize);
		_ysize  = std::move(__tmp._ysize);
		_dev    = std::move(__tmp._dev);
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
		// copy host memory to device memory
		if (!_dmem) device_alloc();
		assert(_dmem);
		CUDA_CHECK(cudaMemcpy(_dmem, _mem, _size*sizeof(Type), cudaMemcpyHostToDevice));
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) return;
		_dev = device::GPU;
		if (_size==0) return;
		assert(_mem);
		// copy host memory to device memory
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
		// copy device memory to host memory
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


	void resize(int __ydim, int __xdim) override {
		// GPU or padded CPU
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		int prev_ydim = _ydim;
		int prev_xdim = _xdim;
		_ydim  = __ydim;
		_xdim  = __xdim;
		_nel   = _ydim * _xdim;
		_xsize = _padgpu ? DIMPAD(_xdim) : _xdim;
		_ysize = _padgpu ? DIMPAD(_ydim) : _ydim;
		int new_size = _ysize * _xsize;
		if (new_size <= _size) {
			if ((_ydim != prev_ydim || _xdim < prev_xdim)) this->zero_active_device();
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

		// CPU only
		_ydim  = __ydim;
		_xdim  = __xdim;
		_nel   = _ydim * _xdim;
		_xsize = _xdim;
		_ysize = _ydim;		
		if (_nel <= _size) return;
		_size = _nel;
		host_free();
		host_alloc();
	}


	template <class Matrix>
	void resize_like(const Matrix& __other) {
		resize(__other.ydim(), __other.xdim());
	}
};


};     // namespace umml

#endif // UMML_UMAT_INCLUDED
