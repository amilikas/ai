#ifndef UMML_UCUB_INCLUDED
#define UMML_UCUB_INCLUDED


/*
 Cube with 2 slices, 2 rows, 3 columns (padded at 4 elements=_xsize])

 1 2 3 .
 4 5 6 .
   a b c .
   d e f .


 Access (z,y,x) = z*zstride + y*xpitch + x;

 A) from a vector 
    [1 2 3 . 4 5 6 . a b c . d e f .]
    zstride = 8, xpitch=4, ypitch=2

 B) from a matrix with 2 rows (one slice per row)
    [1 2 3 . 4 5 6 .]
    [a b c . d e f .]
    zstride = 8, xpitch = 4, ypitch=2

 C) from a matrix with 4 rows (2 rows per slice)
    [1 2 3 .]
    [4 5 6 .]
    [a b c .]
    [d e f .]
    zstride = 8, xpitch=4, ypitch=2
*/

#include "umat.hpp"


namespace umml {


template <typename Type>
std::string format3d(const Type* __mem, int __zdim, int __ydim, int __xdim, 
					 int __xsize, int __ysize, int __zstride,
					 int __decimals, int __padding, int __maxy, char __sep);



////////////////////////////////////////////////////////////////////////////////////
// ucub_base class
//
// Does not do any allocations
// Methods that must be implemented in derived uc_ref and ucub:
// - force_device
// - force_padding
// - host_alloc
// - device_alloc
// - host_free
// - device_free
// - to_gpu
// - to_cpu
// - resize


template <typename Type>
class ucub_base {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::ucub_base<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;

 protected:
	Type*  _mem;
	int    _zdim, _ydim, _xdim;
	int    _zsize, _ysize, _xsize;
	int    _nel, _size;
	device _dev;
	bool   _padgpu;
	#ifdef __USE_CUDA__
	Type*  _dmem;
	#endif
	#ifdef __USE_OPENCL__
	cl::Buffer _dmem;
	#endif

 public:
	ucub_base() {
		_mem    = nullptr;
		_zdim   = _ydim = _xdim = 0;
		_xsize  = _ysize = _zsize = 0;
		_nel    = _size = 0;
		_dev    = device::CPU;
		_padgpu = true;
		#ifdef __USE_CUDA__
		_dmem   = nullptr;
		#endif
		#ifdef __USE_OPENCL__
		_dmem = NullBuffer;
		#endif
	}

	virtual ~ucub_base() {}

	virtual void force_device(device __dev) = 0;
	virtual void force_padding(bool __padgpu) = 0;

	// properties
	bool   empty() const { return _nel==0; }
	int    len() const { return _nel; }
	int    size() const { return _size; }
	dims4  dims() const { return { _xdim,_ydim,_zdim,1 }; }
	int    zdim() const { return _zdim; }
	int    ydim() const { return _ydim; }
	int    xdim() const { return _xdim; }
	int    xsize() const { return _xsize; }
	int    ysize() const { return _ysize; }
	int    zsize() const { return _zsize; }
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
	virtual void resize(int __zdim, int __ydim, int __xdim) = 0;


	Type& operator()(int __z, int __y, int __x) { 
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "Cannot use operator() in GPU memory.");
		#endif
		return _mem[__z*_zsize + __y*_xsize + __x]; 
	}
	
	const Type& operator()(int __z, int __y, int __x) const { 
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "Cannot use operator() in GPU memory.");
		#endif
		return _mem[__z*_zsize + __y*_xsize + __x];
	}

	// set the value of the (y,x) position
	void set_element(int __z, int __y, int __x, Type __value) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.set_device_element<Type>(_dmem, __z*_zsize+__y*_xsize+__x, &__value);
		#elif defined(__USE_OPENCL__)
		__ocl__.set_buffer_element<Type>(_dmem, __z*_zsize, __y*_xsize+__x, &__value);
		#endif
		return;
		}
		_mem[__z*_zsize+__y*_xsize+__x] = __value;
	}
	
	// return the value in the (y,x) position
	Type get_element(int __z, int __y, int __x) const {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		return __cuda__.get_device_element<Type>(_dmem, __z*_zsize+__y*_xsize+__x);
		#elif defined(__USE_OPENCL__)
		return __ocl__.get_buffer_element<Type>(_dmem, __z*_zsize+__y*_xsize+__x);
		#endif
		}
		return _mem[__z*_zsize+__y*_xsize+__x];
	}

	// returns an element at index 'idx' (like if cube was a vector)
	Type sequential(int idx) const {
		int m = idx % zdim();
		return get_element(idx/zdim(), m/xdim(), m%xdim());
	}

	// returns a single slice (matrix) of the cube as a umemory<Type>
	umem slice_offset(int __z) {
		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { return umem(&_dmem[__z*_zsize]); }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) {
		cl_buffer_region reg;
		reg.origin = __z*_zsize*sizeof(Type);
		reg.size = _zsize*sizeof(Type);
		cl::Buffer sub = _dmem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg);
		return umem(sub);
		}
		#endif

		return umem(&_mem[__z*_zsize]);
	}

	// returns a single slice (matrix) of the cube as a umemory<Type>
	umem slice_offset(int __z) const {
		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { return umem(&_dmem[__z*_zsize]); }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) {
		cl_buffer_region reg;
		reg.origin = __z*_zsize*sizeof(Type);
		reg.size = _zsize*sizeof(Type);
		cl::Buffer sub = static_cast<cl::Buffer>(_dmem).createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg);
		return umem(sub);
		}
		#endif

		return umem(&_mem[__z*_zsize]);
	}

/* BUG with __xdim, needs also __xsize
	void set_slice(int __z, const umem& __vals, int __ydim, int __xdim) {
		assert(__z < _zdim && __ydim <= _ydim && __xdim <= _xdim);

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.copy<Type>(__vals.get_cdmem(), _dmem, 0, __z*_zsize, __ydim*__xdim);
		#elif defined(__USE_OPENCL__)
		__ocl__.copy<Type>(__vals.get_cdmem(), _dmem, 0, __z*_zsize, __ydim*__xdim);
		#endif
		return;
		}

		cpu_copy(__vals.get_cmem(), _mem, 0, __z*_zsize, __ydim*__xdim);
	}
*/


	um_ref<Type> slice(int __z) {
		return um_ref<Type>(slice_offset(__z), _dev, _ydim, _xdim, _xsize, _ysize);
	}

	const um_ref<Type> slice(int __z) const {
		return um_ref<Type>(slice_offset(__z), _dev, _ydim, _xdim, _xsize, _ysize);
	}

	// clears the allocated memory in the cpu memory
	void zero_cpu() {
		if (_mem) std::memset(_mem, 0, _size*sizeof(Type));
	}

	// clears the allocated memory in the gpu memory
	void zero_gpu() {
		#ifdef __USE_CUDA__
		if (_dmem) {
		const int BLOCKS = (_size+THREADS-1) / THREADS;
		gpu_vecset<Type><<<BLOCKS,THREADS>>>(_dmem, Type(0), _size);
		CUDA_CHECK(cudaDeviceSynchronize());
		//cudaMemset(_dmem, 0, _size*sizeof(Type));
		}
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


	// copy 2d rects from all slices of 'src' to this cube
	template <template <typename> class Matrix>
	void copy2d(const Matrix<Type>& __src, int __sy, int __sx, int __dy, int __dx, int __ylen, int __xlen) {
		assert(_zdim==__src.zdim() && _ydim >= __dy+__ylen && _xdim >= __dx+__xlen);

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.copy3d<Type>(__src.zdim(), __src.cdmem(), __src.xsize(), __src.zsize(),
							  _dmem, _xsize, _zsize, __sy, __sx, __dy, __dx, __ylen, __xlen);
		#elif defined(__USE_OPENCL__)
		__ocl__.copy3d<Type>(__src.zdim(), __src.cdmem(), __src.xsize(), __src.zsize(),
							  _dmem, _xsize, _zsize, __sy, __sx, __dy, __dx, __ylen, __xlen);
		#endif
		return;
		}
		
		cpu_copy3d(__src.zdim(), __src.cmem(), __src.xsize(), __src.zsize(),
				   _mem, _xsize, _zsize, __sy, __sx, __dy, __dx, __ylen, __xlen);
	}

	// pads 'src' with zeros
	template <template <typename> class Cube>
	void zero_padded(const Cube<Type>& __src, int __ypad, int __xpad, bool __zero=true) {
		resize(__src.zdim(), __src.ydim()+2*__ypad, __src.xdim()+2*__xpad);
		assert(_zdim==__src.zdim() && _ydim==__src.ydim()+2*__ypad && _xdim==__src.xdim()+2*__xpad);

		if (_dev==device::GPU) {
		if (__zero) zero_gpu();
		#if defined(__USE_CUDA__)
		__cuda__.copy3d<Type>(__src.zdim(), __src.cdmem(), __src.xsize(), __src.zsize(),
							 _dmem, _xsize, _zsize, 0, 0, __ypad, __xpad, __src.ydim(), __src.xdim());
		#elif defined(__USE_OPENCL__)
		__ocl__.copy3d<Type>(__src.zdim(), __src.cdmem(), __src.xsize(), __src.zsize(),
							 _dmem, _xsize, _zsize, 0, 0, __ypad, __xpad, __src.ydim(), __src.xdim());
		#endif
		return;
		}
		
		if (__zero) zero_cpu();
		cpu_copy3d(__src.zdim(), __src.cmem(), __src.xsize(), __src.zsize(),
				   _mem, _xsize, _zsize, 0, 0, __ypad, __xpad, __src.ydim(), __src.xdim());
		/*
		for (int z=0; z<__src.zdim(); ++z)
		for (int y=0; y<__src.ydim(); ++y)
		for (int x=0; x<__src.xdim(); ++x) (*this)(z, y+__ypad, x+__xpad) = __src(z,y,x);
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

		float ratio = (__ratio != 1.0f ? __ratio / (_zdim*_ydim) : 1.0f);
		for (int z=0; z<_zdim; ++z)
		for (int i=0; i<_ydim; ++i) 
			umml::uniform_random_reals(&_mem[z*_zsize + i*_xsize], _xdim, __min, __max, ratio);
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

		float ratio = (__ratio != 1.0f ? __ratio / (_zdim*_ydim) : 1.0f);
		for (int z=0; z<_zdim; ++z)
		for (int i=0; i<_ydim; ++i) 
			umml::uniform_random_ints(&_mem[z*_zsize + i*_xsize], _xdim, __min, __max, ratio);
	}

	// C = A
	template <template <typename> class Cube>
	void set(const Cube<Type>& __other) {
		assert(_zdim==__other.zdim() && _ydim==__other.ydim() && _xdim==__other.xdim());
		assert(_dev==__other.dev());

		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.cub_set<Type>(_dmem, _zdim, _ydim, _xdim, _xsize, _ysize, Type(1), __other.cdmem(), __other.xsize(), __other.ysize());
		
		#elif defined(__USE_OPENCL__)
		__ocl__.cub_set<Type>(_dmem, _zdim, _ydim, _xdim, _xsize, _ysize, Type(1), __other.cdmem(), __other.xsize(), __other.ysize());
		#endif
		return;
		}

		cpu_cubset(_mem, _zdim, _ydim, _xdim, _xsize, _ysize, __other.cmem(), __other.xsize(), __other.ysize());
	}


	// apply the function 'f' to all elements of the cube
	void apply_function(int __f) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.cub_func<Type>(_dmem, _zdim, _ydim, _xdim, _xsize, _zsize, __f);
		#elif defined(__USE_OPENCL__)
		__ocl__.cub_func<Type>(_dmem, _zdim, _ydim, _xdim, _xsize, _zsize, __f);
		#endif
		return;
		}
		cpu_apply_function3d(_mem, _zdim, _ydim, _xdim, _xsize, _zsize, __f);
	}

	template <template <typename> class Matrix1, template <typename> class Matrix2>
	void outer(const Matrix1<Type>& __v, const Matrix2<Type>& __u) {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.outer3d<Type>(_dmem, _zdim, _zsize, _xsize,
							  __v.cdmem(), __v.xdim(), __v.xsize(), __u.cdmem(), __u.xdim(), __u.xsize());

		#elif defined(__USE_OPENCL__)
		__ocl__.outer3d<Type>(_dmem, _zdim, _zsize, _xsize,
							  __v.cdmem(), __v.xdim(), __v.xsize(), __u.cdmem(), __u.xdim(), __u.xsize());
		#endif
		return;
		}

		cpu_outer3d(_mem, _zdim, _zsize, _xsize, 
					__v.cmem(),__v.xdim(), __v.xsize(), __u.cmem(), __u.xdim(), __u.xsize());
	}

	// C_z = A_z . B_z
	template <template <typename> class Cube1, template <typename> class Cube2>
	void mul(const Cube1<Type>& __a, const Cube2<Type>& __b) {
std::abort();
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

	// sum all cube's elements
	Type sum() const {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "ucub: sum() only works for CPU memory.");
		#endif
		Type s = (Type)0;
		for (int z=0; z<_zdim; ++z) 
		for (int i=0; i<_ydim; ++i) 
		for (int j=0; j<_xdim; ++j) s += (*this)(z,i,j);
		return s;
	}


	// sum all cube's slices
	template <template <typename> class Matrix>
	void reduce_sum(Matrix<Type>& __A) const {
		if (_dev==device::GPU) {
		#if defined(__USE_CUDA__)
		__cuda__.reduce_sum3d<Type>(_dmem, _zdim, _ydim, _xdim, _xsize, _ysize, __A.dmem());
		#elif defined(__USE_OPENCL__)
		__ocl__.reduce_sum3d<Type>(_dmem, _zdim, _ydim, _xdim, _xsize, _ysize, __A.dmem());
		#endif
		return;
		}
		cpu_reduce_sum3d(_mem, _zdim, _ydim, _xdim, _xsize, _ysize, __A.mem());
	}


	std::string shape() const { 
		std::stringstream ss;
		ss << "(" << _zdim << "," << _ydim << "," << _xdim << ")";
		if (_ysize != _ydim || _xsize != _xdim) ss << "[" << _ysize << "," << _xsize << "]";
		return ss.str();
	}

	std::string format(int __decimals=0, int __padding=0, int __maxrows=0, char __sep=' ') const {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "format() only works for CPU memory.");
		#endif
		return format3d(_mem, _zdim, _ydim, _xdim, _xsize, _ysize, _zsize, __decimals, __padding, __maxrows, __sep);
	}
};


template <typename Type>
std::string format3d(const Type* __mem, int __zdim, int __ydim, int __xdim, 
					 int __xsize, int __ysize, int __zsize,
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
	for (int z=0; z<__zdim; ++z) {
		for (int i=start; i<end; ++i) {
			if (z > 0) ss << std::setw((__padding ? __padding:4)*z) << " ";
			for (int j=0; j<__xdim; ++j) {
				Type value = __mem[z*__zsize + i*__xsize + j];
				ss << std::setw(__padding) << value;
				if (j != __xdim-1) ss << __sep;
			}
			if (i != end-1) ss << "\n";
		}
		if (z != __zdim-1) ss << "\n";
	}
	return ss.str();
}






// --------------------------------------------------------------------------------------
// uc_ref
// --------------------------------------------------------------------------------------

template <typename Type=float>
class uc_ref: public ucub_base<Type> {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::uc_ref<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;
 using ucub_base<Type>::_mem;
 using ucub_base<Type>::_nel;
 using ucub_base<Type>::_size;
 using ucub_base<Type>::_xdim;
 using ucub_base<Type>::_ydim;
 using ucub_base<Type>::_zdim;
 using ucub_base<Type>::_xsize;
 using ucub_base<Type>::_ysize;
 using ucub_base<Type>::_zsize;
 using ucub_base<Type>::_dev;
 using ucub_base<Type>::_padgpu;
 #if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
 using ucub_base<Type>::_dmem;
 #endif

 public:
 	uc_ref() = delete;
	uc_ref(device __dev): ucub_base<Type>() { _dev = __dev; }

	uc_ref(umem __ref, device __dev, int __zdim, int __ydim, int __xdim, int __xsize, int __ysize, int __zsize) {
		_zdim    = __zdim;
		_ydim    = __ydim;
		_xdim    = __xdim;
		_nel     = _zdim * _ydim * _xdim;
		_xsize   = __xsize;
		_ysize   = __ysize;
		_zsize   = __zsize;
		_dev     = __dev;
		_mem     = nullptr;

		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { _dmem = (Type*)__ref.get_dmem(); return; }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) { _dmem = __ref.get_dmem(); return; }
		#endif

		_mem = (Type*)__ref.get_mem();
	}

	void   force_device(device __dev) {}
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
	void   resize(int __zdim, int __ydim, int __xdim) {}
};




// --------------------------------------------------------------------------------------
// ucub
// --------------------------------------------------------------------------------------

template <typename Type=float>
class ucub: public ucub_base<Type> {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::ucub<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;
 using ucub_base<Type>::_mem;
 using ucub_base<Type>::_nel;
 using ucub_base<Type>::_size;
 using ucub_base<Type>::_xdim;
 using ucub_base<Type>::_ydim;
 using ucub_base<Type>::_zdim;
 using ucub_base<Type>::_xsize;
 using ucub_base<Type>::_ysize;
 using ucub_base<Type>::_zsize;
 using ucub_base<Type>::_dev;
 using ucub_base<Type>::_padgpu;
 #if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
 using ucub_base<Type>::_dmem;
 #endif

 public:
	ucub(device __dev=device::CPU): ucub_base<Type>() { _dev = __dev; }

	ucub(int __zdim, int __ydim, int __xdim, device __dev=device::CPU, bool __padgpu=true) {
		_zdim    = __zdim;
		_ydim    = __ydim;
		_xdim    = __xdim;
		_nel     = _zdim * _ydim * _xdim;
		_dev     = __dev;
		_padgpu  = __padgpu;
		_mem     = nullptr;

		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		#ifdef __USE_CUDA__
		_dmem    = nullptr;
		#else
		_dmem    = NullBuffer;
		#endif
		_xsize   = _padgpu ? DIMPAD(_xdim) : _xdim;
		_ysize   = _padgpu ? DIMPAD(_ydim) : _ydim;
		_zsize   = _xsize * _ysize;
		_size    = _zdim * _zsize;
		if (_dev==device::CPU) host_alloc();
		else device_alloc();
		this->zero_active_device();
		return;
		#endif

		_xsize   = _xdim;
		_ysize   = _ydim;
		_zsize   = _xsize * _ysize;
		_size    = _nel;
		host_alloc();
	}

	// copy constructor (CPU only or empty, to avoid accidental use for GPU memory)
	ucub(const ucub& __other): ucub() {
		if (__other.dev()==device::CPU) {
			_dev = __other.dev();
			_padgpu = __other._padgpu;
			resize(__other.zdim(), __other.ydim(), __other.xdim());
			int copysize = std::min(_size, __other._size);
			if (_mem) std::memcpy(_mem, __other.cmem(), copysize*sizeof(Type));
			#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
			device_free();
			#endif
		} else {
			assert(__other.empty() && "ucub<Type> copy constructor allowed only for empty cubes");
			_dev = __other.dev();
			_padgpu = __other._padgpu;
		}
	}

	// move constructor
	ucub(ucub&& __tmp) {
		_mem    = std::move(__tmp._mem);
		_nel    = std::move(__tmp._nel);
		_size   = std::move(__tmp._size);
		_xdim   = std::move(__tmp._xdim);
		_ydim   = std::move(__tmp._ydim);
		_zdim   = std::move(__tmp._zdim);
		_xsize  = std::move(__tmp._xsize);
		_ysize  = std::move(__tmp._ysize);
		_zsize  = std::move(__tmp._zsize);
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
	}

	~ucub() override {
		host_free();
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		device_free();
		#endif
	}

	// move assignment (CPU and GPU)
	ucub& operator =(ucub&& __tmp) {
		_mem    = std::move(__tmp._mem);
		_nel    = std::move(__tmp._nel);
		_size   = std::move(__tmp._size);
		_xdim   = std::move(__tmp._xdim);
		_ydim   = std::move(__tmp._ydim);
		_zdim   = std::move(__tmp._zdim);
		_xsize  = std::move(__tmp._xsize);
		_ysize  = std::move(__tmp._ysize);
		_zsize  = std::move(__tmp._zsize);
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
 		return *this;
	}

	// useful when constructed with the default constructor
	void force_device(device __dev) { _dev = __dev; }
	void force_padding(bool __padgpu) { _padgpu = __padgpu; }

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
		if (_dmem) device_free();
		device_alloc();
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
		if (_mem) host_free();
		host_alloc();
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


	void resize(int __zdim, int __ydim, int __xdim) override {
		// GPU or padded CPU
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		int prev_zdim = _zdim;
		int prev_ydim = _ydim;
		int prev_xdim = _xdim;
		_zdim    = __zdim;
		_ydim    = __ydim;
		_xdim    = __xdim;
		_nel     = _zdim * _ydim * _xdim;
		_xsize   = _padgpu ? DIMPAD(_xdim) : _xdim;
		_ysize   = _padgpu ? DIMPAD(_ydim) : _ydim;
		_zsize   = _xsize * _ysize;
		int new_size = _zdim * _zsize;
		if (new_size <= _size) {
			if ((_zdim != prev_zdim || _ydim != prev_ydim || _xdim < prev_xdim)) this->zero_active_device();
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
		_zdim   = __zdim;
		_ydim   = __ydim;
		_xdim   = __xdim;
		_nel    = _zdim * _ydim * _xdim;
		_ysize  = _ydim;		
		_xsize  = _xdim;
		_zsize  = _xsize * _ysize;
		if (_nel <= _size) return;
		_size   = _nel;
		host_free();
		host_alloc();
	}

	template <class Cube>
	void resize_like(const Cube& __other) {
		resize(__other.zdim(), __other.ydim(), __other.xdim());
	}
};


};     // namespace umml

#endif // UMML_UCUB_INCLUDED
