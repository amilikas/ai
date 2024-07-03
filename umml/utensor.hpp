#ifndef UMML_UTENSOR_INCLUDED
#define UMML_UTENSOR_INCLUDED


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

#include "ucub.hpp"


namespace umml {


template <typename Type>
std::string format4d(const Type* __mem, int __wdim, int __zdim, int __ydim, int __xdim, 
					 int __xsize, int __ysize, int __zsize, int __wstride,
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
// IMPLEMENT
// random
// argmaxto1hot

template <typename Type>
class utensor_base {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::utensor_base<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;

 protected:
	Type*  _mem;
	int    _wdim, _zdim, _ydim, _xdim;
	int    _wsize, _zsize, _ysize, _xsize;
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
	utensor_base() {
		_mem    = nullptr;
		_wdim   = _zdim = _ydim = _xdim = 0;
		_wsize  = _zsize = _ysize = _xsize = 0;
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

	virtual ~utensor_base() {}

	virtual void force_device(device __dev) = 0;
	virtual void force_padding(bool __padgpu) = 0;

	// properties
	bool   empty() const { return _nel==0; }
	int    len() const { return _nel; }
	int    size() const { return _size; }
	dims4  dims() const { return { _xdim,_ydim,_zdim,_wdim }; }
	int    wdim() const { return _wdim; }
	int    zdim() const { return _zdim; }
	int    ydim() const { return _ydim; }
	int    xdim() const { return _xdim; }
	int    xsize() const { return _xsize; }
	int    ysize() const { return _ysize; }
	int    zsize() const { return _zsize; }
	int    wsize() const { return _wsize; }
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
	virtual void resize(int __wdim, int __zdim, int __ydim, int __xdim) = 0;


	Type& operator()(int __w, int __z, int __y, int __x) { 
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "Cannot use operator() in GPU memory.");
		#endif
		return _mem[__w*_wsize + __z*_zsize + __y*_xsize + __x]; 
	}
	
	const Type& operator()(int __w, int __z, int __y, int __x) const { 
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "Cannot use operator() in GPU memory.");
		#endif
		return _mem[__w*_wsize + __z*_zsize + __y*_xsize + __x];
	}

	// returns a single slice (matrix) of the cube as a umemory<Type>
	umem cube_offset(int __w) {
		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { return umem(&_dmem[__w*_wsize]); }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) {
		cl_buffer_region reg;
		reg.origin = __w*_wsize*sizeof(Type);
		reg.size = _wsize*sizeof(Type);
		cl::Buffer sub = _dmem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg);
		return umem(sub);
		}
		#endif

		return umem(&_mem[__w*_wsize]);
	}

	// returns a single slice (matrix) of the cube as a umemory<Type>
	umem cube_offset(int __w) const {
		#ifdef __USE_CUDA__
		if (_dev==device::GPU) { return umem(&_dmem[__w*_wsize]); }
		#endif

		#ifdef __USE_OPENCL__
		if (_dev==device::GPU) {
		cl_buffer_region reg;
		reg.origin = __w*_wsize*sizeof(Type);
		reg.size = _wsize*sizeof(Type);
		cl::Buffer sub = static_cast<cl::Buffer>(_dmem).createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg);
		return umem(sub);
		}
		#endif

		return umem(&_mem[__w*_wsize]);
	}

	uc_ref<Type> cube(int __w) {
		return uc_ref<Type>(cube_offset(__w), _dev, _zdim, _ydim, _xdim, _xsize, _ysize, _zsize);
	}

	const uc_ref<Type> cube(int __w) const {
		return uc_ref<Type>(cube_offset(__w), _dev, _zdim, _ydim, _xdim, _xsize, _ysize, _zsize);
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
	template <template <typename> class Tensor>
	void copy2d(const Tensor<Type>& __src, int __sy, int __sx, int __dy, int __dx, int __ylen, int __xlen) {
		assert(_wdim==__src.wdim() && _zdim==__src.zdim() && _ydim >= __dy+__ylen && _xdim >= __dx+__xlen);

		if (_dev==device::GPU) {
		for (int w=0; w<_wdim; ++w) {
			uc_ref<Type> in3 = __src.cube(w);
			uc_ref<Type> out3 = cube(w);
			#if defined(__USE_CUDA__)
			__cuda__.copy3d<Type>(in3.zdim(), in3.cdmem(), in3.xsize(), in3.zsize(),
							  out3.dmem(), out3.xsize(), out3.zsize(), __sy, __sx, __dy, __dx, __ylen, __xlen);
			#elif defined(__USE_OPENCL__)
			__ocl__.copy3d<Type>(in3.zdim(), in3.cdmem(), in3.xsize(), in3.zsize(),
							  out3.dmem(), out3.xsize(), out3.zsize(), __sy, __sx, __dy, __dx, __ylen, __xlen);
			#endif
		}
		return;
		}
		
		for (int w=0; w<_wdim; ++w) {
			uc_ref<Type> in3 = __src.cube(w);
			uc_ref<Type> out3 = cube(w);
			cpu_copy3d(in3.zdim(), in3.cmem(), in3.xsize(), in3.zsize(),
							  out3.mem(), out3.xsize(), out3.zsize(), __sy, __sx, __dy, __dx, __ylen, __xlen);
		}
	}

	// pads 'src' with zeros
	template <template <typename> class Tensor>
	void zero_padded(const Tensor<Type>& __src, int __ypad, int __xpad, bool __zero=true) {
		resize(__src.wdim(), __src.zdim(), __src.ydim()+2*__ypad, __src.xdim()+2*__xpad);
		assert(_wdim==__src.wdim() && _zdim==__src.zdim() && _ydim==__src.ydim()+2*__ypad && _xdim==__src.xdim()+2*__xpad);

		if (_dev==device::GPU) {
		if (__zero) zero_gpu();
		for (int w=0; w<_wdim; ++w) {
			uc_ref<Type> in3 = __src.cube(w);
			uc_ref<Type> out3 = cube(w);
			#if defined(__USE_CUDA__)
			__cuda__.copy3d<Type>(in3.zdim(), in3.cdmem(), in3.xsize(), in3.zsize(),
								 out3.dmem(), out3.xsize(), out3.zsize(), 0, 0, __ypad, __xpad, in3.ydim(), in3.xdim());
			#elif defined(__USE_OPENCL__)
			__ocl__.copy3d<Type>(in3.zdim(), in3.cdmem(), in3.xsize(), in3.zsize(),
								 out3.dmem(), out3.xsize(), out3.zsize(), 0, 0, __ypad, __xpad, in3.ydim(), in3.xdim());
			#endif
		}
		return;
		}
		
		if (__zero) zero_cpu();
		for (int w=0; w<__src.wdim(); ++w) {
			uc_ref<Type> in3 = __src.cube(w);
			uc_ref<Type> out3 = cube(w);
			cpu_copy3d(in3.zdim(), in3.cmem(), in3.xsize(), in3.zsize(),
					   out3.mem(), out3.xsize(), out3.zsize(), 0, 0, __ypad, __xpad, in3.ydim(), in3.xdim());
		}
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

		float ratio = (__ratio != 1.0f ? __ratio / (_wdim*_zdim*_ydim) : 1.0f);
		for (int w=0; w<_wdim; ++w)
		for (int z=0; z<_zdim; ++z)
		for (int i=0; i<_ydim; ++i) 
			umml::uniform_random_reals(&_mem[w*_wsize + z*_zsize + i*_xsize], _xdim, __min, __max, ratio);
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

		float ratio = (__ratio != 1.0f ? __ratio / (_wdim*_zdim*_ydim) : 1.0f);
		for (int w=0; w<_wdim; ++w)
		for (int z=0; z<_zdim; ++z)
		for (int i=0; i<_ydim; ++i) 
			umml::uniform_random_ints(&_mem[w*_wsize + z*_zsize + i*_xsize], _xdim, __min, __max, ratio);
	}


	// sum all cube's elements
	Type sum() const {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "ucub: sum() only works for CPU memory.");
		#endif
		Type s = (Type)0;
		for (int w=0; w<_wdim; ++w) 
		for (int z=0; z<_zdim; ++z) 
		for (int i=0; i<_ydim; ++i) 
		for (int j=0; j<_xdim; ++j) s += (*this)(w,z,i,j);
		return s;
	}


	std::string shape() const { 
		std::stringstream ss;
		ss << "(" << _wdim << "," << _zdim << "," << _ydim << "," << _xdim << ")";
		if (_ysize != _ydim || _xsize != _xdim) ss << "[" << _ysize << "," << _xsize << "]";
		return ss.str();
	}

	std::string format(int __decimals=0, int __padding=0, int __maxrows=0, char __sep=' ') const {
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		assert(_dev==device::CPU && "format() only works for CPU memory.");
		#endif
		return format4d(_mem, _wdim, _zdim, _ydim, _xdim, _xsize, _ysize, _zsize, _wsize, __decimals, __padding, __maxrows, __sep);
	}
};


template <typename Type>
std::string format4d(const Type* __mem, int __wdim, int __zdim, int __ydim, int __xdim, 
					 int __xsize, int __ysize, int __zsize, int __wsize,
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
	for (int w=0; w<__wdim; ++w) {
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
		if (w != __wdim-1) ss << "\n";
	}
	return ss.str();
}






// --------------------------------------------------------------------------------------
// ut_ref
// --------------------------------------------------------------------------------------

template <typename Type=float>
class ut_ref: public utensor_base<Type> {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::ut_ref<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;
 using utensor_base<Type>::_mem;
 using utensor_base<Type>::_nel;
 using utensor_base<Type>::_size;
 using utensor_base<Type>::_xdim;
 using utensor_base<Type>::_ydim;
 using utensor_base<Type>::_zdim;
 using utensor_base<Type>::_wdim;
 using utensor_base<Type>::_xsize;
 using utensor_base<Type>::_ysize;
 using utensor_base<Type>::_zsize;
 using utensor_base<Type>::_wsize;
 using utensor_base<Type>::_dev;
 using utensor_base<Type>::_padgpu;
 #if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
 using utensor_base<Type>::_dmem;
 #endif

 public:
 	ut_ref() = delete;
	ut_ref(device __dev): utensor_base<Type>() { _dev = __dev; }

	ut_ref(umem __ref, device __dev, int __wdim, int __zdim, int __ydim, int __xdim, 
		   int __xsize, int __ysize, int __zsize, int __wsize) {
		_wdim    = __wdim;
		_zdim    = __zdim;
		_ydim    = __ydim;
		_xdim    = __xdim;
		_nel     = _wdim * _zdim * _ydim * _xdim;
		_xsize   = __xsize;
		_ysize   = __ysize;
		_zsize   = __zsize;
		_wsize   = __wsize;
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
	void   resize(int __wdim, int __zdim, int __ydim, int __xdim) {}
};




// --------------------------------------------------------------------------------------
// ucub
// --------------------------------------------------------------------------------------

template <typename Type=float>
class utensor: public utensor_base<Type> {
 #ifndef UMML_MSVC
 static_assert(std::is_trivial<Type>(), "umml::ucub<Type>: `Type` must be trivial.");
 #endif

 using umem = umemory<Type>;
 using utensor_base<Type>::_mem;
 using utensor_base<Type>::_nel;
 using utensor_base<Type>::_size;
 using utensor_base<Type>::_xdim;
 using utensor_base<Type>::_ydim;
 using utensor_base<Type>::_zdim;
 using utensor_base<Type>::_wdim;
 using utensor_base<Type>::_xsize;
 using utensor_base<Type>::_ysize;
 using utensor_base<Type>::_zsize;
 using utensor_base<Type>::_wsize;
 using utensor_base<Type>::_dev;
 using utensor_base<Type>::_padgpu;
 #if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
 using utensor_base<Type>::_dmem;
 #endif

 public:
	utensor(device __dev=device::CPU): utensor_base<Type>() { _dev = __dev; }

	utensor(int __wdim, int __zdim, int __ydim, int __xdim, device __dev=device::CPU, bool __padgpu=true) {
		_wdim    = __wdim;
		_zdim    = __zdim;
		_ydim    = __ydim;
		_xdim    = __xdim;
		_nel     = _wdim * _zdim * _ydim * _xdim;
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
		_wsize   = _zdim * _zsize;
		_size    = _wdim * _wsize;
		if (_dev==device::CPU) host_alloc();
		else device_alloc();
		this->zero_active_device();
		return;
		#endif

		_xsize   = _xdim;
		_ysize   = _ydim;
		_zsize   = _xsize * _ysize;
		_wsize   = _zdim * _zsize;
		_size    = _nel;
		host_alloc();
	}

	// copy constructor (CPU only or empty, to avoid accidental use for GPU memory)
	utensor(const utensor& __other): utensor() {
		if (__other.dev()==device::CPU) {
			_dev = __other.dev();
			_padgpu = __other._padgpu;
			resize(__other.wdim(), __other.zdim(), __other.ydim(), __other.xdim());
			int copysize = std::min(_size, __other._size);
			if (_mem) std::memcpy(_mem, __other.cmem(), copysize*sizeof(Type));
			#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
			device_free();
			#endif
		} else {
			assert(__other.empty() && "utensor<Type> copy constructor allowed only for empty cubes");
			_dev = __other.dev();
			_padgpu = __other._padgpu;
		}
	}

	// move constructor
	utensor(utensor&& __tmp) {
		_mem    = std::move(__tmp._mem);
		_nel    = std::move(__tmp._nel);
		_size   = std::move(__tmp._size);
		_xdim   = std::move(__tmp._xdim);
		_ydim   = std::move(__tmp._ydim);
		_zdim   = std::move(__tmp._zdim);
		_wdim   = std::move(__tmp._wdim);
		_xsize  = std::move(__tmp._xsize);
		_ysize  = std::move(__tmp._ysize);
		_zsize  = std::move(__tmp._zsize);
		_wsize  = std::move(__tmp._wsize);
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

	~utensor() override {
		host_free();
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		device_free();
		#endif
	}

	// move assignment (CPU and GPU)
	utensor& operator =(utensor&& __tmp) {
		_mem    = std::move(__tmp._mem);
		_nel    = std::move(__tmp._nel);
		_size   = std::move(__tmp._size);
		_xdim   = std::move(__tmp._xdim);
		_ydim   = std::move(__tmp._ydim);
		_zdim   = std::move(__tmp._zdim);
		_wdim   = std::move(__tmp._wdim);
		_xsize  = std::move(__tmp._xsize);
		_ysize  = std::move(__tmp._ysize);
		_zsize  = std::move(__tmp._zsize);
		_wsize  = std::move(__tmp._wsize);
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


	void resize(int __wdim, int __zdim, int __ydim, int __xdim) override {
		// GPU or padded CPU
		#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)
		int prev_wdim = _wdim;
		int prev_zdim = _zdim;
		int prev_ydim = _ydim;
		int prev_xdim = _xdim;
		_wdim    = __wdim;
		_zdim    = __zdim;
		_ydim    = __ydim;
		_xdim    = __xdim;
		_nel     = _wdim * _zdim * _ydim * _xdim;
		_xsize   = _padgpu ? DIMPAD(_xdim) : _xdim;
		_ysize   = _padgpu ? DIMPAD(_ydim) : _ydim;
		_zsize   = _xsize * _ysize;
		_wsize   = _zdim * _zsize;
		int new_size = _wdim * _wsize;
		if (new_size <= _size) {
			if ((_wdim != prev_wdim || _zdim != prev_zdim || _ydim != prev_ydim || _xdim < prev_xdim)) 
				this->zero_active_device();
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
		_wdim   = __wdim;
		_zdim   = __zdim;
		_ydim   = __ydim;
		_xdim   = __xdim;
		_nel    = _wdim * _zdim * _ydim * _xdim;
		_ysize  = _ydim;		
		_xsize  = _xdim;
		_zsize  = _xsize * _ysize;
		_wsize  = _zdim * _zsize;
		if (_nel <= _size) return;
		_size   = _nel;
		host_free();
		host_alloc();
	}

	template <class Tensor>
	void resize_like(const Tensor& __other) {
		resize(__other.wdim(), __other.zdim(), __other.ydim(), __other.xdim());
	}
};


};     // namespace umml

#endif // UMML_UTENSOR_INCLUDED
