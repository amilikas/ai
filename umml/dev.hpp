#ifndef UMML_DEV_INCLUDED
#define UMML_DEV_INCLUDED

#ifdef __USE_OPENMP__
#include <thread>
#endif

#ifdef __USE_OPENCL__
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#endif


namespace umml {


// device type
enum device {
	CPU,
	GPU,
};

// device name
const char* device_name(device dev) {
	switch (dev) {
	case CPU: return "CPU";
#if defined(__USE_CUDA__) || defined(__USE_OPENCL__)	
	case GPU: return "GPU";
#else
	case GPU: return "CPU";
#endif
	}
	return "UNKNOWN";
} 


// OpenMP
template <typename T=void>
struct openmp {
	static int threads;
};

template<typename T>
int openmp<T>::threads = 1;

void umml_set_openmp_threads(int threads=-1)
{
	#ifdef __USE_OPENMP__
	if (threads <= -1) threads = std::thread::hardware_concurrency();
	openmp<>::threads = threads;
	#endif
}


#if defined(__USE_CUDA__)
constexpr int TILES   = 32;
constexpr int BLOCKS  = 8;
constexpr int THREADS = 256;

#elif defined(__USE_OPENCL__)
constexpr int TILES   = 16;
constexpr int BLOCKS  = 4;
constexpr int THREADS = 256;

std::string ocl_defines_code = R"(
#define TILE_SIZE  16
#define BLOCK_SIZE 4
#define WARP_SIZE  256
)";

#else
constexpr int TILES   = 1;
constexpr int BLOCKS  = 1;
constexpr int THREADS = 1;
#endif

constexpr int GPUPAD  = 1; //TILES;

template <int N>
int PAD(int d) { return (N * ((unsigned)(d+N-1)/N)); }

// pad dimentions for gpu
inline int DIMPAD(int d)   { return (GPUPAD * ((unsigned)(d+GPUPAD-1)/GPUPAD)); }

// pad kernel argument (matrix)
inline int TILEPAD(int d)  { return (TILES * ((unsigned)(d+TILES-1)/TILES)); }

// pad kernel argument (vector)
inline int WARPPAD(int d)  { return (THREADS * ((unsigned)(d+THREADS-1)/THREADS)); }

#ifdef __USE_OPENCL__
// null buffer (why is this missing from cl2.hpp?)
static const cl::Buffer NullBuffer;
#endif


// unified memory wrapper
#if defined(__USE_OPENCL__)
// ===== OpenCL
// C++ uses Type* memory access but OpenCL uses cl::Buffer
template <typename Type>
struct umemory {
	umemory(): _mem(nullptr), _dmem(NullBuffer) {}
	umemory(Type* __mem): _mem(__mem), _dmem(NullBuffer) {}
	umemory(const Type* __mem): _mem((Type*)__mem), _dmem(NullBuffer) {}
	umemory(cl::Buffer& __dmem): _mem(nullptr), _dmem(__dmem) {}
	umemory(const cl::Buffer& __dmem): _mem(nullptr), _dmem(__dmem) {}
	umemory(umemory& __u): _mem(__u._mem), _dmem(__u._dmem) {}
	umemory(const umemory& __u): _mem(__u._mem), _dmem(__u._dmem) {}

	umemory& operator =(const umemory& __other) { _mem = __other.mem; _dmem = __other._dmem; return *this; }
	
	Type* get_mem() { return _mem; }
	const Type* get_cmem() const { return (const Type*)_mem; }
	cl::Buffer& get_dmem() { return _dmem; }
	const cl::Buffer& get_cdmem() const { return _dmem; }
	
	Type* _mem;
	cl::Buffer _dmem;
};

#else
// ===== CPU and CUDA
// Both C++ and CUDA uses Type* for memory access
template <typename Type>
struct umemory {
	umemory(): _mem(nullptr) {}
	umemory(Type* __mem): _mem(__mem) {}
	umemory(const Type* __mem): _mem((Type*)__mem) {}
	umemory(umemory& __mem): _mem(__mem._mem) {}
	umemory(const umemory& __mem): _mem(__mem._mem) {}
	
	umemory& operator =(const umemory& __other) { _mem = __other.mem; return *this; }

	Type* get_mem() { return _mem; }
	const Type* get_cmem() const { return (const Type*)_mem; }
	Type* get_dmem() { return _mem; }
	const Type* get_cdmem() const { return (const Type*)_mem; }

	Type* _mem;
};

#endif



#ifndef __USE_CUDA__
#define __host__
#define __device__
#endif


// CUDA error checking macro
#ifdef __USE_CUDA__
#define CUDA_DEBUG 1
#if CUDA_DEBUG==1
#define CUDA_CHECK(err) \
	do { \
		cudaError_t err_ = (err); \
		if (err_ != cudaSuccess) { \
			std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
					  << ": " << cudaGetErrorString(err_) << std::endl; \
			exit(EXIT_FAILURE); \
		} \
	} while (0)
#else
#define CUDA_CHECK(err) do { (err); } while (0)
#endif
#endif


};     // namespace umml
#endif // UMML_DEV_INCLUDED
