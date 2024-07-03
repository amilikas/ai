#ifndef UMML_CONV_INCLUDED
#define UMML_CONV_INCLUDED

/*
Example of how 2d convolution of a matrix A(4,4) with a kernel K(2,2) is performed
as a matrix multiplication of the matrix P2C(4,9) and flattened 
K (1,k*k).  conv_width * conv_height = 9

    A(4,4)      K(2,2)
 01 02 03 04    A B
 05 06 07 08    C D
 09 10 11 12
 13 14 15 16

                        P2C(4,9)
               01 02 03 05 06 07 09 10 11
 [A B C D]  x  02 03 04 06 07 08 10 11 12   ==> Conv2D(A,K)
               05 06 07 09 10 11 13 14 15
               06 07 08 10 11 12 14 15 16


Given a col of P2C, the corresponding A(y,x) is calculated as:
 y = stride * (col / ((n-k)/stride+1))
 x = stride * (col % ((n-k)/stride+1))
*/

#include "../../utils.hpp"
#include "../../utensor.hpp"


// CPU implementation
#include "conv_cpu.hpp"

// CUDA implementation
#ifdef __USE_CUDA__
#include "../../cuda.hpp"
#include "conv_cuda.hpp"
#endif

// OpenCL implementation
#ifdef __USE_OPENCL__
#include "../../ocl.hpp"
#include "conv_ocl.hpp"
#endif


namespace umml {


// general 2D convolution
// allocates P2C matrix
template <typename Type, 
		  template <typename> class CubeIn, 
		  template <typename> class Kernel, 
		  template <typename> class MatrixOut>
void geconv2d(const CubeIn<Type>& A, const Kernel<Type>& K, int stride, int pad, MatrixOut<Type>& C)
{
	device dev = A.dev();
	assert(C.dev()==dev && K.dev()==dev);

	int p2c_rows, p2c_cols;
	conv2d_p2c_size(A.zdim(), A.ydim(), A.xdim(), K.ydim(), K.xdim(), stride, &p2c_rows, &p2c_cols);
	umat<Type> p2c(p2c_rows, p2c_cols, dev);
	umat<Type> k(1, K.len(), dev);
	//umat<Type> m; m.force_device(dev);
	const int cw = conv_size(A.xdim(), K.xdim(), stride);
	const int ch = conv_size(A.ydim(), K.ydim(), stride);
	umat<Type> m(ch, cw, dev);

	// GPU
	if (dev==device::GPU) {
	#if defined(__USE_CUDA__)
	k.set(K.cdmem(), K.len());
	__cudaconv__.conv2d_p2c<Type>(A.cdmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
								  K.ydim(), K.xdim(), stride, p2c.dmem(), p2c.xsize());
	m.mul(k, p2c);
	C.set(m.cdmem(), C.xdim());
	#elif defined(__USE_OPENCL__)
	k.set(K.cdmem(), K.len());
	__oclconv__.conv2d_p2c<Type>(A.cdmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
								 K.ydim(), K.xdim(), stride, p2c.dmem(), p2c.xsize());
	m.mul(k, p2c);
	C.set(m.cdmem(), C.xdim());
	#endif
	goto finalize;
	}

	// CPU
	k.set(K.cmem(), K.len());
	cpu_conv2d_p2c(A.cmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
				   K.ydim(), K.xdim(), stride, p2c.mem(), p2c.xsize());
	m.mul(k, p2c);
	C.set(m.cmem(), C.xdim());
	goto finalize;

finalize:
return;
/*
std::cout << "A" << A.shape() << "\n"; if (A.dev()==device::CPU) std::cout << A.format(0,3) << "\n";
std::cout << "k" << k.shape() << "\n"; if (k.dev()==device::GPU) { k.to_cpu(); } std::cout << k.format(0,3) << "\n";
std::cout << "p2c" << p2c.shape() << "\n"; if (p2c.dev()==device::GPU) { p2c.to_cpu(); } std::cout << p2c.format(0,3) << "\n";
std::cout << "m" << m.shape() << "\n"; if (m.dev()==device::GPU) { m.to_cpu(); } std::cout << m.format(0,3) << "\n";
*/
}


// forward convolution, faster version but needs P2C matrix to be instansiated elsewhere. 
// Allocates memory for it only once for consecutive calls. 
// Supports multiple convolutions with multiple kernels K (they must be flattened, with one kernel per row).
// Output is a matrix with one convolution per row, also flattened.
template <typename Type, 
		  template <typename> class CubeIn, 
		  template <typename> class Kernels, 
		  template <typename> class MatrixOut>
void conv2d(const CubeIn<Type>& A, umat<Type>& p2c, const Kernels<Type>& K, 
			int kh, int kw, int stride, int pad, MatrixOut<Type>& C)
{
	device dev = A.dev();
	assert(C.dev()==dev && K.dev()==dev && p2c.dev()==dev);

	int p2c_rows, p2c_cols;
	conv2d_p2c_size(A.zdim(), A.ydim(), A.xdim(), kh, kw, stride, &p2c_rows, &p2c_cols);
	p2c.resize(p2c_rows, p2c_cols);

	// GPU
	if (dev==device::GPU) {
	#if defined(__USE_CUDA__)
	__cudaconv__.conv2d_p2c<Type>(A.cdmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
								  kh, kw, stride, p2c.dmem(), p2c.xsize());
	#elif defined(__USE_OPENCL__)
	__oclconv__.conv2d_p2c<Type>(A.cdmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
								 kh, kw, stride, p2c.dmem(), p2c.xsize());
	#endif
	goto finalize;
	}

	// CPU
	cpu_conv2d_p2c(A.cmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
				   kh, kw, stride, p2c.mem(), p2c.xsize());
	goto finalize;

finalize:

/*
std::cout << "A" << A.shape() << "\n";
std::cout << "K" << K.shape() << "\n"; 
std::cout << "p2c" << p2c.shape() << "\n";
std::cout << "C" << C.shape() << "\n";
*/
	C.mul(K, p2c);
}

// backward convolution, faster version but needs P2C matrix to be instansiated elsewhere. 
// Allocates memory for it only once for consecutive calls. 
// Supports multiple convolutions with multiple kernels K (they must be flattened, with one kernel per row).
// Output is a matrix with one convolution per row, also flattened.
template <typename Type, 
		  template <typename> class CubeIn, 
		  template <typename> class Kernels, 
		  template <typename> class MatrixOut>
void backconv2d(const CubeIn<Type>& A, umat<Type>& p2c, const Kernels<Type>& K, 
				int kh, int kw, int stride, int pad, MatrixOut<Type>& C)
{
	device dev = A.dev();
	assert(C.dev()==dev && K.dev()==dev && p2c.dev()==dev);

	int p2c_rows, p2c_cols;
	back2d_p2c_size(A.zdim(), A.ydim(), A.xdim(), kh, kw, stride, &p2c_rows, &p2c_cols);
	p2c.resize(p2c_rows, p2c_cols);

	// GPU
	if (dev==device::GPU) {
	#if defined(__USE_CUDA__)
	__cudaconv__.back2d_p2c<Type>(A.cdmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
								  kh, kw, stride, p2c.dmem(), p2c.xsize());
	#elif defined(__USE_OPENCL__)
	__oclconv__.back2d_p2c<Type>(A.cdmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
								 kh, kw, stride, p2c.dmem(), p2c.xsize());
	#endif
	goto finalize;
	}

	// CPU
	cpu_back2d_p2c(A.cmem(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(),
				   kh, kw, stride, p2c.mem(), p2c.xsize());
	goto finalize;

finalize:

/*
std::cout << "A" << A.shape() << "\n";
std::cout << "K" << K.shape() << "\n"; 
std::cout << "p2c" << p2c.shape() << "\n";
std::cout << "C" << C.shape() << "\n";
*/
	C.mul(K, p2c);
}


// forward convolution, faster version but needs P2C matrix to be instansiated elsewhere. 
// Allocates memory for it only once for consecutive calls. 
// Supports multiple convolutions with multiple kernels K matrix with one kernel per row (flattened).
// Output is a cube with N convolutions per slice.
template <typename Type, 
		  template <typename> class TensorIn, 
		  template <typename> class Kernels, 
		  template <typename> class CubeOut>
void batched_conv2d(const TensorIn<Type>& A, ucub<Type>& p2c, const Kernels<Type>& K, 
					int kh, int kw, int stride, int pad, CubeOut<Type>& C)
{
	device dev = A.dev();
	assert(C.dev()==dev && K.dev()==dev && p2c.dev()==dev);
	//assert(A.zdim()==K.ydim());

	int p2c_rows, p2c_cols;
	conv2d_p2c_size(A.zdim(), A.ydim(), A.xdim(), kh, kw, stride, &p2c_rows, &p2c_cols);
	p2c.resize(A.wdim(), p2c_rows, p2c_cols);

	// GPU
	if (dev==device::GPU) {
	#if defined(__USE_CUDA__)
	__cudaconv__.batchedconv2d_p2c<Type>(A.cdmem(), A.wdim(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(), A.wsize(),
						kh, kw, stride, p2c.dmem(), p2c.xsize(), p2c.zsize());
	#elif defined(__USE_OPENCL__)
	__oclconv__.batchedconv2d_p2c<Type>(A.cdmem(), A.wdim(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(), A.wsize(),
						kh, kw, stride, p2c.dmem(), p2c.xsize(), p2c.zsize());
	#endif
	goto finalize;
	}

	// CPU
	cpu_batchedconv2d_p2c(A.cmem(), A.wdim(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(), A.wsize(),
						kh, kw, stride, p2c.mem(), p2c.xsize(), p2c.zsize());
	goto finalize;

finalize:
/*
std::cout << "A" << A.shape() << "\n";
std::cout << "K" << K.shape() << "\n"; 
std::cout << "p2c" << p2c.shape() << "\n";
std::cout << "C" << C.shape() << "\n";
*/
	#if !defined(__USE_CUDA__) && !defined(__USE_OPENCL__)
	#pragma omp parallel for
	#endif
	for (int w=0; w<A.wdim(); ++w) {
		um_ref<Type> p = p2c.slice(w);
		um_ref<Type> out = C.slice(w);
		out.mul(K, p);
	}
}

// backward convolution, faster version but needs P2C matrix to be instansiated elsewhere. 
// Allocates memory for it only once for consecutive calls. 
// Supports multiple convolutions with multiple kernels K cube, with nsamples slices, with one kernel per row (flattened).
// Output is a cube with N convolutions per slice.
template <typename Type, 
		  template <typename> class TensorIn, 
		  template <typename> class Kernels, 
		  template <typename> class CubeOut>
void batched_backconv2d(const TensorIn<Type>& A, ucub<Type>& p2c, const Kernels<Type>& K, 
					int kh, int kw, int stride, int pad, CubeOut<Type>& C)
{
	device dev = A.dev();
	assert(C.dev()==dev && K.dev()==dev && p2c.dev()==dev);
	//assert(A.zdim()==K.ydim());

	int p2c_rows, p2c_cols;
	back2d_p2c_size(A.zdim(), A.ydim(), A.xdim(), kh, kw, stride, &p2c_rows, &p2c_cols);
	p2c.resize(A.wdim(), p2c_rows, p2c_cols);

	// GPU
	if (dev==device::GPU) {
	#if defined(__USE_CUDA__)
	__cudaconv__.batchedback2d_p2c<Type>(A.cdmem(), A.wdim(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(), A.wsize(),
						kh, kw, stride, p2c.dmem(), p2c.xsize(), p2c.zsize());
	#elif defined(__USE_OPENCL__)
	__oclconv__.batchedback2d_p2c<Type>(A.cdmem(), A.wdim(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(), A.wsize(),
						kh, kw, stride, p2c.dmem(), p2c.xsize(), p2c.zsize());
	#endif
	goto finalize;
	}

	// CPU
	cpu_batchedback2d_p2c(A.cmem(), A.wdim(), A.zdim(), A.ydim(), A.xdim(), A.xsize(), A.zsize(), A.wsize(),
						kh, kw, stride, p2c.mem(), p2c.xsize(), p2c.zsize());
	goto finalize;

finalize:
/*
std::cout << "A" << A.shape() << "\n";
std::cout << "K" << K.shape() << "\n"; 
std::cout << "p2c" << p2c.shape() << "\n";
std::cout << "C" << C.shape() << "\n";
*/
	#if !defined(__USE_CUDA__) && !defined(__USE_OPENCL__)
	#pragma omp parallel for
	for (int w=0; w<A.wdim(); ++w) {
		um_ref<Type> p = p2c.slice(w);
		um_ref<Type> k = K.slice(w);
		um_ref<Type> out = C.slice(w);
		out.mul(k, p);
	}
	#else
	/*
	__ocl__.bmm<Type>(K.cdmem(), p2c.cdmem(), C.dmem(), A.zdim(), K.ydim(), K.xdim(), p2c.xdim(), 
				K.xsize(), p2c.xsize(), C.xsize(), K.zsize(), p2c.zsize(), C.zsize());
	*/
	for (int w=0; w<A.wdim(); ++w) {
		um_ref<Type> p = p2c.slice(w);
		um_ref<Type> k = K.slice(w);
		um_ref<Type> out = C.slice(w);
		out.mul(k, p);
	}
	#endif
}


// rotate filters 180-degree
//      z1   z2  
// w1: abcd efgh
// w2: ijkl mnop   --rot180-->  dcba lkji tsrq
// w3: qrst vuxy                hgfe ponm yxuv
//
// Args
// m: number of filters (w1, w2, w3)
// k: kernel size (in the example k=2)
// c: number of channels (z1, z2)
// output dims (rows,cols) are: (c, m*k*k)
template <typename Type, 
		  template <typename> class MatrixIn, 
		  template <typename> class MatrixOut>
void rot180(const MatrixIn<Type>& K, int k, int c, MatrixOut<Type>& C)
{
	assert(C.dev()==K.dev());

	// GPU
	if (K.dev()==device::GPU) {
	#if defined(__USE_CUDA__)
	__cudaconv__.rot180<Type>(K.cdmem(), K.ydim(), k, c, K.xsize(), C.dmem(), C.xsize());
	
	#elif defined(__USE_OPENCL__)
	__oclconv__.rot180<Type>(K.cdmem(), K.ydim(), k, c, K.xsize(), C.dmem(), C.xsize());
	#endif
	return;
	}

	// CPU
	cpu_rot180<Type>(K.cmem(), K.ydim(), k, c, K.xsize(), C.mem(), C.xsize());
}

};     // namespace umml

#endif // UMML_CONV_INCLUDED
