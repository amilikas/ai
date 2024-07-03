#ifndef UMML_BLAS_CUDA
#define UMML_BLAS_CUDA

#include "dev.hpp"


namespace umml {

// reciprocal
// y = α(1/x)
template <typename Type>
__global__ void gpu_reciprocal(Type alpha, const Type* x, int n, Type* y) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = (x[i] != 0 ? alpha/x[i] : 0);
}


// axpy : vector(n) = scalar*vector(n) + vector(n)
// y = αx + y
template <typename Type>
__global__ void gpu_axpy(Type alpha, const Type* x, int n, Type* y) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] += alpha*x[i];
}


// zaxpby : vector(n) = scalar*vector(n) + scalar*vector(n)
// z = αx + βy
template <typename Type>
__global__ void gpu_zaxpby(Type alpha, const Type* x, int n, Type beta, const Type* y, Type* z) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) z[i] = alpha*x[i] + beta*y[i];
}


// Y = α*X + Y
template <typename Type>
__global__ void gpu_axpy(Type alpha, const Type* a, int nrows, int ncols, int apitch, Type* c, int cpitch)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols) 
		c[y*cpitch+x] += alpha*a[y*apitch+x];
}


// Z = αX + βY
template <typename Type>
__global__ void gpu_zaxpbym(Type alpha, const Type* a, int nrows, int ncols, int apitch, 
							Type beta, const Type* b, int bpitch, Type* c, int cpitch)
{
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y >= nrows || x >= ncols) return;
	c[y*cpitch+x] = alpha*a[y*apitch+x] + beta*b[y*bpitch+x];
}


// dot : scalar = vector(n) . vector(n)
// dot = x.y
template <typename Type>
__global__ void gpu_dot_partial(Type alpha, const Type* x, int n, Type beta, const Type* y, Type* partial) 
{
	__shared__ Type cache[THREADS];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int cacheIndex = threadIdx.x;
	
	Type temp = Type(0);
	while (i < n) {
		temp += alpha*x[i] * beta*y[i];
		i += blockDim.x * gridDim.x;
	}
	
	// set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	__syncthreads();
	
	// for reductions, threadsPerBlock must be a power of 2
	i = blockDim.x/2;
	while (i != 0) {
		if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0)
		partial[blockIdx.x] = cache[0];
}


// gemv : vector(m) = scalar*matrix(m,n)*vector(n) + scalar*vector(m)
// y = αA*x + βy
template <typename Type>
__global__ void gpu_gemv1(Type alpha, const Type* a, int m, int n, int pitch, const Type* x, Type beta, Type* y) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	Type acc = 0;
	if (i < m) {
		for (int j=0; j<n; ++j) acc += alpha*a[i*pitch + j] * x[j];
		y[i] = acc + (beta==Type(0) ? 0 : beta*y[i]);
	}
}

// v(m) = A(mn) . u(n) matrix.vector 
// Works: Yes but it is slow, 2.81 sec in nntest-cuda (CPU is 0.91 sec)
template <typename Type>
__global__ void gpu_gemv2(Type alpha, const Type* a, int m, int n, int pitch, const Type* x, Type beta, Type* y) 
{
	// compute each thread's global row and column index
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	// statically allocated shared memory
	__shared__ Type tile[TILES];

	// compute partial dot product
	Type acc = 0;
	for (int k=col; k<n; k += gridDim.x*blockDim.x)
		acc += alpha*a[row*pitch+k] * x[k];

	// each thread stores its partial sum in 'tile'
	int cols = blockDim.x; 
	tile[threadIdx.y+blockDim.y*threadIdx.x] = acc;
	__syncthreads();

	// reduce sums in log2(cols) steps
	while (cols > 1) {
		cols /= 2;
		if (threadIdx.x < cols) 
			tile[threadIdx.y+blockDim.y*threadIdx.x] += tile[threadIdx.y+blockDim.y*(threadIdx.x+cols)];
		__syncthreads();
	}

	if (threadIdx.x == 0) y[row] = tile[threadIdx.y] + (beta==Type(0) ? 0 : beta*y[row]);
}


// (v_m) x (u_n) = M(mxn) (vector outer product)
template <typename Type>
__global__ void gpu_outer(Type* c, int pitch, const Type* a, int m, const Type* b, int n)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (y < m && x < n) 
		c[y*pitch+x] = a[y] * b[x];
}


// M_z = [v x u]_z (z vector outer product)
template <typename Type>
__global__ void gpu_outer3d(Type* out, int h, int zstride, int pitch,
							const Type* v, int m, int vzstride, const Type* u, int n, int uzstride)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (z < h && y < m && x < n) 
		out[z*zstride + y*pitch + x] = v[z*vzstride+y] * u[z*uzstride+x];
}


// gemm : matrix(m,p) = matrix(m,n) * matrix(n,p)
// C = A.B
template <typename Type>
__global__ void gpu_gemm(const Type* a, const Type* b, Type* c, int m, int n, int p, int apitch, int bpitch, int cpitch) 
{
	// compute each thread's global row and column index
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	// statically allocated shared memory
	__shared__ Type s_a[TILES][TILES];
	__shared__ Type s_b[TILES][TILES];

	Type sum = Type(0);

	// sweep tile across matrix
	for (int j=0; j<n; j+=blockDim.x) {
		// load elements for this tile and wait for both tiles to be loaded
		s_a[threadIdx.y][threadIdx.x] = a[row*apitch + j + threadIdx.x];
		s_b[threadIdx.y][threadIdx.x] = b[(j+threadIdx.y)*bpitch + col];
		__syncthreads();

		// do matrix multiplication on the tile and wait for all threads to finish
		for (int k=0; k<blockDim.x; k++)
			sum += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
		__syncthreads();
	}

	// write back results
	if (row < m && col < p) 
		c[row*cpitch + col] = sum;
}


// gramm : A(n,n) = A(m,n).T * A(m,n) = A(n,m) * A(m,n)
// https://en.wikipedia.org/wiki/In-place_matrix_transposition
template <typename Type>
__global__ void gpu_gram(const Type* a, int m, int n, int apitch, Type* c, int cpitch) 
{
	// compute each thread's global row and column index
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	// statically allocated shared memory
	__shared__ Type s_a[TILES][TILES];
	__shared__ Type s_b[TILES][TILES];

	Type sum = Type(0);

	// sweep tile across matrix
	for (int j=0; j<n; j+=blockDim.x) {
		// load elements for this tile. in fact we multiply column 'row' x column 'col'
		s_a[threadIdx.y][threadIdx.x] = a[j*apitch + threadIdx.x*apitch + row];
		s_b[threadIdx.y][threadIdx.x] = a[j*apitch + threadIdx.y*apitch + col];
		__syncthreads();

		// do matrix multiplication on the tile
		for (int k=0; k<blockDim.x; k++) 
			sum += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
		__syncthreads();
	}

	// write back results
	if (row < n && col < n) 
		c[row*cpitch + col] = sum;
}


template <typename Type>
__global__ void gpu_gemt(const Type* a, int m, int n, int apitch, Type* t, int tpitch) 
{
	__shared__ Type tile[TILES][TILES+1];
	int x = blockIdx.x*TILES + threadIdx.x;
	int y = blockIdx.y*TILES + threadIdx.y;

	// load matrix into tile, every thread loads 4 elements into tile.
	for (int i=0; i<TILES; i+=BLOCKS){
		if (x < n  && (y+i) < m)
			tile[threadIdx.y+i][threadIdx.x] = a[(y+i)*apitch + x];
	}
	__syncthreads();

	x = blockIdx.y*TILES + threadIdx.x; 
	y = blockIdx.x*TILES + threadIdx.y;
	for (int i=0; i<TILES; i+=BLOCKS){
		if (x < m && (y+i) < n)
			t[(y+i)*tpitch + x] = tile[threadIdx.x][threadIdx.y + i];
	}
}


};     // namespace umml

#endif // UMML_BLAS_CUDA
