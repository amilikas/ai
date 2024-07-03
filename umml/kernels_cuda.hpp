#ifndef UMML_KERNELS_CUDA_INCLUDED
#define UMML_KERNELS_CUDA_INCLUDED

#include <math.h>
#include "blas_cuda.hpp"


namespace umml {


__device__ double datomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
    	assumed = old;
    	old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    	// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ float fatomicAdd(float* address, float val)
{
	unsigned int* address_as_u = (unsigned int*)address;
	unsigned int  old = *address_as_u, assumed;
	do {
    	assumed = old;
    	old = atomicCAS(address_as_u, assumed, __float_as_uint(val + __uint_as_float(assumed)));
    	// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
	return __uint_as_float(old);
}


// copy a 2d region from src to dst
template <typename Type>
__global__ void gpu_copy2d(const Type* src, int spitch, Type* dst, int dpitch, 
						   int sy, int sx, int dy, int dx, int ylen, int xlen) 
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x < dx+xlen && y < dy+ylen && x < sx+xlen && y < sy+ylen)
		dst[(y+dy)*dpitch + (x+dx)] = src[(y+sy)*spitch + (x+sx)];
}

// copy a 2d region from src to dst
template <typename Type>
__global__ void gpu_copy3d(int zdim, const Type* src, int spitch, int szstride, Type* dst, int dpitch, int dzstride, 
						   int sy, int sx, int dy, int dx, int ylen, int xlen) 
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (x < dx+xlen && y < dy+ylen && x < sx+xlen && y < sy+ylen && z < zdim)
		dst[z*dzstride + (y+dy)*dpitch + (x+dx)] = src[z*szstride + (y+sx)*spitch + (x+sx)];
}


// y = α
template <typename Type>
__global__ void gpu_vecset(Type* y, int n, Type alpha)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = alpha;
}


// y = αx
template <typename Type>
__global__ void gpu_vecset(Type* y, const Type* x, int n, Type alpha)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = alpha*x[i];
}


// y = x^2
template <typename Type>
__global__ void gpu_yx2(Type* y, const Type* x, int n)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = x[i]*x[i];
}


// y += α
template <typename Type>
__global__ void gpu_vecplus(Type* y, int n, Type alpha)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] += alpha;
}


// Hadamard (vector)
template <typename Type>
__global__ void gpu_hadamard(const Type* a, const Type* b, Type* c, int n)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x < n) c[x] = a[x] * b[x];
}


// s = Σv[i]
template <typename Type>
__global__ void gpu_sve(const Type* in, int n, Type* result) 
{
	Type acc = 0;
	for (int i=0; i<n; ++i) acc += in[i];
	*result = acc;
}


// s[g] = Σv[i], partial (blocking)
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <typename Type>
__global__ void gpu_svep(const Type* in, int n, Type* partial) 
{
	__shared__ Type sdata[THREADS];
	// each thread loads one element from global to shared mem
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) sdata[tid] = in[i];
	else sdata[tid] = 0;
	__syncthreads();
	// do reduction in shared mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// s[g] = Σ(v[i]+α)^2, partial (blocking)
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <typename Type>
__global__ void gpu_svep2(const Type* in, int n, Type alpha, Type* partial) 
{
	__shared__ Type sdata[THREADS];
	// each thread loads one element from global to shared mem
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) sdata[tid] = (in[i]+alpha)*(in[i]+alpha);
	else sdata[tid] = 0;
	__syncthreads();
	// do reduction in shared mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) partial[blockIdx.x] = sdata[0];
}


template <typename Type>
__global__ void gpu_cnteq(const Type* a, int n, const Type* b, int* partial) 
{
	__shared__ int sdata[THREADS];
	// each thread loads one element from global to shared mem
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n && a[i]==b[i]) sdata[tid] = 1;
	else sdata[tid] = 0;
	__syncthreads();
	// do reduction in shared mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) partial[blockIdx.x] = sdata[0];
}

template <typename Type>
__global__ void gpu_eucl2(const Type* a, int n, const Type* b, Type* partial) 
{
	__shared__ Type sdata[THREADS];
	// each thread loads one element from global to shared mem
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) sdata[tid] = (a[i]-b[i])*(a[i]-b[i]);
	else sdata[tid] = 0;
	__syncthreads();
	// do reduction in shared mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) partial[blockIdx.x] = sdata[0];
}

template <typename Type>
__global__ void gpu_manh(const Type* a, int n, const Type* b, Type* partial) 
{
	__shared__ Type sdata[THREADS];
	// each thread loads one element from global to shared mem
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) sdata[tid] = fabs(a[i]-b[i]);
	else sdata[tid] = 0;
	__syncthreads();
	// do reduction in shared mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// vector argmaxp, partial (with blocking)
template <typename Type>
__global__ void gpu_vargmaxp(const Type* in, int n, Type* partial, int* partialIdx) 
{
	__shared__ Type local_max[THREADS];
	__shared__ int local_imax[THREADS];
	
	int global_id = blockIdx.x*blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;
	
	Type mv = (Type)-INFINITY;
	int mi = -1;
	for (int i = global_id; i < n; i += gridDim.x*blockDim.x) {
	    if (i < n && in[i] > mv) {
			mv = in[i];
			mi = i;
		}
	}
	
	local_max[local_id] = mv;
	local_imax[local_id] = mi;
	__syncthreads();
	
	// Reduce localMax and localMaxIndex arrays within the workgroup
	for (int s = blockDim.x/2; s > 0; s >>= 1) {
	    if (local_id < s) {
	        if (local_max[local_id + s] > local_max[local_id]) {
	            local_max[local_id] = local_max[local_id + s];
	            local_imax[local_id] = local_imax[local_id + s];
	        }
	    }
	    __syncthreads();
	}
	
	// Store the final max value and index in output and outputIndex
	if (local_id == 0) {
	    partial[blockIdx.x] = local_max[0];
	    partialIdx[blockIdx.x] = local_imax[0];
	}
}

// vector argmax
template <typename Type>
__global__ void gpu_vargmax(const Type* in, const int* idcs, int n, int* result) 
{
	int imax;
	Type m = (Type)-INFINITY;
	if (idcs==NULL) {
		for (int i=0; i<n; ++i) if (in[i] > m) { m = in[i]; imax = i; }
	} else {
		for (int i=0; i<n; ++i)	if (in[i] > m) { m = in[i];	imax = idcs[i];	}
	}
	*result = imax;
}

// vector argminp, partial (with blocking)
template <typename Type>
__global__ void gpu_vargminp(const Type* in, int n, Type* partial, int* partialIdx) 
{
	__shared__ Type local_min[THREADS];
	__shared__ int local_imin[THREADS];
	
	int global_id = blockIdx.x*blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;
	
	Type mv = (Type)INFINITY;
	int mi = -1;
	for (int i = global_id; i < n; i += gridDim.x*blockDim.x) {
	    if (i < n && in[i] < mv) {
			mv = in[i];
			mi = i;
		}
	}
	
	local_min[local_id] = mv;
	local_imin[local_id] = mi;
	__syncthreads();
	
	// Reduce localMax and localMaxIndex arrays within the workgroup
	for (int s = blockDim.x/2; s > 0; s >>= 1) {
	    if (local_id < s) {
	        if (local_min[local_id + s] < local_min[local_id]) {
	            local_min[local_id] = local_min[local_id + s];
	            local_imin[local_id] = local_imin[local_id + s];
	        }
	    }
	    __syncthreads();
	}
	
	// Store the final max value and index in output and outputIndex
	if (local_id == 0) {
	    partial[blockIdx.x] = local_min[0];
	    partialIdx[blockIdx.x] = local_imin[0];
	}
}

// vector argmax
template <typename Type>
__global__ void gpu_vargmin(const Type* in, const int* idcs, int n, int* result) 
{
	int imax;
	Type m = (Type)INFINITY;
	if (idcs==NULL) {
		for (int i=0; i<n; ++i) if (in[i] < m) { m = in[i]; imin = i; }
	} else {
		for (int i=0; i<n; ++i)	if (in[i] < m) { m = in[i];	imin = idcs[i];	}
	}
	*result = imin;
}


// v = f(v+αx)
template <typename Type>
__global__ void gpu_apply_function1d(Type* c, int n, int f, Type alpha, const Type* x)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		c[i] = umml::gpu_function(c[i] + (alpha==0 ? 0 : alpha*x[i]), f);
}


/*
 GPU memory is always padded to THREADS
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 m11  m12  m13  ---
 m21  m22  m23  ---
 m32  m32  m33  ---
 ---  ---  ---  ---

 pitch (pad) = 4
*/

template <typename Type>
__global__ void gpu_matset(Type* c, int nrows, int ncols, int pitch, Type val)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols)
		c[y*pitch+x] = val;
}


template <typename Type>
__global__ void gpu_matset(Type* c, int nrows, int ncols, int pitch, Type alpha, const Type* a, int apitch)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols) 
		c[y*pitch+x] = alpha*a[y*apitch+x];
}

// M += val
template <typename Type>
__global__ void gpu_matplus(Type* c, int nrows, int ncols, int pitch, Type val)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols)
		c[y*pitch+x] += val;
}

// Y = α*x + Y
template <typename Type>
__global__ void gpu_matplus(Type* c, int nrows, int ncols, int pitch, Type alpha, const Type* v, int axis)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols) 
		c[y*pitch+x] += alpha*v[(axis==0 ? x:y)];
}

// M *= val
template <typename Type>
__global__ void gpu_matmul(Type* c, int nrows, int ncols, int pitch, Type val)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols)
		c[y*pitch+x] *= val;
}

// C = A*B (Hadamard)
template <typename Type>
__global__ void gpu_hadamard(const Type* a, const Type* b, Type* c, int m, int n, int pitch)
{
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y >= m || x >= n) return;
	c[y*pitch+x] = a[y*pitch+x] * b[y*pitch+x];
}

// M *= v (element-wise)
template <typename Type>
__global__ void gpu_mprodv(Type* c, int nrows, int ncols, int pitch, const Type* v, int axis)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols)
		c[y*pitch+x] *= v[(axis==0 ? x:y)];
}

// M /= v
template <typename Type>
__global__ void gpu_mdivv(Type* c, int nrows, int ncols, int pitch, const Type* v, int axis)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols)
		c[y*pitch+x] /= v[(axis==0 ? x:y)];
}

// M = f(β*(M+αx))
template <typename Type>
__global__ void gpu_apply_function2d(Type* c, int nrows, int ncols, int pitch, int f, Type beta, Type alpha, const Type* v, int axis)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y < nrows && x < ncols) 
		c[y*pitch+x] = umml::gpu_function(beta*(c[y*pitch+x] + (alpha==0 ? 0 : alpha*v[(axis==0 ? x:y)])), f);
}

// matrix argmax per row
template <typename Type>
__global__ void gpu_margmax(const Type* in, int m, int n, int pitch, int* rowidcs) 
{	
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	if (row >= m) return;

	Type mv = (Type)-INFINITY;
	int mi = -1;
	for (int j=0; j<n; ++j) {
	    Type val = in[row*pitch + j];
	    if (val > mv) {
	        mv = val;
	        mi = j;
	    }
	}

	rowidcs[row] = mi;
}

// matrix max per col/row
template <typename Type>
__global__ void gpu_matmax(const Type* a, int m, int n, int pitch, Type* out, int axis)
{	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	Type mv = (Type)-INFINITY;

	if (axis==0) {
		if (id >= n) return;
		for (int i=0; i<m; ++i) {
			Type val = a[i*pitch + id];
			if (val > mv) mv = val;
		}
	} else {
		if (id >= m) return;
		for (int j=0; j<n; ++j) {
			Type val = a[id*pitch + j];
			if (val > mv) mv = val;
		}
	}

	out[id] = mv;
}

// matrix argmin per row
template <typename Type>
__global__ void gpu_margmin(const Type* in, int m, int n, int pitch, int* rowidcs) 
{	
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	if (row >= m) return;

	Type mv = (Type)INFINITY;
	int mi = -1;
	for (int j=0; j<n; ++j) {
	    Type val = in[row*pitch + j];
	    if (val < mv) {
	        mv = val;
	        mi = j;
	    }
	}

	rowidcs[row] = mi;
}

// matrix max per col/row
template <typename Type>
__global__ void gpu_matmin(const Type* a, int m, int n, int pitch, Type* out, int axis)
{	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	Type mv = (Type)INFINITY;

	if (axis==0) {
		if (id >= n) return;
		for (int i=0; i<m; ++i) {
			Type val = a[i*pitch + id];
			if (val < mv) mv = val;
		}
	} else {
		if (id >= m) return;
		for (int j=0; j<n; ++j) {
			Type val = a[id*pitch + j];
			if (val < mv) mv = val;
		}
	}

	out[id] = mv;
}


/*
 CUDA <-> OpenCL

 threadIdx.x                         = get_local_id(0)
 blockDim.x                          = get_local_size(0)
 gridDim.x*blockDim.x                = get_global_size(0)
 blockIdx.x*blockDim.x + threadIdx.x = get_global_id(0)
 blockIdx.x                          = get_group_id(0)
 gridDim.x                           = get_num_groups(0)
*/

// sum of cube's slices
template <typename Type>
__global__ void gpu_sum3d(const Type* in, int h, int m, int n, int xpitch, int ypitch, Type* out)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	Type acc = 0;
	for (int z=0; z<h; ++z) acc += in[(z*ypitch+y)*xpitch+x];
	out[y*xpitch+x] = acc;
}

// reduction: sum of matrix rows (partial)
template <typename Type>
__global__ void gpu_smrp(const Type* a, int m, int n, int pitch, Type* rows)
{
	__shared__ Type warp[THREADS];

	// each thread loads one element from global to local mem
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int lx = threadIdx.x;
	const int workgroups = (n+THREADS-1) / THREADS;

	if (y < m && x < n) warp[lx] = a[y*pitch+x];
	else warp[lx] = 0;
	__syncthreads();

	// do reduction in local mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (lx == 0) rows[blockIdx.y*workgroups + blockIdx.x] = warp[0];
}

// reduction: sum of matrix rows squared (partial)
template <typename Type>
__global__ void gpu_smrp2(const Type* a, int m, int n, int pitch, Type alpha, Type* rows)
{
	__shared__ Type warp[THREADS];

	// each thread loads one element from global to local mem
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int lx = threadIdx.x;
	const int workgroups = (n+THREADS-1) / THREADS;

	if (y < m && x < n) warp[lx] = (a[y*pitch+x]+alpha)*(a[y*pitch+x]+alpha);
	else warp[lx] = 0;
	__syncthreads();

	// do reduction in local mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (lx == 0) rows[blockIdx.y*workgroups + blockIdx.x] = warp[0];
}

// reduction: sum of matrix rows (full)
template <typename Type>
__global__ void gpu_smr(const Type* a, int m, int n, int pitch, Type* rows)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < m) {
		Type acc = 0;
		for (int j=0; j<n; ++j) acc += a[i*pitch+j];
		rows[i] = acc;
	}
}

// reduction: sum of matrix columns (full)
template <typename Type>
__global__ void gpu_smc(const Type* a, int m, int n, int pitch, Type* cols)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if (j < n) {
		Type acc = 0;
		for (int i=0; i<m; ++i) acc += a[i*pitch+j];
		cols[j] = acc;
	}
}


template <typename Type>
__global__ void gpu_cnteq(const Type* a, int m, int n, int apitch, 
						  const Type* b, int bpitch, Type novalue, int* rows)
{
	__shared__ int warp[THREADS];

	// each thread loads one element from global to local mem
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int lx = threadIdx.x;
	const int workgroups = (n+THREADS-1) / THREADS;

	Type av = a[y*apitch+x];
	Type bv = b[y*bpitch+x];
	if (y < m && x < n && av != novalue && av == bv) warp[lx] = 1;
	else warp[lx] = 0;
	__syncthreads();

	// do reduction in local mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (lx == 0) rows[blockIdx.y*workgroups + blockIdx.x] = warp[0];
}


template <typename Type>
__global__ void gpu_eucl2(const Type* a, int m, int n, int apitch, 
						  const Type* b, int bpitch, Type* rows)
{
	__shared__ Type warp[THREADS];

	// each thread loads one element from global to local mem
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int lx = threadIdx.x;
	const int workgroups = (n+THREADS-1) / THREADS;

	if (y < m && x < n) {
		Type d = a[y*apitch+x] - b[y*bpitch+x];
		warp[lx] = d*d;
	} else warp[lx] = 0;
	__syncthreads();

	// do reduction in local mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (lx == 0) rows[blockIdx.y*workgroups + blockIdx.x] = warp[0];
}


template <typename Type>
__global__ void gpu_manh(const Type* a, int m, int n, int apitch, 
						 const Type* b, int bpitch, Type* rows)
{
	__shared__ Type warp[THREADS];

	// each thread loads one element from global to local mem
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int lx = threadIdx.x;
	const int workgroups = (n+THREADS-1) / THREADS;

	if (y < m && x < n) warp[lx] = fabs(a[y*apitch+x] - b[y*bpitch+x]);
	else warp[lx] = 0;
	__syncthreads();

	// do reduction in local mem
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (lx == 0) rows[blockIdx.y*workgroups + blockIdx.x] = warp[0];
}


// C = αN
template <typename Type>
__global__ void gpu_cubset(Type* dst, int c, int m, int n, int xpitch, int ypitch, 
						   Type alpha, const Type* src, int sxpitch, int sypitch) 
{ 
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (z < c && y < m && x < n)
		dst[z*xpitch*ypitch + y*xpitch + x] = alpha*src[z*sxpitch*sypitch + y*sxpitch + x]; 
}

// C = f(C)
template <typename Type>
__global__ void gpu_apply_function3d(Type* c, int h, int m, int n, int pitch, int zstride, int f)
{
 	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
 	int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (z < h && y < m && x < n) {
		int i = z*zstride+y*pitch+x;
		c[i] = umml::gpu_function(c[i], f);
	}
}


};     // namespace umml

#endif // UMML_KERNELS_CUDA_INCLUDED
