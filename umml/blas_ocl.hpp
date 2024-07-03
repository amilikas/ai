#ifndef UMML_BLAS_OCL_INCLUDED
#define UMML_BLAS_OCL_INCLUDED

/*
 CUDA <-> OpenCL

 threadIdx.x                         = get_local_id(0)
 blockDim.x                          = get_local_size(0)
 gridDim.x*blockDim.x                = get_global_size(0)
 blockIdx.x*blockDim.x + threadIdx.x = get_global_id(0)
 blockIdx.x                          = get_group_id(0)
 gridDim.x                           = get_num_groups(0)
*/


namespace umml {


// reciprocal
// y = α(1/x)
std::string ocl_reciprocal_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* x, int n, __global __type__* y) 
{
	const int i = get_global_id(0);
	if (i < n) y[i] = (x[i] != 0 ? alpha/x[i] : 0);
}
)";


// axpy : vector(n) += scalar*vector(n)
// y = αx + y
std::string ocl_axpy_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* x, int n, __global __type__* y) 
{
	const int i = get_global_id(0);
	if (i < n) y[i] = alpha*x[i] + y[i];
}
)";


// axpby : vector(n) = scalar*vector(n) + scalar*vector(n)
// z = αx + βy
std::string ocl_zaxpby_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* x, int n, __type__ beta, __global __type__* y, __global __type__* z) 
{
	const int i = get_global_id(0);
	if (i < n) z[i] = alpha*x[i] + beta*y[i];
}
)";


// axpym : matrix(m,n) += scalar*matrix(m,n)
// Y = αX + Y
std::string ocl_axpym_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* x, int m, int n, int xpitch, __global __type__* y, int ypitch) 
{
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	if (row >= m || col >= n) return;
	y[row*ypitch + col] += alpha*x[row*xpitch + col];
}
)";


// zaxpbym : matrix(m,n) = scalar*matrix(m,n) + scalar*matrix(m,n)
// Z = αX + βY
std::string ocl_zaxpbym_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* x, int m, int n, int xpitch, 
					   __type__ beta, __global __type__* y, int ypitch, __global __type__* z, int zpitch) 
{
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	if (row >= m || col >= n) return;
	z[row*zpitch + col] = alpha*x[row*xpitch + col] + beta*y[row*ypitch + col];
}
)";


// dot, partial
// dot = αx.βy
/* DOES NOT WORK
std::string ocl_dotp_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* x, int n, __type__ beta, __global __type__* y, __global __type__* result) 
{
	// Define local memory buffer for partial sums
	__local __type__ localSum[16]; // Use a local size matching the local work-group size

	// Initialize localSum with zeros
	localSum[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Compute partial dot products and store in localSum
	for (unsigned int i = get_local_id(0); i < n; i += get_local_size(0)) {
		if (i < n) localSum[get_local_id(0)] += alpha*x[i] * beta*y[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform the reduction within the work-group
	for (unsigned int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
		if (get_local_id(0) < stride && (get_local_id(0) + stride) < n) {
			localSum[get_local_id(0)] += localSum[get_local_id(0) + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the final result in global memory
	if (get_local_id(0) == 0) {
		result[0] = localSum[0];
	}
}
)";
*/

std::string ocl_dotp_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* x, int n, __type__ beta, __global __type__* y, __global __type__* result) 
{
	__local __type__ warp[WARP_SIZE];
	
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);

	__type__ acc = 0;
	for (int i = global_id; i < n; i += get_global_size(0)) 
		acc += alpha*x[i] * beta*y[i];
	warp[local_id] = acc;
	barrier(CLK_LOCAL_MEM_FENCE);

	// reduce tile array within the workgroup
	for (int s = WARP_SIZE/2; s > 0; s >>= 1) {
		if (local_id < s) warp[local_id] += warp[local_id + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// store the final partial sum in result
	if (local_id == 0) result[get_group_id(0)] = warp[0];
}
)";


// outer : matrix(m,n) = vector(m) x vector(n)
// M = v x u
std::string ocl_outer_code = R"(
__kernel void __name__(__global __type__* out, int pitch, __global __type__* v, int m, __global __type__* u, int n) 
{ 
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (y < m && x < n) 
		out[y*pitch + x] = v[y] * u[x];
}
)";


// M_z = [v x u]_z (z vector outer product)
std::string ocl_outer3d_code = R"(
__kernel void __name__(__global __type__* out, int h, int zstride, int pitch,
							__global __type__* v, int m, int vzstride, __global __type__* u, int n, int uzstride)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	if (z < h && y < m && x < n) 
		out[z*zstride + y*pitch + x] = v[z*vzstride+y] * u[z*uzstride+x];
}
)";


// gemv : vector(m) = scalar*matrix(m,n)*vector(n) + scalar*vector(m)
// y = αA*x + βy
// NOT WORKING
std::string ocl_gemv1_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* a, int m, int n, int pitch, 
					   __global __type__* x, __type__ beta, __global __type__* y) 
{
	__local __type__ tile[16];

	// Each work-group handles as many matrix rows as necessary
	for (uint i = get_group_id(0); i < m; i += get_num_groups(0)) {

		// Row pointer
		const __global __type__* row = a + i*pitch;
		
		// Each work-item accumulates as many products as necessary
		// into local variable "sum"
		__type__ sum = 0;
		for (uint j = get_local_id(0); j < n; j += get_local_size(0))
			sum += alpha*row[j] * x[j];

		// Each partial dot product is stored in shared memory
		tile[get_local_id(0)] = sum;
		
	 // Thread local ID within a warp
	  uint id = get_local_id(0) & (32 - 1); 

	  // Each warp reduces 64 consecutive elements
	  __type__ warpResult = 0;
	  if (get_local_id(0) < get_local_size(0)/2 )
	  {
		  volatile __local __type__* p = tile + 2 * get_local_id(0) - id;
		  p[0] += p[32];
		  p[0] += p[16];
		  p[0] += p[8];
		  p[0] += p[4];
		  p[0] += p[2];
		  p[0] += p[1];
		  warpResult = p[0];
	  }

	  // Synchronize to make sure each warp is done reading
	  // partialDotProduct before it is overwritten in the next step
	  barrier(CLK_LOCAL_MEM_FENCE);

	  // The first thread of each warp stores the result of the reduction
	  // at the beginning of partialDotProduct
	  if (id == 0)
		 tile[get_local_id(0) / 32] = warpResult;

	  // Synchronize to make sure each warp is done writing to
	  // partialDotProduct before it is read in the next step
	  barrier(CLK_LOCAL_MEM_FENCE);

	  // Number of remaining elements after the first reduction
	  uint size = get_local_size(0) / (2 * 32);

	  // get_local_size(0) is less or equal to 512 on NVIDIA GPUs, so
	  // only a single warp is needed for the following last reduction
	  // step
	  if (get_local_id(0) < size / 2) {
		 volatile __local __type__* p = tile + get_local_id(0);
		 if (size >= 8)
			p[0] += p[4];
		 if (size >= 4)
			p[0] += p[2];
		 if (size >= 2)
			p[0] += p[1];
	  }

		// Write the result of the reduction to global memory
		if (get_local_id(0)==0) y[i] = tile[0] + (beta==0 ? 0 : beta*y[i]);

		// Synchronize to make sure the first work-item is done with
		// reading partialDotProduct
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
)";

// gemv : vector(m) = scalar*matrix(m,n)*vector(n) + scalar*vector(m)
// http://www.bealto.com/gpu-gemv_v2.html
std::string ocl_gemv2_code = R"(
__kernel void __name__(__type__ alpha, __global __type__* a, int m, int n, int pitch, 
					   __global __type__* x, __type__ beta, __global __type__* y) 
{
	int col = get_global_id(0); // blockIdx.x*blockDim.x + threadIdx.x;
	int row = get_global_id(1); // blockIdx.y*blockDim.y + threadIdx.y;

	__local __type__ tile[TILE_SIZE];

	// compute partial dot product
	__type__ acc = 0;
	for (int k=col; k<n; k += get_global_size(0))
		acc += alpha*a[row*pitch+k] * x[k];

	// each thread stores its partial sum in 'tile'
	tile[get_local_id(1)+get_local_size(1)*get_local_id(0)] = acc;
	barrier(CLK_LOCAL_MEM_FENCE);

	// reduce sums in log2(cols) steps
	int cols = get_local_size(0);
	while (cols > 1) {
		int lx = get_local_id(0);
		int ly = get_local_id(1);
		int lrows = get_local_size(1);
		cols /= 2;
		if (lx < cols) tile[ly+lrows*lx] += tile[ly+lrows*(lx+cols)];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (get_local_id(0)==0) y[row] = tile[get_local_id(1)] + (beta==0 ? 0 : beta*y[row]);
}
)";


// gemm : matrix(m,p) = matrix(m,n) * matrix(n,p)
// C = A*B
// https://cnugteren.github.io/tutorial/pages/page5.html
std::string ocl_gemm1_code = R"(
__kernel void __name__(__global __type__* a, __global __type__* b, __global __type__* c, 
					   int m, int n, int p, int apitch, int bpitch, int cpitch) 
{
	const int col = get_global_id(0); // col of C (0..p)
	const int row = get_global_id(1); // row of C (0..m)

	__type__ acc = 0;
	for (int k=0; k<n; k++) {
		acc += a[row*apitch + k] * b[k*bpitch + col];
	}

	if (row < m && col < p) c[row*cpitch + col] = acc;
}
)";

/*
std::string ocl_gemm2_code = R"(
__kernel void __name__(__global __type__* a, __global __type__* b, __global __type__* c, 
					   int m, int n, int p, int apitch, int bpitch, int cpitch) 
{
	const int lx = get_local_id(0); // local col (0..TILE_SIZE)
	const int ly = get_local_id(1); // local row (0..TILE_SIZE)
	const int x  = TILE_SIZE*get_group_id(0) + lx; // col of C (0..P)
	const int y  = TILE_SIZE*get_group_id(1) + ly; // row of C (0..M)

	// local memory to fit a tile of TILE_SIZE*TILE_SIZE elements of A and B
	__local __type__ s_a[TILE_SIZE][TILE_SIZE];
	__local __type__ s_b[TILE_SIZE][TILE_SIZE];
 
	// initialise the accumulation register
	__type__ acc = 0;
	
	// loop over all tiles
	int numTiles = apitch/TILE_SIZE;
	for (int t=0; t<numTiles; t++) {
		// Load one tile of A and B into local memory
		int ty = TILE_SIZE*t + ly;
		int tx = TILE_SIZE*t + lx;
		s_a[ly][lx] = a[tx + y*apitch];
		s_b[ly][lx] = b[x + ty*bpitch];
		barrier(CLK_LOCAL_MEM_FENCE);
 
		// Perform the computation for a single tile
		for (int k=0; k<TILE_SIZE; k++) acc += s_a[ly][k] * s_b[k][lx];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
 
	// Store the final result in C
	if (x < p && y < m) c[x + y*cpitch] = acc;
}
)";
*/

std::string ocl_gemm2_code = R"(
__kernel void __name__(__global __type__* a, __global __type__* b, __global __type__* c, 
					   int m, int n, int p, int apitch, int bpitch, int cpitch) 
{
	const int lx = get_local_id(0); // local col (0..TILE_SIZE)
	const int ly = get_local_id(1); // local row (0..TILE_SIZE)
	const int x  = TILE_SIZE*get_group_id(0) + lx; // col of C (0..P)
	const int y  = TILE_SIZE*get_group_id(1) + ly; // row of C (0..M)

	// local memory to fit a tile of TILE_SIZE*TILE_SIZE elements of A and B
	__local __type__ s_a[TILE_SIZE][TILE_SIZE];
	__local __type__ s_b[TILE_SIZE][TILE_SIZE];
 
	// initialise the accumulation register
	__type__ acc = 0;
	
	// loop over all tiles
	int numTiles = (n+TILE_SIZE-1)/TILE_SIZE;
	for (int t=0; t<numTiles; t++) {
		// load one tile of A and B into local memory
		const int tx = TILE_SIZE*t + lx;
		const int ty = TILE_SIZE*t + ly;
		if (tx < n && y < m) s_a[ly][lx] = a[tx + y*apitch];
		else s_a[ly][lx] = 0;
		if (x < p && ty < n) s_b[ly][lx] = b[x + ty*bpitch];
		else s_b[ly][lx] = 0;
		barrier(CLK_LOCAL_MEM_FENCE);
 
		// perform the computation for a single tile
		for (int k=0; k<TILE_SIZE; k++) acc += s_a[ly][k] * s_b[k][lx];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
 
	// store the final result in C
	if (x < p && y < m) c[x + y*cpitch] = acc;
}
)";

// gramm : A(n,n) = A(m,n).T * A(m,n) = A(n,m) * A(m,n)
// https://en.wikipedia.org/wiki/In-place_matrix_transposition
std::string ocl_gram_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, __global __type__* c, int cpitch) 
{
	const int x  = get_global_id(0);
	const int y  = get_global_id(1);
	const int lx = get_local_id(0);
	const int ly = get_local_id(1);

	__local __type__ s_a[TILE_SIZE][TILE_SIZE];
	__local __type__ s_b[TILE_SIZE][TILE_SIZE];

	__type__ acc = 0;

	// sweep tile across matrix
	for (int j=0; j<n; j+=get_local_size(0)) {
		// load elements for this tile. in fact we multiply column 'y' x column 'x'
		s_a[ly][lx] = a[j*apitch + lx*apitch + y];
		s_b[ly][lx] = a[j*apitch + ly*apitch + x];
		barrier(CLK_LOCAL_MEM_FENCE);

		// do matrix multiplication on the tile
		for (int k=0; k<get_local_size(0); k++) 
			acc += s_a[ly][k] * s_b[k][lx];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write back results
	if (x < n && y < n) c[y*cpitch + x] = acc;
}
)";

// tiled matrix transpose (works ok)
std::string ocl_gemt_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, __global __type__* t, int tpitch) 
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	__local __type__ tile[TILE_SIZE*(TILE_SIZE+1)];
	if (x < n && y < m) tile[get_local_id(1)*(TILE_SIZE+1)+get_local_id(0)] = a[y*apitch + x];
	barrier(CLK_LOCAL_MEM_FENCE);

	// write the transposed matrix tile to global memory
	x = get_group_id(1)*TILE_SIZE + get_local_id(0);
	y = get_group_id(0)*TILE_SIZE + get_local_id(1);
	if (x < m && y < n) t[y*tpitch + x] = tile[get_local_id(0)*(TILE_SIZE+1)+get_local_id(1)];
}
)";

// tiled batched matrix multiplication
// https://github.com/salehjg/batch-matmul-cuda/blob/master/src/batch-matmul-cuda.cu
std::string ocl_bmm_code = R"(
__kernel void __name__(__global __type__* a, __global __type__* b, __global __type__* c, 
					   int bs, int m, int n, int p, 
					   int apitch, int bpitch, int cpitch, int astride, int bstride, int cstride,
					   __local __type__* shmem) 
{
	//int dim1A=m, int dim2A=n,
	//int dim1B=n, int dim2B=p,
	//int dim1C=m, int dim2C=p)

	const int TS = 6;
	//extern __shared __type__* shmem;

	const int len_sub = TS*n;
	const unsigned long
		len_A = bs*astride,
		len_B = bs*bstride,
		len_C = bs*cstride;

	const int x  = get_group_id(0); // blockIdx.x : c[z].x
	const int y  = get_group_id(1); // blockIdx.y : c[z].y
	const int z  = get_group_id(2); // blockIdx.z : z = batch index
	const int lx = get_local_id(0); // threadIdx.x
	const int ly = get_local_id(1); // threadIdx.y;

	unsigned int  c_pos_x, c_pos_y;
	c_pos_x = x*TS + lx;
	c_pos_y = y*TS + ly;

	unsigned long gidx1,gidx2;
	unsigned int _d1,_d2;

	unsigned long offsetA = y*TS*n;
	unsigned long offsetB = x*TS; //first row (d1=0)

	// Load sub matrices from global memory into shared memory

	unsigned long idxA, idxB;
	idxA = ly*TS + lx;
	idxB = ly*TS + lx;
	while (idxA < len_sub) {
		gidx1 = offsetA + idxA;
		if (idxA < len_sub && gidx1 < len_A) shmem[idxA] = a[z*astride + gidx1];
		else shmem[idxA] = 0;
		idxA += TS*TS;
	}
	while (idxB < len_sub) {
		//gidx2 = offsetB + (x*TS)*p + (idxB % p);
		_d2 = idxB % TS;
		_d1 = idxB / TS;
		gidx2 = offsetB + _d1*p + _d2;
		if (idxB < len_sub && _d1<n && _d2<p) shmem[len_sub+idxB] = b[z*bstride + gidx2];
		else shmem[len_sub+idxB] = 0;
		idxB += TS*TS;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Multiply and add each result to produce output element of current thread in the thread block.
	if (c_pos_x<p && c_pos_y<m) {
		unsigned long idx = ly*TS + lx;
		__type__ output_element = 0;

		// dim2A=dim1B is common equal dimension of 2 matrices  --- block-stride loop
		for (int k = 0; k < n; k++) {
			output_element += shmem[ly*n+k] * shmem[len_sub + k*TS + lx];
		}

		// TODO: Check matC index to not to exceed the len of matC!
		c[z*cstride + c_pos_y*cpitch + c_pos_x] = output_element;
	}
}
)";


/* not so fast
std::string ocl_gemt_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, __global __type__* t, int tpitch) 
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	__local __type__ tile[TILE_SIZE][TILE_SIZE+1];
	#pragma unroll
	for (int i=0; i<TILE_SIZE; i+=BLOCK_SIZE) {
		if (x < n  && (y+i) < m) tile[get_local_id(1)+i][get_local_id(0)] = a[(y+i)*apitch + x];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	x = get_group_id(1)*TILE_SIZE + get_local_id(0); 
	y = get_group_id(0)*TILE_SIZE + get_local_id(1);
	#pragma unroll
	for (int i=0; i<TILE_SIZE; i+=BLOCK_SIZE) {
		if (x < m && (y+i) < n) t[(y+i)*tpitch + x] = tile[get_local_id(0)][get_local_id(1) + i];
	}
}
)";
*/

};     // namespace umml

#endif // UMML_BLAS_OCL_INCLUDED
