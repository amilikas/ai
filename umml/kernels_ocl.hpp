#ifndef UMML_KERNELS_OCL_INCLUDED
#define UMML_KERNELS_OCL_INCLUDED

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


// copy2d
std::string ocl_copy2d_code = R"(
__kernel void __name__(__global __type__* src, int spitch, __global __type__* dst, int dpitch, 
						int sy, int sx, int dy, int dx, int ylen, int xlen) 
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x < dx+xlen && y < dy+ylen && x < sx+xlen && y < sy+ylen)
		dst[(y+dy)*dpitch + (x+dx)] = src[(y+sy)*spitch + (x+sx)];
}
)";

// copy3d
std::string ocl_copy3d_code = R"(
__kernel void __name__(int zdim, __global __type__* src, int spitch, int szstride, 
					__global __type__* dst, int dpitch, int dzstride, int sy, int sx, int dy, int dx, int ylen, int xlen) 
{ 
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	if (x < dx+xlen && y < dy+ylen && x < sx+xlen && y < sy+ylen && z < zdim)
		dst[z*dzstride + (y+dy)*dpitch + (x+dx)] = src[z*szstride + (y+sx)*spitch + (x+sx)];
}
)";


// ya : vector(n) = scalar
// y = α
std::string ocl_yseta_code = R"(
__kernel void __name__(__global __type__* y, int n, __type__ alpha) 
{ 
	const int i = get_global_id(0);
	if (i < n) y[i] = alpha;
}
)";


// yax : vector(n) = scalar*vector(n)
// y = αx
std::string ocl_ysetax_code = R"(
__kernel void __name__(__global __type__* y, __global __type__* x, int n, __type__ alpha) 
{ 
	const int i = get_global_id(0); 
	if (i < n) y[i] = alpha*x[i]; 
}
)";


// yx2 : vector(n) = vector(n)^2
// y = x^2
std::string ocl_yx2_code = R"(
__kernel void __name__(__global __type__* y, __global __type__* x, int n) 
{ 
	const int i = get_global_id(0); 
	if (i < n) y[i] = x[i]*x[i];
}
)";


// ypa : vector(n) += scalar
// y += α
std::string ocl_ypa_code = R"(
__kernel void __name__(__global __type__* y, int n, __type__ alpha) 
{ 
	const int i = get_global_id(0); 
	if (i < n) y[i] += alpha; 
}
)";


// sve = Σv[i]
std::string ocl_sve_code = R"(
__kernel void __name__(__global __type__* a, int n, __global __type__* result)
{
	__type__ acc = 0;
	for (int i = 0; i < n; ++i) acc += a[i];
	*result = acc;
}
)";



/*
 CUDA <-> OpenCL

 threadIdx.x                         = get_local_id(0)
 blockDim.x                          = get_local_size(0)
 gridDim.x*blockDim.x                = get_global_size(0)
 blockIdx.x*blockDim.x + threadIdx.x = get_global_id(0)
 blockIdx.x                          = get_group_id(0)
 gridDim.x                           = get_num_groups(0)
*/
std::string ocl_svep_code = R"(
__kernel void __name__(__global __type__* a, int n, __global __type__* partial)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	int tid = get_local_id(0);
	int i = get_global_id(0);
	warp[tid] = (i < n ? a[i] : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (tid < s) warp[tid] += warp[tid + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (tid == 0) partial[get_group_id(0)] = warp[0];
}
)";

std::string ocl_svep2_code = R"(
__kernel void __name__(__global __type__* a, int n, __type__ alpha, __global __type__* partial)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	int tid = get_local_id(0);
	int i = get_global_id(0);
	warp[tid] = (i < n ? (a[i]+alpha)*(a[i]+alpha) : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (tid < s) warp[tid] += warp[tid + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (tid == 0) partial[get_group_id(0)] = warp[0];
}
)";

// count equal (partial, needs sve)
std::string ocl_vcnteq_code = R"(
__kernel void __name__(__global __type__* a, int n, __global __type__* b, __global int* partial)
{
	__local int warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	int tid = get_local_id(0);
	int i = get_global_id(0);
	warp[tid] = (i < n && a[i]==b[i] ? 1 : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (tid < s) warp[tid] += warp[tid + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (tid == 0) partial[get_group_id(0)] = warp[0];
}
)";

// euclidean distance squared (partial, needs sve)
std::string ocl_veucl2_code = R"(
__kernel void __name__(__global __type__* a, int n, __global __type__* b, __global __type__* partial)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	int tid = get_local_id(0);
	int i = get_global_id(0);
	if (i < n) {
		__type__ d = a[i] - b[i];
		warp[tid] = d*d;
	} else warp[tid] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (tid < s) warp[tid] += warp[tid + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (tid == 0) partial[get_group_id(0)] = warp[0];
}
)";

// manhattan distance (partial, needs sve)
std::string ocl_vmanh_code = R"(
__kernel void __name__(__global __type__* a, int n, __global __type__* b, __global __type__* partial)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	int tid = get_local_id(0);
	int i = get_global_id(0);
	warp[tid] = (i < n ? fabs(a[i]-b[i]) : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (tid < s) warp[tid] += warp[tid + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (tid == 0) partial[get_group_id(0)] = warp[0];
}
)";

// vector argmaxp, partial (with blocking)
std::string ocl_vargmaxp_code = R"(
__kernel void __name__(__global __type__* a, int n, __global __type__* partial, __global int* partialIdx)
{
	__local __type__ local_max[WARP_SIZE];
	__local int local_imax[WARP_SIZE];
	
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	
	__type__ mv = -INFINITY;
	int mi = -1;
	for (int i = global_id; i < n; i += get_global_size(0)) {
	    if (i < n && a[i] > mv) {
	        mv = a[i];
	        mi = i;
	    }
	}

	if (global_id < n) {
		local_max[local_id] = mv;
		local_imax[local_id] = mi;
	} else {
		local_max[local_id] = -INFINITY;
		local_imax[local_id] = -1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Reduce localMax and localMaxIndex arrays within the workgroup
	for (int s = WARP_SIZE/2; s > 0; s >>= 1) {
	    if (local_id < s) {
	        if (local_max[local_id + s] > local_max[local_id]) {
	            local_max[local_id] = local_max[local_id + s];
	            local_imax[local_id] = local_imax[local_id + s];
	        }
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// Store the final max value and index in output and outputIndex
	if (local_id == 0) {
	    partial[get_group_id(0)] = local_max[0];
	    partialIdx[get_group_id(0)] = local_imax[0];
	}
}
)";

// vector argmax
std::string ocl_vargmax_code = R"(
__kernel void __name__(__global __type__* a, __global int* ai, int n, __global int* result)
{
	int imax;
	__type__ m = -INFINITY;
	if (ai==NULL) {
		for (int i=0; i<n; ++i) if (a[i] > m) {	m = a[i]; imax = i;	}
	} else {
		for (int i=0; i<n; ++i) if (a[i] > m) {	m = a[i]; imax = ai[i]; }
	}
	*result = imax;
}
)";










// vector argmaxp, partial (with blocking)
std::string ocl_vargminp_code = R"(
__kernel void __name__(__global __type__* a, int n, __global __type__* partial, __global int* partialIdx)
{
	__local __type__ local_min[WARP_SIZE];
	__local int local_imin[WARP_SIZE];
	
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	
	__type__ mv = INFINITY;
	int mi = -1;
	for (int i = global_id; i < n; i += get_global_size(0)) {
	    if (i < n && a[i] < mv) {
	        mv = a[i];
	        mi = i;
	    }
	}

	if (global_id < n) {
		local_min[local_id] = mv;
		local_imin[local_id] = mi;
	} else {
		local_min[local_id] = INFINITY;
		local_imin[local_id] = -1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Reduce localMin and localMinIndex arrays within the workgroup
	for (int s = WARP_SIZE/2; s > 0; s >>= 1) {
	    if (local_id < s) {
	        if (local_min[local_id + s] < local_min[local_id]) {
	            local_min[local_id] = local_min[local_id + s];
	            local_imin[local_id] = local_imin[local_id + s];
	        }
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// Store the final max value and index in output and outputIndex
	if (local_id == 0) {
	    partial[get_group_id(0)] = local_min[0];
	    partialIdx[get_group_id(0)] = local_imin[0];
	}
}
)";

// vector argmin
std::string ocl_vargmin_code = R"(
__kernel void __name__(__global __type__* a, __global int* ai, int n, __global int* result)
{
	int imin;
	__type__ m = INFINITY;
	if (ai==NULL) {
		for (int i=0; i<n; ++i) if (a[i] < m) { m = a[i]; imin = i;	}
	} else {
		for (int i=0; i<n; ++i) if (a[i] < m) { m = a[i]; imin = ai[i]; }
	}
	*result = imin;
}
)";












// Hadamard (vector)
std::string ocl_vhadamard_code = R"(
__kernel void __name__(__global __type__* a, __global __type__* b, __global __type__* c, int m, int n, int pitch)
{
	const int x = get_global_id(0);
	if (x < n) c[x] = a[x] * b[x];
}
)";

// fypax : vector(n) = f(vector(n)) + scalar*vector(n)
// y = f(y + αx)
std::string ocl_fypax_code = R"(
__kernel void __name__(__global __type__* y, int n, int f, __type__ alpha, __global __type__* x) 
{ 
	int i = get_global_id(0); 
	if (i < n) y[i] = ocl___type___func(y[i] + (alpha != 0 ? alpha*x[i] : 0) , f);
}
)";



// ma : matrix(m,n) = scalar
// M = α
std::string ocl_ma_code = R"(
__kernel void __name__(__global __type__* y, int m, int n, int pitch, __type__ alpha) 
{
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	if (row < m && col < n)
		y[row*pitch + col] = alpha;
}
)";


// man : matrix(m,n) = scalar*matrix(m,n)
// M = αN
std::string ocl_man_code = R"(
__kernel void __name__(__global __type__* y, int m, int n, int pitch, __type__ alpha, __global __type__* x, int xpitch) 
{ 
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	if (row < m && col < n)
		y[row*pitch + col] = alpha*x[row*xpitch + col]; 
}
)";


// mpa : matrix(m,n) += scalar
// M += α
std::string ocl_mpa_code = R"(
__kernel void __name__(__global __type__* y, int m, int n, int pitch, __type__ alpha) 
{ 
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	if (row < m && col < n)
		y[row*pitch + col] += alpha; 
}
)";


// mpax : matrix(m,n) += scalar*vector(m)
// M += αx
std::string ocl_mpax_code = R"(
__kernel void __name__(__global __type__* y, int m, int n, int pitch, __type__ alpha, __global __type__* x, int axis) 
{ 
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	if (row < m && col < n)
		y[row*pitch + col] += alpha*x[(axis==0 ? col:row)]; 
}
)";


// mma : matrix(m,n) *= scalar
// M *= α
std::string ocl_mma_code = R"(
__kernel void __name__(__global __type__* y, int m, int n, int pitch, __type__ alpha) 
{ 
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	if (row < m && col < n) 
		y[row*pitch + col] *= alpha; 
}
)";


// C = A*B (Hadamard)
std::string ocl_hadamard_code = R"(
__kernel void __name__(__global __type__* a, __global __type__* b, __global __type__* c, int m, int n, int pitch)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (y < m && x < n) 
		c[y*pitch+x] = a[y*pitch+x] * b[y*pitch+x];
}
)";

// M *= v (element-wise)
std::string ocl_mmulv_code = R"(
__kernel void __name__(__global __type__* c, int m, int n, int pitch, __global __type__* v, int axis)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (y < m && x < n)
		c[y*pitch+x] *= v[(axis==0 ? x:y)];
}
)";

// M /= v
std::string ocl_mdivv_code = R"(
__kernel void __name__(__global __type__* c, int m, int n, int pitch, __global __type__* v, int axis)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (y < m && x < n)
		c[y*pitch+x] /= v[(axis==0 ? x:y)];
}
)";


// fmpax : matrix(m,n) = f(scalar * (matrix(m,n) + scalar*vector(m)))
// M = f(β*(M+αx))
std::string ocl_fmpax_code = R"(
__kernel void __name__(__global __type__* y, int m, int n, int pitch, int f, __type__ beta, 
						__type__ alpha, __global __type__* x, int axis) 
{ 
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	if (row < m && col < n) {
		int i = row*pitch + col;
		y[i] = ocl___type___func(beta * (y[i] + (alpha != 0 ? alpha*x[(axis==0 ? col:row)] : 0)) , f);
	}
}
)";


// matrix max per col/row
std::string ocl_matmax_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __global __type__* out, int axis)
{	
	int id = get_global_id(0);
	__type__ mv = -INFINITY;

	if (axis==0) {
		if (id >= n) return;
		for (int i=0; i<m; ++i) {
			__type__ val = a[i*pitch + id];
			if (val > mv) mv = val;
		}
	} else {
		if (id >= m) return;
		for (int j=0; j<n; ++j) {
			__type__ val = a[id*pitch + j];
			if (val > mv) mv = val;
		}
	}

	out[id] = mv;
}
)";

// matrix argmax per row
std::string ocl_margmax_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __global int* rowidcs)
{	
	int row = get_global_id(0);
	if (row >= m) return;

	__type__ mv = -INFINITY;
	int mi = -1;
	for (int j=0; j<n; ++j) {
	    __type__ val = a[row*pitch + j];
	    if (val > mv) {
	        mv = val;
	        mi = j;
	    }
	}

	rowidcs[row] = mi;
}
)";

// matrix min per col/row
std::string ocl_matmin_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __global __type__* out, int axis)
{	
	int id = get_global_id(0);
	__type__ mv = INFINITY;

	if (axis==0) {
		if (id >= n) return;
		for (int i=0; i<m; ++i) {
			__type__ val = a[i*pitch + id];
			if (val < mv) mv = val;
		}
	} else {
		if (id >= m) return;
		for (int j=0; j<n; ++j) {
			__type__ val = a[id*pitch + j];
			if (val < mv) mv = val;
		}
	}

	out[id] = mv;
}
)";

// matrix argmin per row
std::string ocl_margmin_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __global int* rowidcs)
{	
	int row = get_global_id(0);
	if (row >= m) return;

	__type__ mv = INFINITY;
	int mi = -1;
	for (int j=0; j<n; ++j) {
	    __type__ val = a[row*pitch + j];
	    if (val < mv) {
	        mv = val;
	        mi = j;
	    }
	}

	rowidcs[row] = mi;
}
)";

// sum of cube's slices
std::string ocl_sum3d_code = R"(
__kernel void __name__(__global __type__* in, int h, int m, int n, int xpitch, int ypitch, __global __type__* out)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	__type__ acc = 0;
	for (int z=0; z<h; ++z) acc += in[(z*ypitch+y)*xpitch+x];
	out[y*xpitch+x] = acc;
}
)";


// reduction: sum of matrix rows (partial)
std::string ocl_smrp_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __global __type__* rows)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int lx = get_local_id(0);
	const int workgroups = (n+WARP_SIZE-1) / WARP_SIZE;

	warp[lx] = (y < m && x < n ? a[y*pitch+x] : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (lx == 0) rows[get_group_id(1)*workgroups + get_group_id(0)] = warp[0];
}
)";

// reduction: sum of matrix rows squared (partial)
std::string ocl_smrp2_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __type__ alpha, __global __type__* rows)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int lx = get_local_id(0);
	const int workgroups = (n+WARP_SIZE-1) / WARP_SIZE;

	warp[lx] = (y < m && x < n ? (a[y*pitch+x]+alpha)*(a[y*pitch+x]+alpha) : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (lx == 0) rows[get_group_id(1)*workgroups + get_group_id(0)] = warp[0];
}
)";

std::string ocl_smr_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __global __type__* rows)
{
	int i = get_global_id(0);
	if (i < m) {
		__type__ acc = 0;
		for (int j=0; j<n; ++j) acc += a[i*pitch+j];
		rows[i] = acc;
	}
}
)";

// sum of natrix cols
std::string ocl_smc_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __global __type__* cols)
{
	int j = get_global_id(0);
	if (j < n) {
		__type__ acc = 0;
		for (int i=0; i<m; ++i) acc += a[i*pitch + j];
		cols[j] = acc;
	}
}
)";

// reduction: max of matrix rows (partial)
std::string ocl_rmaxp_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int pitch, __global __type__* rows)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int lx = get_local_id(0);
	const int workgroups = (n+WARP_SIZE-1) / WARP_SIZE;

	warp[lx] = (y < m && x < n ? a[y*pitch+x] : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (lx == 0) rows[get_group_id(1)*workgroups + get_group_id(0)] = warp[0];
}
)";

std::string ocl_mcnteq_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, 
					   __global __type__* b, int bpitch, __type__ novalue, __global int* rows)
{
	__local int warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int lx = get_local_id(0);
	const int workgroups = (n+WARP_SIZE-1) / WARP_SIZE;

	__type__ av = a[y*apitch+x];
	__type__ bv = b[y*bpitch+x];
	if (y < m && x < n && av != novalue && av == bv) warp[lx] = 1;
	else warp[lx] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (lx == 0) rows[get_group_id(1)*workgroups + get_group_id(0)] = warp[0];
}
)";

std::string ocl_meucl2_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, 
					   __global __type__* b, int bpitch, __global __type__* partial)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int lx = get_local_id(0);
	const int workgroups = (n+WARP_SIZE-1) / WARP_SIZE;

	if (y < m && x < n) {
		__type__ d = a[y*apitch+x] - b[y*bpitch+x];
		warp[lx] = d*d;
	} else warp[lx] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (lx == 0) partial[get_group_id(1)*workgroups + get_group_id(0)] = warp[0];
}
)";

std::string ocl_mmanh_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, 
					   __global __type__* b, int bpitch, __global __type__* partial)
{
	__local __type__ warp[WARP_SIZE];

	// each thread loads one element from global to local mem
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int lx = get_local_id(0);
	const int workgroups = (n+WARP_SIZE-1) / WARP_SIZE;

	warp[lx] = (y < m && x < n ? fabs(a[y*apitch+x] - b[y*bpitch+x]) : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in local mem
	for (int s=get_local_size(0)/2; s>0; s>>=1) {
		if (lx < s) warp[lx] += warp[lx + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (lx == 0) partial[get_group_id(1)*workgroups + get_group_id(0)] = warp[0];
}
)";


// C = αN
std::string ocl_can_code = R"(
__kernel void __name__(__global __type__* dst, int c, int m, int n, int xpitch, int ypitch, 
					   __type__ alpha, __global __type__* src, int sxpitch, int sypitch) 
{ 
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	if (z < c && y < m && x < n)
		dst[z*xpitch*ypitch + y*xpitch + x] = alpha*src[z*sxpitch*sypitch + y*sxpitch + x]; 
}
)";

// C = f(C)
std::string ocl_func3d_code = R"(
__kernel void __name__(__global __type__* c, int h, int m, int n, int pitch, int zstride, int f) 
{
 	int x = get_global_id(0);
	int y = get_global_id(1);
 	int z = get_global_id(2);
	if (z < h && y < m && x < n) {
		int i = z*zstride+y*pitch+x;
		c[i] = ocl___type___func(c[i], f);
	}
}
)";


/*
My first attempt, really slow!!!

//execute(kernels[fm_smr], cl::NullRange, cl::NDRange(PADY(m)), cl::NullRange);

std::string ocl_mcnteq_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, 
					   __global __type__* b, int bpitch, __type__ novalue, __global int* rows)
{
	int i = get_global_id(0);
	if (i < m) {
		int acc = 0;
		for (int j=0; j<n; ++j) 
			if (a[i*apitch+j] != novalue && a[i*apitch+j] == b[i*bpitch+j]) acc++;
		rows[i] = acc;
	}
}
)";

std::string ocl_mmanh_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, 
					   __global __type__* b, int bpitch, __global __type__* rows)
{
	int i = get_global_id(0);
	if (i < m) {
		__type__ acc = 0;
		for (int j=0; j<n; ++j) acc += fabs(a[i*apitch+j] - b[i*bpitch+j]);
		rows[i] = acc;
	}
}
)";

std::string ocl_meucl2_code = R"(
__kernel void __name__(__global __type__* a, int m, int n, int apitch, 
					   __global __type__* b, int bpitch, __global __type__* rows)
{
	int i = get_global_id(0);
	if (i < m) {
		__type__ acc = 0;
		for (int j=0; j<n; ++j) {
			__type__ d = (a[i*apitch+j] - b[i*bpitch+j]);
			acc += d*d;
		}

		rows[i] = acc;
	}
}
)";
*/

};     // namespace umml

#endif // UMML_KERNELS_OCL_INCLUDED
