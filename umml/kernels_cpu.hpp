#ifndef UMML_KERNELS_CPU_INCLUDED
#define UMML_KERNELS_CPU_INCLUDED

#include "func.hpp"
#include "blas_cpu.hpp"


namespace umml {


// set n elements of dst[dst_offset] to a value
template <typename Type>
void cpu_fill(Type* dst, Type val, int offset, int n) {
	std::fill(dst+offset, dst+offset+n, val);
}

// copy n elements from src[src_offset] to dst[dst_offset]
template <typename Type>
void cpu_copy(const Type* src, Type* dst, int src_offset, int dst_offset, int n) {
	std::copy(src+src_offset, src+src_offset+n, dst+dst_offset);
}

// copy the region (ylen x xlen) from src[sy,sx] to dst[dy,dx]
template <typename Type>
void cpu_copy2d(const Type* src, int spitch, Type* dst, int dpitch, int sy, int sx, int dy, int dx, int ylen, int xlen) {
	for (int i=0; i<ylen; ++i)
		std::copy(src+(i+sy)*spitch+sx, src+(i+sy)*spitch+sx+xlen, dst+(i+dy)*dpitch+dx);
}

// copy the region (ylen x xlen) from src[sy,sx] to dst[dy,dx]
template <typename Type>
void cpu_copy3d(int zdim, const Type* src, int spitch, int szstride, 
				Type* dst, int dpitch, int dzstride, int sy, int sx, int dy, int dx, int ylen, int xlen) {
	for (int z=0; z<zdim; ++z)
	for (int y=0; y<ylen; ++y)
		std::copy(src+z*szstride+(y+sy)*spitch+sx, src+z*szstride+(y+sy)*spitch+sx+xlen, dst+z*dzstride+(y+dy)*dpitch+dx);
}

// y = α
template <typename Type>
void cpu_vecset(Type* y, int n, Type alpha) {
	for (int i=0; i<n; ++i) y[i] = alpha;
}

// y = αx
template <typename Type>
void cpu_vecset(Type* y, const Type* x, int n, Type alpha) {
	for (int i=0; i<n; ++i) y[i] = alpha*x[i];
}

// v == u
template <typename Type>
bool cpu_vecequal(const Type* v, const Type* u, int n, Type tolerance=Type(1e-8)) {
	bool equal = true;
	for (int i=0; i<n && equal; ++i) equal = similar_values(v[i], u[i], tolerance);
	return equal;
}

// y += α
template <typename Type>
void cpu_vecplus(Type* y, int n, Type alpha) {
	for (int i=0; i<n; ++i) y[i] += alpha;
}

// Hadamard (vector)
template <typename Type>
void cpu_hadamard(const Type* a, const Type* b, Type* c, int n)
{
	for (int j=0; j<n; ++j) c[j] = a[j] * b[j];
}

// v = f(v)
template <typename Type>
void cpu_apply_function1d(Type* c, int n, int f)
{
	for (int i=0; i<n; ++i) c[i] = umml::cpu_function(c[i], f);
}

// v = f(v+αx)
template <typename Type>
void cpu_apply_function1d(Type* c, int n, int f, Type alpha, const Type* x)
{
	for (int i=0; i<n; ++i) c[i] = umml::cpu_function(c[i]+alpha*x[i], f);
}

// result = Σxi
template <typename Type>
void cpu_sum(const Type* x, int n, Type* result) {
	Type s = 0;
	for (int i=0; i<n; ++i) s += x[i];
	*result = s;
}

// result = Σ(xi+α)^2
template <typename Type>
void cpu_sum2(const Type* x, int n, Type alpha, Type* result) {
	Type s = 0;
	for (int i=0; i<n; ++i) s += (x[i]+alpha)*(x[i]+alpha);
	*result = s;
}

// y = y / sqrt(Σ(yi*yi))
template <typename Type>
void cpu_vec_normalize(Type* y, int n, Type* norm=nullptr) {
	Type s;
	cpu_sumsq(y, n, &s);
	s = std::sqrt(s);
	for (int i=0; i<n; ++i) y[i] /= s;
	if (norm != nullptr) *norm = s;
}

// [4,-3,10,2,5] -> [0,0,1,0,0]
template <typename IType, typename OType>
void cpu_vec_argmaxto1hot(OType* y, int n, const IType* x) {
	std::fill(y, y+n, 0);
	int imax = 0;
	for (int i=1; i<n; ++i) if (x[i] > x[imax]) imax = i;
	y[imax] = OType(1);
}



/*
 CPU only case (no padding)
 ~~~~~~~~~~~~~~~~~~~~~~~~~~

 m11  m12  m13 
 m21  m22  m23
 m32  m32  m33


 CPU with padding
 ~~~~~~~~~~~~~~~~

 m11  m12  m13  ---
 m21  m22  m23  ---
 m32  m32  m33  ---
 ---  ---  ---  ---

 pitch = 4

*/


template <typename Type>
void cpu_matset(Type* c, int m, int n, int pitch, Type val) 
{
	for (int y=0; y<m; ++y) 
	for (int x=0; x<n; ++x) c[y*pitch+x] = val;
}

template <typename Type>
void cpu_matset(Type* c, int m, int n, int pitch, const Type* a, int apitch) 
{
	for (int y=0; y<m; ++y) 
	for (int x=0; x<n; ++x) c[y*pitch+x] = a[y*apitch+x];
}

// M += α
template <typename Type>
void cpu_matplus(Type* c, int m, int n, int pitch, Type val)
{
	for (int y=0; y<m; ++y) 
	for (int x=0; x<n; ++x) c[y*pitch+x] += val;
}

// M += α*v 
template <typename Type>
void cpu_mplusv(Type* c, int m, int n, int pitch, Type alpha, const Type* v, int axis)
{
	if (axis==0) {
		for (int y=0; y<m; ++y) 
		for (int x=0; x<n; ++x) c[y*pitch+x] += alpha*v[x];
	} else {
		for (int y=0; y<m; ++y) 
		for (int x=0; x<n; ++x) c[y*pitch+x] += alpha*v[y];
	}
}

// M *= α
template <typename Type>
void cpu_matmul(Type* c, int m, int n, int pitch, Type val)
{
	for (int y=0; y<m; ++y) 
	for (int x=0; x<n; ++x) c[y*pitch+x] *= val;
}

// Hadamard (matrix)
template <typename Type>
void cpu_hadamard(const Type* a, const Type* b, Type* c, int m, int n, int pitch)
{
	for (int i=0; i<m; ++i)
	for (int j=0; j<n; ++j) c[i*pitch+j] = a[i*pitch+j] * b[i*pitch+j];
}

// M *= v
template <typename Type>
void cpu_mprodv(Type* c, int m, int n, int pitch, const Type* v, int axis)
{
	if (axis==0) {
		for (int y=0; y<m; ++y)
		for (int x=0; x<n; ++x) c[y*pitch+x] *= v[x];
	} else {
		for (int y=0; y<m; ++y)
		for (int x=0; x<n; ++x) c[y*pitch+x] *= v[y];
	}
}

// M /= v
template <typename Type>
void cpu_mdivv(Type* c, int m, int n, int pitch, const Type* v, int axis)
{
	if (axis==0) {
		for (int y=0; y<m; ++y)
		for (int x=0; x<n; ++x) c[y*pitch+x] /= v[x];
	} else {
		for (int y=0; y<m; ++y)
		for (int x=0; x<n; ++x) c[y*pitch+x] /= v[y];
	}
}


// M = f(M)
template <typename Type>
void cpu_apply_function2d(Type* c, int m, int n, int pitch, int f)
{
	for (int y=0; y<m; ++y)
	for (int x=0; x<n; ++x) c[y*pitch+x] = umml::cpu_function(c[y*pitch+x], f);
}


// M = f(β*(M+αv))
template <typename Type>
void cpu_apply_function2d(Type* c, int m, int n, int pitch, int f, Type beta, Type alpha, const Type* v, int axis)
{
	if (axis==0) {
		for (int y=0; y<m; ++y)
		for (int x=0; x<n; ++x) c[y*pitch+x] = umml::cpu_function(beta*(c[y*pitch+x]+alpha*v[x]), f);
	} else {
		for (int y=0; y<m; ++y)
		for (int x=0; x<n; ++x) c[y*pitch+x] = umml::cpu_function(beta*(c[y*pitch+x]+alpha*v[y]), f);
	}
}


// [4,-3, 10, 2, 5] -> [0,0,1,0,0]
// [2, 5,  3,-4, 8]    [0,0,0,0,1]
template <typename IType, typename OType>
void cpu_mat_argmaxto1hot(OType* c, int m, int n, int pitch, const IType* a, int apitch) {
	std::memset(c, 0, m*pitch*sizeof(OType));
	for (int i=0; i<m; ++i) {
		const int aofs = i*apitch;
		int jmax = 0;
		for (int j=1; j<n; ++j) if (a[aofs+j] > a[aofs+jmax]) jmax = j;
		c[i*pitch+jmax] = OType(1);
	}
}

template <typename Type>
void cpu_sum(const Type* a, int m, int n, int pitch, Type* result) {
	Type s = 0;
	for (int i=0; i<m; ++i) {
		Type acc = 0;
		for (int j=0; j<n; ++j) acc += a[i*pitch+j];
		s += acc;
	}
	*result = s;
}

template <typename Type>
void cpu_sum2(const Type* a, int m, int n, int pitch, Type alpha, Type* result) {
	Type s = 0;
	for (int i=0; i<m; ++i) {
		Type acc = 0;
		for (int j=0; j<n; ++j) acc += (a[i*pitch+j]+alpha)*(a[i*pitch+j]+alpha);
		s += acc;
	}
	*result = s;
}

// vectors distance squared (a0-b0)^2 + (a1-b1)^2 + ...
template <typename Type>
Type cpu_distance_squared(const Type* a, int n, const Type* b)
{
	Type acc = 0;
	for (int i=0; i<n; ++i) {
		Type d = a[i] - b[i];
		acc += d*d;
	}
	return acc;
}

// vectors manhattan distance |a0-b0| + |a1-b1| + ...
template <typename Type>
Type cpu_manhattan(const Type* a, int n, const Type* b)
{
	Type acc = 0;
	for (int i=0; i<n; ++i) acc += std::abs(a[i] - b[i]);
	return acc;
}


// matrices distance squared (a00-b00)^2 + (a01-b01)^2 + ...
template <typename Type>
Type cpu_distance_squared(const Type* a, int m, int n, int apitch, const Type* b, int bpitch)
{
	Type dist = 0;
	for (int i=0; i<m; ++i) {
		Type acc = 0;
		for (int j=0; j<n; ++j) {
			Type d = a[i*apitch+j] - b[i*bpitch+j];
			acc += d*d;
		}
		dist += acc;
	}
	return dist;
}

// matrices manhattan distance |a00-b00| + |a01-b01| + ...
template <typename Type>
Type cpu_manhattan(const Type* a, int m, int n, int apitch, const Type* b, int bpitch)
{
	Type dist = 0;
	for (int i=0; i<m; ++i) {
		Type acc = 0;
		for (int j=0; j<n; ++j) acc += std::abs(a[i*apitch+j] - b[i*bpitch+j]);
		dist += acc;
	}
	return dist;
}

// sum matrix cols/rows
template <typename Type>
void cpu_reduce_sum2d(const Type* a, int m, int n, int pitch, Type* b, int axis)
{
	if (axis==0) {
		std::fill(b, b+n, 0);
		for (int i=0; i<m; ++i)
		for (int j=0; j<n; ++j) b[j] += a[i*pitch+j];
	} else {
		for (int i=0; i<m; ++i) {
			Type acc = 0;
			for (int j=0; j<n; ++j) acc += a[i*pitch+j];
			b[i] = acc;
		}
	}
}

// max of matrix cols/rows
template <typename Type>
void cpu_reduce_max2d(const Type* a, int m, int n, int pitch, Type* b, int axis)
{
	if (axis==0) {
		std::fill(b, b+n, std::numeric_limits<Type>::min());
		for (int i=0; i<m; ++i)
		for (int j=0; j<n; ++j) {
			Type val = a[i*pitch+j];
			if (val > b[j]) b[j] = val;
		}
	} else {
		for (int i=0; i<m; ++i) {
			b[i] = a[i*pitch];
			for (int j=1; j<n; ++j) {
				Type val = a[i*pitch+j];
				if (val > b[i]) b[i] = val;
			}
		}
	}
}

// min of matrix cols/rows
template <typename Type>
void cpu_reduce_min2d(const Type* a, int m, int n, int pitch, Type* b, int axis)
{
	if (axis==0) {
		std::fill(b, b+n, std::numeric_limits<Type>::max());
		for (int i=0; i<m; ++i)
		for (int j=0; j<n; ++j) {
			Type val = a[i*pitch+j];
			if (val < b[j]) b[j] = val;
		}
	} else {
		for (int i=0; i<m; ++i) {
			b[i] = a[i*pitch];
			for (int j=1; j<n; ++j) {
				Type val = a[i*pitch+j];
				if (val < b[i]) b[i] = val;
			}
		}
	}
}


// cube = src_cube
template <typename Type>
void cpu_cubset(Type* dst, int c, int m, int n, int xpitch, int ypitch, const Type* a, int axpitch, int aypitch) 
{
	for (int z=0; z<c; ++z) 
	for (int y=0; y<m; ++y) 
	for (int x=0; x<n; ++x) dst[z*xpitch*ypitch+y*xpitch+x] = a[z*axpitch*aypitch+y*axpitch+x];
}

// sum cube's slices
template <typename Type>
void cpu_reduce_sum3d(const Type* a, int h, int m, int n, int xpitch, int ypitch, Type* b)
{
	std::fill(b, b+m*xpitch, 0);
	for (int k=0; k<h; ++k)
	for (int i=0; i<m; ++i)
	for (int j=0; j<n; ++j) b[i*xpitch+j] += a[(k*ypitch+i)*xpitch+j];
}

// C = f(C)
template <typename Type>
void cpu_apply_function3d(Type* c, int h, int m, int n, int pitch, int zstride, int f)
{
	for (int z=0; z<h; ++z)
	for (int y=0; y<m; ++y)
	for (int x=0; x<n; ++x) {
		int i = z*zstride+y*pitch+x;
		c[i] = umml::cpu_function(c[i], f);
	}
}


};     // namespace umml

#endif // UMML_KERNELS_CPU_INCLUDED
