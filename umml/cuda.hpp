#ifndef UMML_CUDA_INCLUDED
#define UMML_CUDA_INCLUDED

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "dev.hpp"
#include "func.hpp"
#include "blas_cuda.hpp"
#include "kernels_cuda.hpp"
#include "utils.hpp"


namespace umml {


class __CUDA {
 public:
	__CUDA() {}

	// no need for synchronization for one CUDA stream (the default)
	void synchronize() {
		//CUDA_CHECK(cudaDeviceSynchronize());
	}

	void synchronize_cuda_streams() {
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	
	template <typename Type>
	Type* alloc(int n) {
		Type* dmem;
		CUDA_CHECK(cudaMalloc((void**)&dmem, n*sizeof(Type)));
		return dmem;
	}

	template <typename Type>
	void to_gpu(Type* dmem, const Type* mem, int n) {
		CUDA_CHECK(cudaMemcpy(dmem, mem, n*sizeof(Type), cudaMemcpyHostToDevice));
	}

	template <typename Type>
	void to_cpu(const Type* dmem, Type* mem, int n) {
		CUDA_CHECK(cudaMemcpy(mem, dmem, n*sizeof(Type), cudaMemcpyDeviceToHost));
	}

	template <typename Type>
	void set_device_element(Type* dmem, int offset, Type value) {
		CUDA_CHECK(cudaMemcpy(dmem+offset, &value, sizeof(Type), cudaMemcpyHostToDevice));
	}

	template <typename Type>
	Type get_device_element(Type* dmem, int offset) {
		Type value;
		CUDA_CHECK(cudaMemcpy(&value, dmem+offset, sizeof(Type), cudaMemcpyDeviceToHost));
		return value;
	}

	template <typename Type>
	void fill(Type* dst, Type val, int offset, int n) {
		vec_set(dst+offset, n, val);
	}

	template <typename Type>
	void copy(const Type* src, Type* dst, int src_offset, int dst_offset, int n) {
		vec_set(dst+dst_offset, src+src_offset, n, Type(1));
	}

	template <typename Type>
	void copy2d(const Type* src, int spitch, Type* dst, int dpitch, int sy, int sx, int dy, int dx, int ylen, int xlen) {
		dim3 blocks(PADX(xlen)/TILES, PADY(ylen)/TILES);
		dim3 threads(TILES, TILES);
		gpu_copy2d<Type><<<blocks,threads>>>(src, spitch, dst, dpitch, sy, sx, dy, dx, ylen, xlen);
		synchronize();
	}

	template <typename Type>
	void copy3d(int zdim, const Type* src, int spitch, int szstride, 
				Type* dst, int dpitch, int dzstride, int sy, int sx, int dy, int dx, int ylen, int xlen) {
		dim3 blocks(PADX(xlen)/TILES, PADY(ylen)/TILES, zdim);
		dim3 threads(TILES, TILES);
		gpu_copy3d<Type><<<blocks,threads>>>(zdim, src, spitch, szstride, dst, dpitch, dzstride, sy, sx, dy, dx, ylen, xlen);
		synchronize();
	}

	template <typename Type>
	void reciprocal(Type alpha, const Type* x, int n, Type* y) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_reciprocal<Type><<<GROUPS,THREADS>>>(alpha, x, n, y);
		synchronize();
	}


 public:
	template <typename Type>
	void vec_set(Type* y, int n, Type alpha) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_vecset<Type><<<GROUPS,THREADS>>>(y, n, alpha);
		synchronize();
	}
	
	template <typename Type>
	void vec_set(Type* y, const Type* x, int n, Type alpha) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_vecset<Type><<<GROUPS,THREADS>>>(y, x, n, alpha);
		synchronize();
	}

	template <typename Type>
	void vec_squared(Type* y, const Type* x, int n) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_yx2<Type><<<GROUPS,THREADS>>>(y, x, n);
		synchronize();
	}
	
	template <typename Type>
	void vec_plus(Type* y, int n, Type alpha) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_vecplus<Type><<<GROUPS,THREADS>>>(y, n, alpha);
		synchronize();
	}
	
	template <typename Type>
	void axpy(Type alpha, const Type* x, int n, Type* y) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_axpy<Type><<<GROUPS,THREADS>>>(alpha, x, n, y);
		synchronize();
	}
	
	template <typename Type>
	void zaxpby(Type alpha, const Type* x, int n, Type beta, const Type* y, Type* z) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_zaxpby<Type><<<GROUPS,THREADS>>>(alpha, x, n, beta, y, z);
		synchronize();
	}

	template <typename Type>
	void hadamard(const Type* x, const Type* y, Type* z, int n) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_hadamard<Type><<<GROUPS,THREADS>>>(x, y, z, n);
		synchronize();
	}
	
	template <typename Type>
	void vec_func(Type* y, int n, int f, Type alpha, const Type* x) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		gpu_apply_function1d<Type><<<GROUPS,THREADS>>>(y, n, f, alpha, x);
		synchronize();
	}
	
	template <typename Type>
	void sve(const Type* a, int n, Type* sum) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		Type* d_partial = alloc<Type>(GROUPS);
		gpu_svep<Type><<<GROUPS,THREADS>>>(a, n, d_partial);
		gpu_sve<Type><<<1,1>>>(d_partial, GROUPS, sum);
		synchronize();
		CUDA_CHECK(cudaFree(d_partial));
	}

	template <typename Type>
	void sve2(const Type* a, int n, Type alpha, Type* sum) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		Type* d_partial = alloc<Type>(GROUPS);
		gpu_svep2<Type><<<GROUPS,THREADS>>>(a, n, alpha, d_partial);
		gpu_sve<Type><<<1,1>>>(d_partial, GROUPS, sum);
		synchronize();
		CUDA_CHECK(cudaFree(d_partial));
	}

	template <typename Type>
	void count_equal(const Type* a, int n, const Type* b, int* cnt) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		int* d_partial = alloc<int>(GROUPS);
		gpu_cnteq<Type><<<GROUPS,THREADS>>>(a, n, b, d_partial);
		gpu_sve<int><<<1,1>>>(d_partial, GROUPS, cnt);
		synchronize();
		CUDA_CHECK(cudaFree(d_partial));
	}

	template <typename Type>
	void dist_squared(const Type* a, int n, const Type* b, Type* dist) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		Type* d_partial = alloc<Type>(GROUPS);
		gpu_eucl2<Type><<<GROUPS,THREADS>>>(a, n, b, d_partial);
		gpu_sve<Type><<<1,1>>>(d_partial, GROUPS, dist);
		synchronize();
		CUDA_CHECK(cudaFree(d_partial));
	}

	template <typename Type>
	void manhattan(const Type* a, int n, const Type* b, Type* dist) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		Type* d_partial = alloc<Type>(GROUPS);
		gpu_manh<Type><<<GROUPS,THREADS>>>(a, n, b, d_partial);
		gpu_sve<Type><<<1,1>>>(d_partial, GROUPS, dist);
		synchronize();
		CUDA_CHECK(cudaFree(d_partial));
	}

	template <typename Type>
	void argmax(const Type* a, int n, int* pos) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		Type* d_partial = alloc<Type>(GROUPS);
		int* d_idcs = alloc<int>(GROUPS);
		gpu_vargmaxp<Type><<<GROUPS,THREADS>>>(a, n, d_partial, d_idcs);
		gpu_vargmax<Type><<<1,1>>>(d_partial, d_idcs, GROUPS, pos);
		synchronize();
		CUDA_CHECK(cudaFree(d_idcs));
		CUDA_CHECK(cudaFree(d_partial));
	}

	template <typename Type>
	void argmax(const Type* a, int m, int n, int pitch, int* idcs) {
		gpu_margmax<Type><<<m,1>>>(a, m, n, pitch, idcs);
	}

	template <typename Type>
	void reduce_max2d(const Type* a, int m, int n, int pitch, Type* out, int axis) {
		int blocks = (axis==0 ? n:m);
		gpu_matmax<Type><<<blocks,1>>>(a, m, n, pitch, out, axis);
	}

	template <typename Type>
	void argmin(const Type* a, int n, int* pos) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		Type* d_partial = alloc<Type>(GROUPS);
		int* d_idcs = alloc<int>(GROUPS);
		gpu_vargminp<Type><<<GROUPS,THREADS>>>(a, n, d_partial, d_idcs);
		gpu_vargmin<Type><<<1,1>>>(d_partial, d_idcs, GROUPS, pos);
		synchronize();
		CUDA_CHECK(cudaFree(d_idcs));
		CUDA_CHECK(cudaFree(d_partial));
	}

	template <typename Type>
	void argmin(const Type* a, int m, int n, int pitch, int* idcs) {
		gpu_margmin<Type><<<m,1>>>(a, m, n, pitch, idcs);
	}

	template <typename Type>
	void reduce_min2d(const Type* a, int m, int n, int pitch, Type* out, int axis) {
		int blocks = (axis==0 ? n:m);
		gpu_matmin<Type><<<blocks,1>>>(a, m, n, pitch, out, axis);
	}

	template <typename Type>
	void dot(Type alpha, const Type* a, int n, Type beta, const Type* b, Type* result) {
		const int GROUPS = (n+THREADS-1) / THREADS;
		Type* d_partial = alloc<Type>(GROUPS);
		gpu_dot_partial<Type><<<GROUPS,THREADS>>>(alpha, a, n, beta, b, d_partial);
		gpu_sve<Type><<<1,1>>>(d_partial, GROUPS, result);
		synchronize();
		CUDA_CHECK(cudaFree(d_partial));
	}
	
	
	template <typename Type>
	void mat_set(Type* y, int m, int n, int pitch, Type alpha) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_matset<Type><<<blocks,threads>>>(y, m, n, pitch, alpha);
		synchronize();
	}
	
	template <typename Type>
	void mat_set(Type* y, int m, int n, int apitch, Type alpha, const Type* x, int xpitch) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_matset<Type><<<blocks,threads>>>(y, m, n, apitch, alpha, x, xpitch);
		synchronize();
	}
	
	template <typename Type>
	void mat_plus(Type* y, int m, int n, int pitch, Type alpha) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_matplus<Type><<<blocks,threads>>>(y, m, n, pitch, alpha);
		synchronize();
	}

	template <typename Type>
	void mplusv(Type* y, int m, int n, int pitch, Type alpha, const Type* x, int axis) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_matplus<Type><<<blocks,threads>>>(y, m, n, pitch, alpha, x, axis);
		synchronize();
	}

	template <typename Type>
	void mat_mul(Type* y, int m, int n, int pitch, Type alpha) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_matmul<Type><<<blocks,threads>>>(y, m, n, pitch, alpha);
		synchronize();
	}
	
	template <typename Type>
	void axpy(Type alpha, const Type* x, int m, int n, int xpitch, Type* y, int ypitch) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_axpy<Type><<<blocks,threads>>>(alpha, x, m, n, xpitch, y, ypitch);
		synchronize();
	}

	template <typename Type>
	void zaxpby(Type alpha, const Type* x, int m, int n, int xpitch, Type beta, const Type* y, int ypitch, Type* z, int zpitch) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_zaxpbym<Type><<<blocks,threads>>>(alpha, x, m, n, xpitch, beta, y, ypitch, z, zpitch);
		synchronize();
	}

	template <typename Type>
	void hadamard(const Type* a, const Type* b, Type* c, int m, int n, int pitch) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_hadamard<Type><<<blocks,threads>>>(a, b, c, m, n, pitch);
		synchronize();
	}
	
	template <typename Type>
	void prod_vec(Type* y, int m, int n, int pitch, const Type* v, int axis) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_mprodv<Type><<<blocks,threads>>>(y, m, n, pitch, v, axis);
		synchronize();
	}

	template <typename Type>
	void div_vec(Type* y, int m, int n, int pitch, const Type* v, int axis) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_mdivv<Type><<<blocks,threads>>>(y, m, n, pitch, v, axis);
		synchronize();
	}
	
	template <typename Type>
	void mat_func(Type* y, int m, int n, int pitch, int f, Type beta, Type alpha, const Type* x, int axis=0) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_apply_function2d<Type><<<blocks,threads>>>(y, m, n, pitch, f, beta, alpha, x, axis);
		synchronize();
	}

	template <typename Type>
	void reduce_sum2d(const Type* a, int m, int n, int pitch, Type* out, int axis) {
		if (axis==0) {
			gpu_smc<Type><<<PADX(n),1>>>(a, m, n, pitch, out);
			synchronize();
		} else {
			const int WS = 256;
			const int workgroups = (n+WS-1) / WS;
			Type* rows_partial = alloc<Type>(PADY(m)*workgroups);
			dim3 blocks(workgroups, PADY(m));
			dim3 threads(WS);
			gpu_smrp<Type><<<blocks,threads>>>(a, m, n, pitch, rows_partial);
			gpu_smr<Type><<<PADY(m),1>>>(rows_partial, m, workgroups, workgroups, out);
			synchronize();
			CUDA_CHECK(cudaFree(rows_partial));
		}
	}

	template <typename Type>
	void sme_full(const Type* a, int m, int n, int pitch, Type* sum) {
		Type* rows_partial = alloc<Type>(PADY(m));
		gpu_smr<Type><<<PADY(m),1>>>(a, m, n, pitch, rows_partial);
		sve<Type>(rows_partial, m, sum);
		synchronize();
		CUDA_CHECK(cudaFree(rows_partial));
	}
	
	template <typename Type>
	void sme(const Type* a, int m, int n, int pitch, Type* sum) {
		const int WS = THREADS;
		const int workgroups = (n+WS-1) / WS;
		Type* rows_partial = alloc<Type>(PADY(m)*workgroups);
		dim3 blocks(workgroups, PADY(m));
		dim3 threads(WS);
		gpu_smrp<Type><<<blocks,threads>>>(a, m, n, pitch, rows_partial);
		sme_full<Type>(rows_partial, m, workgroups, workgroups, sum);
		CUDA_CHECK(cudaFree(rows_partial));
	}

	template <typename Type>
	void sme2(const Type* a, int m, int n, int pitch, Type alpha, Type* sum) {
		const int WS = THREADS;
		const int workgroups = (n+WS-1) / WS;
		Type* rows_partial = alloc<Type>(PADY(m)*workgroups);
		dim3 blocks(workgroups, PADY(m));
		dim3 threads(WS);
		gpu_smrp2<Type><<<blocks,threads>>>(a, m, n, pitch, alpha, rows_partial);
		sme_full<Type>(rows_partial, m, workgroups, workgroups, sum);
		CUDA_CHECK(cudaFree(rows_partial));
	}
	
	template <typename Type>
	void count_equal(const Type* a, int m, int n, int apitch, const Type* b, int bpitch, Type novalue, int* cnt) {
		const int WS = THREADS;
		const int workgroups = (n+WS-1) / WS;
		int* rows_partial = alloc<int>(PADY(m)*workgroups);
		dim3 blocks(workgroups, PADY(m));
		dim3 threads(WS);
		gpu_cnteq<Type><<<blocks,threads>>>(a, m, n, apitch, b, bpitch, novalue, rows_partial);
		sme_full<int>(rows_partial, m, workgroups, workgroups, cnt);
		CUDA_CHECK(cudaFree(rows_partial));
	}

	template <typename Type>
	void dist_squared(const Type* a, int m, int n, int apitch, const Type* b, int bpitch, Type* dist) {
		const int WS = THREADS;
		const int workgroups = (n+WS-1) / WS;
		Type* rows_partial = alloc<Type>(PADY(m)*workgroups);
		dim3 blocks(workgroups, PADY(m));
		dim3 threads(WS);
		gpu_eucl2<Type><<<blocks,threads>>>(a, m, n, apitch, b, bpitch, rows_partial);
		sme_full<Type>(rows_partial, m, workgroups, workgroups, dist);
		CUDA_CHECK(cudaFree(rows_partial));
	}
	
	template <typename Type>
	void manhattan(const Type* a, int m, int n, int apitch, const Type* b, int bpitch, Type* dist) {
		const int WS = THREADS;
		const int workgroups = (n+WS-1) / WS;
		Type* rows_partial = alloc<Type>(PADY(m)*workgroups);
		dim3 blocks(workgroups, PADY(m));
		dim3 threads(WS);
		gpu_manh<Type><<<blocks,threads>>>(a, m, n, apitch, b, bpitch, rows_partial);
		sme_full<Type>(rows_partial, m, workgroups, workgroups, dist);
		CUDA_CHECK(cudaFree(rows_partial));
	}

	template <typename Type>
	void outer(Type* z, int pitch, const Type* x, int m, const Type* y, int n) {
		dim3 blocks(pitch/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_outer<Type><<<blocks,threads>>>(z, pitch, x, m, y, n);
		synchronize();
	}
	
	template <typename Type>
	void gemv1(Type alpha, const Type* a, int m, int n, int pitch, const Type* x, Type beta, Type* y) {
		const int GROUPS = (m+THREADS-1) / THREADS;
		gpu_gemv1<Type><<<GROUPS,THREADS>>>(alpha, a, m, n, pitch, x, beta, y);
		synchronize();
	}
	
	template <typename Type>
	void gemv2(Type alpha, const Type* a, int m, int n, int pitch, const Type* x, Type beta, Type* y) {
		dim3 blocks(1,m);
		dim3 threads(TILES);
		gpu_gemv2<Type><<<blocks,threads>>>(alpha, a, m, n, pitch, x, beta, y);
		synchronize();
	}
	
	template <typename Type>
	void gemm(const Type* a, const Type* b, Type* c, int m, int n, int p, int apitch, int bpitch, int cpitch) {
		dim3 blocks(bpitch/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_gemm<Type><<<blocks,threads>>>(a, b, c, m, n, p, apitch, bpitch, cpitch);
		synchronize();
	}
	
	template <typename Type>
	void gemt(const Type* a, int m, int n, int apitch, Type* t, int tpitch) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, 8);
		gpu_gemt<Type><<<blocks,threads>>>(a, m, n, apitch, t, tpitch);
		synchronize();
	}
	
	template <typename Type>
	void gram(const Type* a, int m, int n, int apitch, Type* c, int cpitch) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_gram<Type><<<blocks,threads>>>(a, m, n, apitch, c, cpitch);
		synchronize();
	}

	template <typename Type>
	void cub_set(Type* y, int c, int m, int n, int xpitch, int ypitch, Type alpha, const Type* x, int xxpitch, int xypitch) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES, c);
		dim3 threads(TILES, TILES);
		gpu_cubset<Type><<<blocks,threads>>>(y, c, m, n, xpitch, ypitch, alpha, x, xxpitch, xypitch);
		synchronize();
	}

	template <typename Type>
	void cub_func(Type* c, int h, int m, int n, int pitch, int zstride, int f) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES, h);
		dim3 threads(TILES, TILES);
		gpu_apply_function3d<Type><<<blocks,threads>>>(c, h, m, n, pitch, zstride, f);
		synchronize();
	}


	template <typename Type>
	void reduce_sum3d(const Type* in, int h, int m, int n, int xpitch, int ypitch, Type* out) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES);
		dim3 threads(TILES, TILES);
		gpu_sum3d<Type><<<blocks,threads>>>(in, h, m, n, xpitch, ypitch, out);
		synchronize();
	}

	template <typename Type>
	void outer3d(Type* out, int h, int zstride, int pitch, 
				 const Type* v, int m, int vzstride, const Type* u, int n, int uzstride) {
		dim3 blocks(PADX(n)/TILES, PADY(m)/TILES, h);
		dim3 threads(TILES, TILES);
		gpu_outer3d<Type><<<blocks,threads>>>(out, h, zstride, pitch, v, m, vzstride, u, n, uzstride);
		synchronize();
	}

};


// all functions should be called via __cuda__. 
static __CUDA __cuda__;


};     // namespace umml

#endif // UMML_CUDA_INCLUDED
