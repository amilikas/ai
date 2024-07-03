#ifndef UMML_MAXPOOL_CUDA_INCLUDED
#define UMML_MAXPOOL_CUDA_INCLUDED


namespace umml {


// output size: (n-k)/stride + 1

template <typename Type>
__global__ void gpu_maxpool2d(const Type* in, int c, int m, int n, int pitch, int zstride,
							  int k, int stride, Type* out, int oxpitch, int ozstride) 
{
	// output(y,x)
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int z = blockIdx.z*blockDim.z + threadIdx.z;

	// corresponding top-left corner of in(iy,ix)
	const int ix = x*stride;
	const int iy = y*stride;
	const int iz = z*zstride;

	Type max = in[iz + iy*pitch + ix];
	for (int i=0; i<k; ++i) {
		for (int j=0; j<k; ++j) {
			Type _m = in[iz + (iy+i)*pitch + (ix+j)];
			if (_m > max) max = _m;
		}
	}
	if (z < c && y < (m-k)/stride+1 && x < (n-k)/stride+1) 
		out[z*ozstride + y*oxpitch + x] = max;
}

template <typename Type>
__global__ void gpu_maxpool2di(const Type* in, int c, int m, int n, int pitch, int zstride,
							   int k, int stride, Type* out, int oxpitch, int ozstride, int* idcs) 
{
	// output(y,x)
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int z = blockIdx.z*blockDim.z + threadIdx.z;

	// corresponding top-left corner of in(iy,ix)
	const int ix = x*stride;
	const int iy = y*stride;
	const int iz = z*zstride;

	int maxloc = iz + iy*pitch + ix;
	Type max = in[maxloc];
	for (int i=0; i<k; ++i) {
		for (int j=0; j<k; ++j) {
			int loc = iz + (iy+i)*pitch + (ix+j);
			Type _m = in[loc];
			if (_m > max) {
				max = _m;
				maxloc = loc;
			}
		}
	}
	if (z < c && y < (m-k)/stride+1 && x < (n-k)/stride+1) 
		out[z*ozstride + y*oxpitch + x] = max;

	const int w = (n-k)/stride+1;
	const int h = (m-k)/stride+1;
	if (z < c && y < h && x < w) {
		out[z*ozstride + y*oxpitch + x] = max;
		idcs[z*w*h + y*w + x] = maxloc;
	}
}


//
// 2D max pooling
//
// No actual need for __cudapool class, just to keep consident with OpenCL's interface.

struct __cudapool {

__cudapool() {}

template <typename Type>
void maxpool2d(const Type* in, int c, int m, int n, int pitch, int zstride, 
			   int k, int stride, Type* out, int oxpitch, int ozstride) {
	int ch = PADY(pooling_size(m,k,stride));
	int cw = PADX(pooling_size(n,k,stride));
	dim3 blocks(cw/TILES,ch/TILES,c);
	dim3 threads(TILES, TILES);
	gpu_maxpool2d<Type><<<blocks,threads>>>(in, c, m, n, pitch, zstride, k, stride, out, oxpitch, ozstride);
	__cuda__.synchronize();
}

template <typename Type>
void maxpool2d(const Type* in, int c, int m, int n, int pitch, int zstride, 
			   int k, int stride, Type* out, int oxpitch, int ozstride, int* idcs) {
	int ch = PADY(pooling_size(m,k,stride));
	int cw = PADX(pooling_size(n,k,stride));
	dim3 blocks(cw/TILES,ch/TILES,c);
	dim3 threads(TILES, TILES);
	gpu_maxpool2di<Type><<<blocks,threads>>>(in, c, m, n, pitch, zstride, k, stride, out, oxpitch, ozstride, idcs);
	__cuda__.synchronize();
}

}; // struct

// actual interface via the static instance
static __cudapool __cudapool__;


};     // namespace umml

#endif // UMML_MAXPOOL_CUDA_INCLUDED
