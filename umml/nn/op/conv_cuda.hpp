#ifndef UMML_CONV_CUDA_INCLUDED
#define UMML_CONV_CUDA_INCLUDED


namespace umml {


// padding size: ((stride-1)*n-stride+k) / 2
// convolution output size: (n-k+2*pad)/stride + 1

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

template <typename Type>
__global__ void gpu_conv2d_p2c(const Type* in, int c, int m, int n, int pitch, int zstride, 
							   int kh, int kw, int stride, Type* out, int opitch) 
{
	const int oh = (m-kh)/stride + 1;
	const int ow = (n-kw)/stride + 1;
	const int xy = blockIdx.x*blockDim.x + threadIdx.x; // 0..ow*oh
	const int z = blockIdx.y*blockDim.y + threadIdx.y;  // 0..c
	const int row_in = (xy / ow) * stride;
	const int col_in = (xy % ow) * stride;

	if (xy >= ow*oh || z >= c) return;
	const int ofs = z*kh*kw*opitch; 
	for (int i=0; i<kh; ++i)
	for (int j=0; j<kw; ++j)
		out[ofs + ((i*kw)+j)*opitch + xy] = in[z*zstride + (row_in+i)*pitch + (col_in+j)];
}

template <typename Type>
__global__ void gpu_back2d_p2c(const Type* in, int c, int m, int n, int pitch, int zstride, 
							   int kh, int kw, int stride, Type* out, int opitch) 
{
	// output(x)
	const int oh = (m-kh)/stride + 1;
	const int ow = (n-kw)/stride + 1;
	const int xy = blockIdx.x*blockDim.x + threadIdx.x; // 0..ow*oh
	const int z = blockIdx.y*blockDim.y + threadIdx.y;  // 0..c
	const int row_in = (xy / ow) * stride;
	const int col_in = (xy % ow) * stride;

	if (xy >= ow*oh || z >= c) return;
	const int ofs = z*oh*ow;
	for (int i=0; i<kh; ++i)
	for (int j=0; j<kw; ++j)
		out[((i*kw)+j)*opitch + ofs + xy] = in[z*zstride + (row_in+i)*pitch + (col_in+j)];
}

template <typename Type>
__global__ void gpu_batchconv2d_p2c(const Type* in, int b, int c, int m, int n, int pitch, int zstride, int bstride,
									int kh, int kw, int stride, Type* out, int opitch, int ozstride) 
{
	const int oh = (m-kh)/stride + 1;
	const int ow = (n-kw)/stride + 1;
	const int xy = blockIdx.x*blockDim.x + threadIdx.x;  // 0..ow*oh
	const int z  = blockIdx.y*blockDim.y + threadIdx.y;  // 0..c
	const int w  = blockIdx.z*blockDim.z + threadIdx.z;  // 0..b

	if (xy >= ow*oh || z >= c) return;
	const int row_in = (xy / ow) * stride;
	const int col_in = (xy % ow) * stride;
	const int ofs = w*ozstride + z*kh*kw*opitch;
	const int ifs = w*bstride + z*zstride;
	for (int i=0; i<kh; ++i)
	for (int j=0; j<kw; ++j)
		out[ofs + ((i*kw)+j)*opitch + xy] = in[ifs + (row_in+i)*pitch + (col_in+j)];
}

template <typename Type>
__global__ void gpu_batchback2d_p2c(const Type* in, int b, int c, int m, int n, int pitch, int zstride, int bstride, 
									int kh, int kw, int stride, Type* out, int opitch, int ozstride) 
{
	// output(x)
	const int oh = (m-kh)/stride + 1;
	const int ow = (n-kw)/stride + 1;
	const int xy = blockIdx.x*blockDim.x + threadIdx.x;  // 0..ow*oh
	const int z  = blockIdx.y*blockDim.y + threadIdx.y;  // 0..c
	const int w  = blockIdx.z*blockDim.z + threadIdx.z;  // 0..b

	if (xy >= ow*oh || z >= c) return;
	const int row_in = (xy / ow) * stride;
	const int col_in = (xy % ow) * stride;
	const int ofs = w*ozstride + z*oh*ow;
	const int ifs = w*bstride + z*zstride;
	for (int i=0; i<kh; ++i)
	for (int j=0; j<kw; ++j)
		out[ofs + ((i*kw)+j)*opitch + xy] = in[ifs + (row_in+i)*xpitch + (col_in+j)];
}

template <typename Type>
__global__ void gpu_rot180(const Type* in, int m, int k, int c, int pitch, Type* out, int opitch)
{
	const int kk = k*k;
	int y = blockIdx.x*blockDim.x + threadIdx.x; // 0..m
	int z = blockIdx.y*blockDim.y + threadIdx.y; // 0..c
	int optr = z*opitch + y*kk;
	int iptr = y*pitch + (z+1)*kk - 1;
	for (int j=0; j<kk; ++j) out[optr++] = in[iptr--];
}


//
// 2D convolution (patches2cols)
// 2D backward convolution (patches2cols)
//
// No actual need for __cudaconv class, just to keep consident with OpenCL's interface.

struct __cudaconv {
__cudaconv() {}

// focus on outout
template <typename Type>
void conv2d_p2c(const Type* in, int c, int m, int n, int pitch, int zstride, 
				int kh, int kw, int stride, Type* out, int opitch) {
	int p2c_rows, p2c_cols;
	conv2d_p2c_size(c, m, n, kh, kw, stride, &p2c_rows, &p2c_cols);
	dim3 blocks(PADX(p2c_cols)/TILES, PADY(c)/TILES);
	dim3 threads(TILES, TILES);
	gpu_conv2d_p2c<Type><<<blocks,threads>>>(in, c, m, n, pitch, zstride, kh, kw, stride, out, opitch);
	__cuda__.synchronize();
}

// focus on outout
template <typename Type>
void back2d_p2c(const Type* in, int c, int m, int n, int pitch, int zstride, 
				int kh, int kw, int stride, Type* out, int opitch) {
	int p2c_rows, p2c_cols;
	back2d_p2c_size(c, m, n, kh, kw, stride, &p2c_rows, &p2c_cols);
	dim3 blocks(PADX(p2c_cols)/TILES, PADY(c)/TILES);
	dim3 threads(TILES, TILES);
	gpu_back2d_p2c<Type><<<blocks,threads>>>(in, c, m, n, pitch, zstride, kh, kw, stride, out, opitch);
	__cuda__.synchronize();
}

// focus on outout
template <typename Type>
void batchedconv2d_p2c(const Type* in, int b, int c, int m, int n, int pitch, int zstride, int bstride,
					   int kh, int kw, int stride, Type* out, int opitch, int ozstride) {
	int p2c_rows, p2c_cols;
	conv2d_p2c_size(c, m, n, kh, kw, stride, &p2c_rows, &p2c_cols);
	dim3 blocks(PADX(p2c_cols)/TILES, c, b);
	dim3 threads(TILES,1,1);
	gpu_batchconv2d_p2c<Type><<<blocks,threads>>>(in, b, c, m, n, pitch, zstride, bstride, kh, kw, stride, out, opitch, ozstride);
	__cuda__.synchronize();
}

// focus on outout
template <typename Type>
void batchedback2d_p2c(const Type* in, int b, int c, int m, int n, int pitch, int zstride, int bstride,
					   int kh, int kw, int stride, Type* out, int opitch, int ozstride) {
	int p2c_rows, p2c_cols;
	back2d_p2c_size(c, m, n, kh, kw, stride, &p2c_rows, &p2c_cols);
	dim3 blocks(PADX(p2c_cols)/TILES, PADY(c)/TILES, b);
	dim3 threads(TILES, TILES);
	gpu_batchback2d_p2c<Type><<<blocks,threads>>>(in, b, c, m, n, pitch, zstride, bstride, kh, kw, stride, out, opitch, ozstride);
	__cuda__.synchronize();
}

template <typename Type>
void rot180(const Type* in, int m, int k, int c, int pitch, Type* out, int opitch) {
	dim3 blocks(m, c);
	dim3 threads(1,1);
	gpu_rot180<Type><<<blocks,threads>>>(in, m, k, c, pitch, out, opitch);
	__cuda__.synchronize();
}

};  // struct

// actual interface via the static instance
static __cudaconv __cudaconv__;


};     // namespace umml

#endif // UMML_CONV_CUDA_INCLUDED
