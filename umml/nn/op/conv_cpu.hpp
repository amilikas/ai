#ifndef UMML_CONV_CPU_INCLUDED
#define UMML_CONV_CPU_INCLUDED


namespace umml {


// creates the matrix 'out' so that the forward convolution of the multi-channel image 'in' 
// will be performed as a matrix multiplication.
// in(C,M,N), out(C*k*k,h*w)
// https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
template <typename Type>
void cpu_conv2d_p2c(const Type* in, int c, int m, int n, int pitch, int zstride,
					int kh, int kw, int stride, Type* out, int opitch) 
{
	const int h = conv_size(m, kh, stride);
	const int w = conv_size(n, kw, stride);

	for (int z=0; z<c; ++z) {
		int px=0, py=0;
		int col=0;
		int zofs = z*kh*kw;
		for (int i=0; i<h; ++i) {
			for (int j=0; j<w; ++j) {
				for (int ki=0; ki<kh; ++ki)
				for (int kj=0; kj<kw; ++kj)
					out[(zofs+(ki*kw)+kj)*opitch + col] = in[z*zstride + (py+ki)*pitch + px+kj];
				px += stride;
				++col;
			}
			py += stride; 
			px = 0;
		}
	}
}


// creates the matrix 'out' so that the backward convolution of the multi-channel image 'in' 
// will be performed as a matrix multiplication.
// in(C,M,N), c(k*k,C*h*w)
template <typename Type>
void cpu_back2d_p2c(const Type* in, int c, int m, int n, int pitch, int zstride,
					int kh, int kw, int stride, Type* out, int opitch) 
{
	const int h = conv_size(m, kh, stride);
	const int w = conv_size(n, kw, stride);

	/*
	for (int z=0; z<c; ++z) {
		const int ofs = z*h*w;
		for (int xy=0; xy<w*h; ++xy) {
			const int row_in = (xy / w) * stride;
			const int col_in = (xy % w) * stride;
			for (int i=0; i<kh; ++i)
			for (int j=0; j<kw; ++j)
				out[((i*kw)+j)*opitch + ofs + xy] = in[z*zstride + (row_in+i)*pitch + (col_in+j)];
		}
	}
	*/
	for (int z=0; z<c; ++z) {
		int px=0, py=0;
		int col=0;
		for (int i=0; i<h; ++i) {
			for (int j=0; j<w; ++j) {
				for (int ki=0; ki<kh; ++ki)
				for (int kj=0; kj<kw; ++kj)
					out[((ki*kw)+kj)*opitch + z*w*h + col] = in[z*zstride + (py+ki)*pitch + px+kj];
				px += stride;
				++col;
			}
			py += stride; 
			px = 0;
		}
	}
}


// creates the cube 'out' so that the forward convolution of a batch of multi-channel images 'in' 
// will be performed as a matrix multiplication.
// in(W,C,M,N) -> out(W, C*kh*kw, h*w)
// https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
template <typename Type>
void cpu_batchedconv2d_p2c(const Type* in, int b, int c, int m, int n, int pitch, int zstride, int bstride,
						   int kh, int kw, int stride, Type* out, int opitch, int ozstride) 
{
	int height = conv_size(m, kh, stride);
	int width = conv_size(n, kw, stride);
	#pragma omp parallel for
	for (int w=0; w<b; ++w) {
		for (int z=0; z<c; ++z) {
			int px=0, py=0;
			int col=0;
			int zofs = z*kh*kw;
			for (int i=0; i<height; ++i) {
				for (int j=0; j<width; ++j) {
					for (int ki=0; ki<kh; ++ki)
					for (int kj=0; kj<kw; ++kj)
						out[w*ozstride + (zofs+(ki*kw)+kj)*opitch + col] = in[w*bstride + z*zstride + (py+ki)*pitch + px+kj];
					px += stride;
					++col;
				}
				py += stride; 
				px = 0;
			}
		}
	}
}


// creates the cube 'out' so that the backward convolution of the multi-channel image 'in' 
// will be performed as a matrix multiplication.
// in(C,M,N), c(k*k,C*h*w)
template <typename Type>
void cpu_batchedback2d_p2c(const Type* in, int b, int c, int m, int n, int pitch, int zstride, int bstride,
						   int kh, int kw, int stride, Type* out, int opitch, int ozstride) 
{
	int height = conv_size(m, kh, stride);
	int width = conv_size(n, kw, stride);
	#pragma omp parallel for
	for (int w=0; w<b; ++w) {
		int col=0;
		for (int z=0; z<c; ++z) {
			int px=0, py=0;
			for (int i=0; i<height; ++i) {
				for (int j=0; j<width; ++j) {
					for (int ki=0; ki<kh; ++ki)
					for (int kj=0; kj<kw; ++kj)
						out[w*ozstride + ((ki*kw)+kj)*opitch + col] = in[w*bstride + z*zstride + (py+ki)*pitch + px+kj];
					px += stride;
					++col;
				}
				py += stride; 
				px = 0;
			}
		}
	}
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
template <typename Type>
void cpu_rot180(const Type* in, int m, int k, int c, int pitch, Type* out, int opitch)
{
	const int kk = k*k;
	for (int z=0; z<c; ++z) {
		Type* optr = &out[z*opitch];
		for (int i=0; i<m; ++i) {
			const Type* iptr = &in[i*pitch + (z+1)*kk - 1];
			for (int j=0; j<kk; ++j) *optr++ = *iptr--;
		}
	}
}

};     // namespace umml

#endif // UMML_CONV_CPU_INCLUDED
