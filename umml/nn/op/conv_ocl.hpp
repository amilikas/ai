#ifndef UMML_CONV_OCL_INCLUDED
#define UMML_CONV_OCL_INCLUDED


namespace umml {


// padding size: ((stride-1)*n-stride+k) / 2
// convolution output size: (n-k+2*pad)/stride + 1

/*
Example of how 2d convolution of a matrix A(4,4) with a kernel K(2,2) is performed
as a matrix multiplication of the matrix P2C(4,9) and flattened  K (1,k*k).  
conv_width * conv_height = 9

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

// in(C,M,N), out(C*k*k,h*w)
const std::string ocl_conv2d_p2c_code = R"(
__kernel void __name__(__global __type__* in, int c, int m, int n, int pitch, int zstride, 
						int kh, int kw, int stride, __global __type__* out, int opitch) 
{
	const int oh = (m-kh)/stride + 1;
	const int ow = (n-kw)/stride + 1;
	const int xy = get_global_id(0);  // 0..ow*oh
	const int z  = get_global_id(1);  // 0..c
	const int row_in = (xy / ow) * stride;
	const int col_in = (xy % ow) * stride;

	if (xy >= ow*oh || z >= c) return;
	const int ofs = z*kh*kw*opitch; 
	for (int i=0; i<kh; ++i)
	for (int j=0; j<kw; ++j)
		out[ofs + ((i*kw)+j)*opitch + xy] = in[z*zstride + (row_in+i)*pitch + (col_in+j)];
}
)";

// in(C,M,N), c(k*k,C*h*w)
// there is a BUG!!!! not the same result as the cpu version!!!
const std::string ocl_back2d_p2c_code = R"(
__kernel void __name__(__global __type__* in, int c, int m, int n, int pitch, int zstride, 
						int kh, int kw, int stride, __global __type__* out, int opitch) 
{
	const int oh = (m-kh)/stride + 1;
	const int ow = (n-kw)/stride + 1;
	const int xy = get_global_id(0);  // 0..ow*oh
	const int z  = get_global_id(1);  // 0..c
	const int row_in = (xy / ow) * stride;
	const int col_in = (xy % ow) * stride;
	
	if (xy >= ow*oh || z >= c) return;
	const int ofs = z*oh*ow;
	for (int i=0; i<kh; ++i)
	for (int j=0; j<kw; ++j)
		out[((i*kw)+j)*opitch + ofs + xy] = in[z*zstride + (row_in+i)*pitch + (col_in+j)];
}
)";

const std::string ocl_batchconv2d_p2c_code = R"(
__kernel void __name__(__global __type__* in, int b, int c, int m, int n, int pitch, int zstride, int bstride,
						int kh, int kw, int stride, __global __type__* out, int opitch, int ozstride) 
{
	const int oh = (m-kh)/stride + 1;
	const int ow = (n-kw)/stride + 1;
	const int xy = get_global_id(0);  // 0..ow*oh
	const int z  = get_global_id(1);  // 0..c
	const int w  = get_global_id(2);  // 0..b

	if (xy >= ow*oh || z >= c) return;
	const int row_in = (xy / ow) * stride;
	const int col_in = (xy % ow) * stride;
	const int ofs = w*ozstride + z*kh*kw*opitch;
	const int ifs = w*bstride + z*zstride;
	for (int i=0; i<kh; ++i)
	for (int j=0; j<kw; ++j)
		out[ofs + ((i*kw)+j)*opitch + xy] = in[ifs + (row_in+i)*pitch + (col_in+j)];
}
)";

const std::string ocl_batchback2d_p2c_code = R"(
__kernel void __name__(__global __type__* in, int b, int c, int m, int n, int xpitch, int zstride, int bstride,
						int kh, int kw, int stride, __global __type__* out, int opitch, int ozstride) 
{
	const int oh = (m-kh)/stride + 1;
	const int ow = (n-kw)/stride + 1;
	const int xy = get_global_id(0);  // 0..ow*oh
	const int z  = get_global_id(1);  // 0..c
	const int w  = get_global_id(2);  // 0..b

	if (xy >= ow*oh || z >= c) return;
	const int row_in = (xy / ow) * stride;
	const int col_in = (xy % ow) * stride;
	const int ofs = w*ozstride + z*oh*ow;
	const int ifs = w*bstride + z*zstride;
	for (int i=0; i<kh; ++i)
	for (int j=0; j<kw; ++j)
		out[ofs + ((i*kw)+j)*opitch + xy] = in[ifs + (row_in+i)*xpitch + (col_in+j)];
}
)";

const std::string ocl_rot180_code = R"(
__kernel void __name__(__global __type__* in, int m, int k, int c, int pitch, __global __type__* out, int opitch)
{
	const int kk = k*k;
	int y = get_global_id(0); // 0..m
	int z = get_global_id(1); // 0..c
	int optr = z*opitch + y*kk;
	int iptr = y*pitch + (z+1)*kk - 1;
	for (int j=0; j<kk; ++j) out[optr++] = in[iptr--];
}
)";


//
// - 2D convolution (patches2cols)
// - 2D backward convolution (patches2cols)
// - rotate kernels 180 degrees (needed in backward propagation of 
//   error gradient in a convolutional layer)

struct __oclconv {

cl::Kernel fconv2d_p2c;
cl::Kernel dconv2d_p2c;
cl::Kernel fback2d_p2c;
cl::Kernel dback2d_p2c;
cl::Kernel fbatchconv2d_p2c;
cl::Kernel dbatchconv2d_p2c;
cl::Kernel fbatchback2d_p2c;
cl::Kernel dbatchback2d_p2c;
cl::Kernel frot180;
cl::Kernel drot180;

__oclconv() {
	cl::Program::Sources sources;
	__ocl__.push_source_code(sources, ocl_conv2d_p2c_code, "fconv2d_p2c", "float");
	__ocl__.push_source_code(sources, ocl_conv2d_p2c_code, "dconv2d_p2c", "double");
	__ocl__.push_source_code(sources, ocl_back2d_p2c_code, "fback2d_p2c", "float");
	__ocl__.push_source_code(sources, ocl_back2d_p2c_code, "dback2d_p2c", "double");
	__ocl__.push_source_code(sources, ocl_batchconv2d_p2c_code, "fbatchconv2d_p2c", "float");
	__ocl__.push_source_code(sources, ocl_batchconv2d_p2c_code, "dbatchconv2d_p2c", "double");
	__ocl__.push_source_code(sources, ocl_batchback2d_p2c_code, "fbatchback2d_p2c", "float");
	__ocl__.push_source_code(sources, ocl_batchback2d_p2c_code, "dbatchback2d_p2c", "double");
	__ocl__.push_source_code(sources, ocl_rot180_code,     "frot180",     "float");
	__ocl__.push_source_code(sources, ocl_rot180_code,     "drot180",     "double");
	cl::Program program = __ocl__.compile_sources(sources);
	fconv2d_p2c = cl::Kernel(program, "fconv2d_p2c");
	dconv2d_p2c = cl::Kernel(program, "dconv2d_p2c");
	fback2d_p2c = cl::Kernel(program, "fback2d_p2c");
	dback2d_p2c = cl::Kernel(program, "dback2d_p2c");
	fbatchconv2d_p2c = cl::Kernel(program, "fbatchconv2d_p2c");
	dbatchconv2d_p2c = cl::Kernel(program, "dbatchconv2d_p2c");
	fbatchback2d_p2c = cl::Kernel(program, "fbatchback2d_p2c");
	dbatchback2d_p2c = cl::Kernel(program, "dbatchback2d_p2c");
	frot180     = cl::Kernel(program, "frot180");
	drot180     = cl::Kernel(program, "drot180");
}

// focus on outout
template <typename Type>
void conv2d_p2c(const cl::Buffer& in, int c, int m, int n, int xpitch, int zstride, 
				int kh, int kw, int stride, cl::Buffer& out, int opitch) {
	cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? dconv2d_p2c : fconv2d_p2c);
	kernel.setArg( 0, in);
	kernel.setArg( 1, c);
	kernel.setArg( 2, m);
	kernel.setArg( 3, n);
	kernel.setArg( 4, xpitch);
	kernel.setArg( 5, zstride);
	kernel.setArg( 6, kh);
	kernel.setArg( 7, kw);
	kernel.setArg( 8, stride);
	kernel.setArg( 9, out);
	kernel.setArg(10, opitch);
	int p2c_rows, p2c_cols;
	conv2d_p2c_size(c, m, n, kh, kw, stride, &p2c_rows, &p2c_cols);
//std::cout << "conv2d_p2c: cols=" << p2c_cols << ", rows=" << p2c_rows << "\n";
	__ocl__.execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(p2c_cols),TILEPAD(c)), cl::NullRange);
}

// focus on outout
template <typename Type>
void back2d_p2c(const cl::Buffer& in, int c, int m, int n, int xpitch, int zstride, 
				int kh, int kw, int stride, cl::Buffer& out, int opitch) {
	cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? dback2d_p2c : fback2d_p2c);
	kernel.setArg( 0, in);
	kernel.setArg( 1, c);
	kernel.setArg( 2, m);
	kernel.setArg( 3, n);
	kernel.setArg( 4, xpitch);
	kernel.setArg( 5, zstride);
	kernel.setArg( 6, kh);
	kernel.setArg( 7, kw);
	kernel.setArg( 8, stride);
	kernel.setArg( 9, out);
	kernel.setArg(10, opitch);
	int p2c_rows, p2c_cols;
	back2d_p2c_size(c, m, n, kh, kw, stride, &p2c_rows, &p2c_cols);
	__ocl__.execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(p2c_cols),TILEPAD(c)), cl::NullRange);
}

// focus on outout
template <typename Type>
void batchedconv2d_p2c(const cl::Buffer& in, int b, int c, int m, int n, int xpitch, int zstride,int bstride,
				int kh, int kw, int stride, cl::Buffer& out, int opitch, int ozstride) {
	cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? dbatchconv2d_p2c : fbatchconv2d_p2c);
	kernel.setArg( 0, in);
	kernel.setArg( 1, b);
	kernel.setArg( 2, c);
	kernel.setArg( 3, m);
	kernel.setArg( 4, n);
	kernel.setArg( 5, xpitch);
	kernel.setArg( 6, zstride);
	kernel.setArg( 7, bstride);
	kernel.setArg( 8, kh);
	kernel.setArg( 9, kw);
	kernel.setArg(10, stride);
	kernel.setArg(11, out);
	kernel.setArg(12, opitch);
	kernel.setArg(13, ozstride);
	int p2c_rows, p2c_cols;
	conv2d_p2c_size(c, m, n, kh, kw, stride, &p2c_rows, &p2c_cols);
//std::cout << "conv2d_p2c: cols=" << p2c_cols << ", rows=" << p2c_rows << "\n";
	__ocl__.execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(p2c_cols),c,b), cl::NullRange);
}

// focus on outout
template <typename Type>
void batchedback2d_p2c(const cl::Buffer& in, int b, int c, int m, int n, int xpitch, int zstride,int bstride,
				int kh, int kw, int stride, cl::Buffer& out, int opitch, int ozstride) {
	cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? dbatchback2d_p2c : fbatchback2d_p2c);
	kernel.setArg( 0, in);
	kernel.setArg( 1, b);
	kernel.setArg( 2, c);
	kernel.setArg( 3, m);
	kernel.setArg( 4, n);
	kernel.setArg( 5, xpitch);
	kernel.setArg( 6, zstride);
	kernel.setArg( 7, bstride);
	kernel.setArg( 8, kh);
	kernel.setArg( 9, kw);
	kernel.setArg(10, stride);
	kernel.setArg(11, out);
	kernel.setArg(12, opitch);
	kernel.setArg(13, ozstride);
	int p2c_rows, p2c_cols;
	back2d_p2c_size(c, m, n, kh, kw, stride, &p2c_rows, &p2c_cols);
	__ocl__.execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(p2c_cols),c,b), cl::NullRange);
}

template <typename Type>
void rot180(const cl::Buffer& in, int m, int k, int c, int pitch, cl::Buffer& out, int opitch) {
	cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? drot180 : frot180);
	kernel.setArg(0, in);
	kernel.setArg(1, m);
	kernel.setArg(2, k);
	kernel.setArg(3, c);
	kernel.setArg(4, pitch);
	kernel.setArg(5, out);
	kernel.setArg(6, opitch);
	__ocl__.execute(kernel, cl::NullRange, cl::NDRange(m,c), cl::NullRange);
}

};  // struct

// actual interface via the static instance
static __oclconv __oclconv__;


};     // namespace umml

#endif // UMML_CONV_OCL_INCLUDED
