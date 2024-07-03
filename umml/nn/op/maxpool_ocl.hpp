#ifndef UMML_MAXPOOL_OCL_INCLUDED
#define UMML_MAXPOOL_OCL_INCLUDED


namespace umml {


// output size: (n-k)/stride + 1

const std::string ocl_maxpool2d_code = R"(
__kernel void __name__(__global __type__* in, int c, int m, int n, int pitch, int zstride,
					   int k, int stride, __global __type__* out, int oxpitch, int ozstride) 
{
	// output(y,x)
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);

	// corresponding top-left corner of in(iy,ix)
	const int ix = x*stride;
	const int iy = y*stride;
	const int iz = z*zstride;

	__type__ max = in[iz + iy*pitch + ix];

	#pragma unroll
	for (int i=0; i<k; ++i)
	for (int j=0; j<k; ++j) {
		__type__ _m = in[iz + (iy+i)*pitch + (ix+j)];
		if (_m > max) max = _m;
	}
	if (z < c && y < (m-k)/stride+1 && x < (n-k)/stride+1) 
		out[z*ozstride + y*oxpitch + x] = max;
}
)";

const std::string ocl_maxpool2di_code = R"(
__kernel void __name__(__global __type__* in, int c, int m, int n, int pitch, int zstride,
					   int k, int stride, __global __type__* out, int oxpitch, int ozstride, 
					   __global int* idcs) 
{
	// output(y,x)
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int z = get_global_id(2);

	// corresponding top-left corner of in(iy,ix)
	const int ix = x*stride;
	const int iy = y*stride;
	const int iz = z*zstride;

	int maxloc = iz + iy*pitch + ix;
	__type__ max = in[maxloc];

	#pragma unroll
	for (int i=0; i<k; ++i)
	for (int j=0; j<k; ++j) {
		int loc = iz + (iy+i)*pitch + (ix+j);
		__type__ _m = in[loc];
		if (_m > max) {
			max = _m;
			maxloc = loc;
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
)";


//
// 2D max pooling
//
struct __oclpool {

cl::Kernel fmaxpool2d;
cl::Kernel dmaxpool2d;
cl::Kernel fmaxpool2di;
cl::Kernel dmaxpool2di;

__oclpool() {
	cl::Program::Sources sources;
	__ocl__.push_source_code(sources, ocl_maxpool2d_code,  "fmaxpool2d",  "float");	
	__ocl__.push_source_code(sources, ocl_maxpool2d_code,  "dmaxpool2d",  "double");
	__ocl__.push_source_code(sources, ocl_maxpool2di_code, "fmaxpool2di", "float");
	__ocl__.push_source_code(sources, ocl_maxpool2di_code, "dmaxpool2di", "double");
	cl::Program program = __ocl__.compile_sources(sources);
	fmaxpool2d  = cl::Kernel(program, "fmaxpool2d");
	dmaxpool2d  = cl::Kernel(program, "dmaxpool2d");
	fmaxpool2di = cl::Kernel(program, "fmaxpool2di");
	dmaxpool2di = cl::Kernel(program, "dmaxpool2di");
}


template <typename Type>
void maxpool2d(const cl::Buffer& in, int c, int m, int n, int xpitch, int zstride, 
			   int k, int stride, cl::Buffer& out, int oxpitch, int ozstride) {
	cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? dmaxpool2d : fmaxpool2d);
	kernel.setArg( 0, in);
	kernel.setArg( 1, c);
	kernel.setArg( 2, m);
	kernel.setArg( 3, n);
	kernel.setArg( 4, xpitch);
	kernel.setArg( 5, zstride);
	kernel.setArg( 6, k);
	kernel.setArg( 7, stride);
	kernel.setArg( 8, out);
	kernel.setArg( 9, oxpitch);
	kernel.setArg(10, ozstride);
	int ch = DIMPAD(pooling_size(m,k,stride));
	int cw = DIMPAD(pooling_size(n,k,stride));
	__ocl__.execute(kernel, cl::NullRange, cl::NDRange(cw,ch,c), cl::NullRange);//cl::NDRange(16,16));
}

template <typename Type>
void maxpool2d(const cl::Buffer& in, int c, int m, int n, int xpitch, int zstride, 
			   int k, int stride, cl::Buffer& out, int oxpitch, int ozstride, cl::Buffer& idcs) {
	cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? dmaxpool2di : fmaxpool2di);
	kernel.setArg( 0, in);
	kernel.setArg( 1, c);
	kernel.setArg( 2, m);
	kernel.setArg( 3, n);
	kernel.setArg( 4, xpitch);
	kernel.setArg( 5, zstride);
	kernel.setArg( 6, k);
	kernel.setArg( 7, stride);
	kernel.setArg( 8, out);
	kernel.setArg( 9, oxpitch);
	kernel.setArg(10, ozstride);
	kernel.setArg(11, idcs);
	int ch = DIMPAD(pooling_size(m,k,stride));
	int cw = DIMPAD(pooling_size(n,k,stride));
	__ocl__.execute(kernel, cl::NullRange, cl::NDRange(cw,ch,c), cl::NullRange);//cl::NDRange(16,16));
}

}; // struct

// actual interface via the static instance
static __oclpool __oclpool__;


};     // namespace umml

#endif // UMML_MAXPOOL_OCL_INCLUDED
