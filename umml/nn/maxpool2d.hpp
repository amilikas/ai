#ifndef UMML_MAXPOOL2D_INCLUDED
#define UMML_MAXPOOL2D_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Artificial neural network layers.

 FILE:     maxpool2d.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: ann, neural networks
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 Do _NOT_ include this file directly, it is automatically included 
 by layers.hpp.

 MaxPool2DLayer
 ~~~~~~~~~~~~~~
 Max pooling layer. 

 [1] Jiangming Yao: Convolutional Neural Networks: Step by Step
 https://jmyao17.github.io/Machine_Learning/Neural_Network/CNN-1/CNN_Build.html
*/ 

#include "../op/maxpool.hpp"


namespace umml {


template <typename Type>
void maxpool2d_gradient_set(umat<Type>& __g, const umat<Type>& __ge, const umat<int>& __idx);


template <typename Type=float>
class MaxPool2DLayer: public Layer<Type>
{
 using Vect = uvec<Type>;
 using Matr = umat<Type>;
 using Cube = ucub<Type>;

 public:
	int       _k;       // kernel size 
	int       _stride;  // kernel stride
	umat<int> _argm;    // indeces of max elements (used in back propagation)
	int       _prev_nsamples;
	using Layer<Type>::_n;
	using Layer<Type>::_c;
	using Layer<Type>::_shape;
	using Layer<Type>::_a;

	// constructors
	MaxPool2DLayer(int __k=2, int __stride=2): Layer<Type>(), _k(__k), _stride(__stride) {
		this->_name = "Maxpool2D";
		this->_type = MaxPool2DLayerType; 
	}

	// sends layer's data to CPU or GPU memory. override to send extra data
	void to_device(device __dev) override {
		Layer<Type>::to_device(__dev);
		_argm.to_device(__dev);
	}

	// clones itself
	Layer<Type>* clone() const override {
		MaxPool2DLayer* cl = new MaxPool2DLayer();
		this->clone_data(cl);
		cl->_k = _k;
		cl->_stride = _stride;
		return (Layer<Type>*)cl;
	}
	
	// allocation and initialization
	bool alloc() override {
		Layer<Type>* pl	= this->_prev;
		if (pl==nullptr) return false;
		_shape.x = pooling_size(pl->_shape.x, _k, _stride);
		_shape.y = pooling_size(pl->_shape.y, _k, _stride);
		_shape.z = pl->_shape.z;
		this->_a_len  = _shape.x*_shape.y*_shape.z;
		this->_a_size = this->_w.xpadding(_shape.x*_shape.y) * _shape.z;
		_n = 1;
		return true;
	}
	
	void init(int init_method) override {
		_prev_nsamples = 0;
		_argm.force_padding(false);
	}

	// layer's parameters (type, kernel size, stride)
	std::string get_parameters(char sep=',') const override {
		return Layer<Type>::get_parameters(sep) + std::to_string(_k) + sep + std::to_string(_stride) + sep;
	}

	// forward pass
	void forward(const Matr& __samples) override {
		dims4 pshape = ((Layer<Type>*)this->_prev)->_shape;
		_a.resize(__samples.ydim(), this->_a_len);

		for (int sample=0; sample<__samples.ydim(); ++sample) {
			uc_ref<Type> in3  = __samples.row(sample).reshape(_shape.z, pshape.y, pshape.x, pshape.x, pshape.y);
			uc_ref<Type> out3 = _a.row(sample).reshape(_shape.z, _shape.y, _shape.x, _shape.x, _shape.y);
			maxpool2d(in3, _k, _stride, out3);
		}

		/*
		#pragma omp parallel for num_threads(openmp<>::threads)
		for (int sample=0; sample<nsamples; ++sample) {
			Vect out; out.reference(a.row_ptr(sample), asize);
			Vect in = samples.row(sample);
			for (int z=0; z<shape.z; ++z) {
				Matr in_ref, out_ref;
				in_ref.const_reference(&in(z*in_sz), pl->shape.y, pl->shape.x);
				out_ref.reference(&out(z*sz), shape.y, shape.x);
				pool.apply(in_ref, out_ref);
			}
		}
		*/
	}

	// forward pass (during training)
	// argm matrix stores the positions (p=y*ncols+x) of the max elements
	void training_forward(const Matr& __samples) override {
		dims4 pshape = ((Layer<Type>*)this->_prev)->_shape;
		_a.resize(__samples.ydim(), this->_a_len);
		_argm.resize_like(_a);

		for (int sample=0; sample<__samples.ydim(); ++sample) {
			uc_ref<Type> in3  = __samples.row(sample).reshape(_shape.z, pshape.y, pshape.x, pshape.x, pshape.y);
			uc_ref<Type> out3 = _a.row(sample).reshape(_shape.z, _shape.y, _shape.x, _shape.x, _shape.y);
			uc_ref<int>  arg3 = _argm.row(sample).reshape(_shape.z, _shape.y, _shape.x, _shape.x, _shape.y);
			maxpool2d(in3, _k, _stride, out3, arg3);
		}

/*		
		Layer<Type>* pl	= (Layer<Type>*)this->prev;
		int in_sz = pl->shape.x * pl->shape.y;
		int sz = shape.x * shape.y;
		int nsamples = samples.nrows();
		a.resize(nsamples, asize);
		argm.resize(nsamples, asize);
		
		#pragma omp parallel for num_threads(openmp<>::threads)
		for (int sample=0; sample<nsamples; ++sample) {
			for (int z=0; z<shape.z; ++z) {
				Matr in_ref, a_ref;
				imat arg_ref;
				in_ref.const_reference(&samples.row_cptr(sample)[z*in_sz], pl->shape.y, pl->shape.x);
				a_ref.reference(&a.row_ptr(sample)[z*sz], shape.y, shape.x);
				arg_ref.reference(&argm.row_ptr(sample)[z*sz], shape.y, shape.x);
				pool.apply(in_ref, a_ref, arg_ref);
			}
		}
*/
	}

	// The position of the max is the input value that ultimately influenced the output, 
	// and therefore the cost. Backprop is computing gradients with respect to the cost, 
	// so anything that influences the ultimate cost should have a non-zero gradient [1]. 
	void backward(const Matr& __ge, Matr& __g, Vect&, Vect&) override {
		// if error gradient should not propagate back to the previous layers, do nothing
		if (!this->_gprop) return;
		Layer<Type>* pl	= (Layer<Type>*)this->_prev;

		// dL/dX
		__g.resize(_a.ydim(), pl->_a_len);
		__g.zero_active_device();
		maxpool2d_gradient_set(__g, __ge, _argm);
	}

	void accumulate(Vect& db, Vect& dw, const Vect& dbs, const Vect& dws) override {
		// nothing to be done here
	}

	void update(const Vect& db, const Vect& dw) override {
		// nothing to be done here
	}

	// write to a file stream
	bool write(std::ofstream& os) override {
		os.write((char*)&_k, sizeof(int));
		os.write((char*)&_stride, sizeof(int));
		return Layer<Type>::write(os);
	}
	
	// read from a file stream
	bool read(std::ifstream& is) override {
		int i;
		is.read((char*)&i, sizeof(int));
		if (i != _k) return false;
		is.read((char*)&i, sizeof(int));
		if (i != _stride) return false;
		return Layer<Type>::read(is);
	}

	// returns a std::string with info about the layer
	std::string info() const override {
		size_t nparams = 0;
		size_t bytes = this->_a_size*sizeof(Type);
		std::stringstream ss;
		ss << this->_name << " shape:" << this->output_shape() <<  " (k:" << _k << ",s:" << _stride << "), " 
		   << "parameters:" << nparams << ", memory:" << memory_footprint(bytes);
		return ss.str();
	}
};


//
// gradient_set
//

template <typename Type>
void cpu_maxpool2d_grad(Type* g, const Type* ge, const int* idx, int gpitch, int gepitch, int ipitch, int m, int n)
{
	for (int y=0; y<m; ++y) 
	for (int x=0; x<n; ++x) g[y*gpitch+idx[y*ipitch+x]] = ge[y*gepitch+x];
	
	/*
	int insz = pl->_shape.x*pl->_shape.y;
	int sz = _shape.x*_shape.y;
	for (int sample=0; sample<_a.ydim(); ++sample) {
		for (int z=0; z<_shape.z; ++z) {
			uv_ref<Type> g_ref(__g.row(sample).offset(z*insz), __g.dev(), insz, insz);
			um_ref<Type> ge_ref(__ge.row(sample).offset(z*sz), __ge.dev(), _shape.y, _shape.x, _shape.x, _shape.y);
			um_ref<int> arg_ref(_argm.row(sample).offset(z*sz), _argm.dev(), _shape.y, _shape.x, _shape.x, _shape.y);
			for (int i=0; i<_shape.y; ++i)
			for (int j=0; j<_shape.x; ++j) {
				g_ref(arg_ref(i,j)) = ge_ref(i,j);
			}
		}
	}
	*/ 
}


#if defined(__USE_CUDA__)

template <typename Type>
__global__ void gpu_maxpool2d_grad(Type* g, const Type* ge, const int* idx, int gpitch, int gepitch, int ipitch, int m, int n)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (y < m && x < n)
		g[y*gpitch+idx[y*ipitch+x]] = ge[y*gepitch+x];
}

template <typename Type>
void cuda_maxpool2d_grad(Type* g, const Type* ge, const int* idx, int gpitch, int gepitch, int ipitch, int m, int n)
{
	dim3 blocks(DIMPAD(n)/TILES, DIMPAD(m)/TILES);
	dim3 threads(TILES, TILES);
	gpu_maxpool2d_grad<Type><<<blocks,threads>>>(g, ge, idx, gpitch, gepitch, ipitch, m, n);
	__cuda__.synchronize();
}

#elif defined(__USE_OPENCL__)

std::string ocl_maxpool2d_grad_code = R"(
__kernel void __name__(__global __type__* g, __global __type__* ge, __global int* idx, 
					   int gpitch, int gepitch, int ipitch, int m, int n) 
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (y < m && x < n)
		g[y*gpitch+idx[y*ipitch+x]] = ge[y*gepitch+x];
}
)";

struct __maxpool2dgrad {

cl::Kernel fgrad;
cl::Kernel dgrad;

__maxpool2dgrad() {
	cl::Program::Sources sources;
	__ocl__.push_source_code(sources, ocl_defines_code, "", "");
	__ocl__.push_source_code(sources, ocl_maxpool2d_grad_code, "fgrad", "float");
	__ocl__.push_source_code(sources, ocl_maxpool2d_grad_code, "dgrad", "double");
	cl::Program program = __ocl__.compile_sources(sources);
	fgrad = cl::Kernel(program, "fgrad");
	dgrad = cl::Kernel(program, "dgrad");
}

// focus on outout
template <typename Type>
void grad(cl::Buffer& g, const cl::Buffer& ge, const cl::Buffer& idx,
		  int gpitch, int gepitch, int ipitch, int m, int n) {
	cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? dgrad : fgrad);
	kernel.setArg( 0, g);
	kernel.setArg( 1, ge);
	kernel.setArg( 2, idx);
	kernel.setArg( 3, gpitch);
	kernel.setArg( 4, gepitch);
	kernel.setArg( 5, ipitch);
	kernel.setArg( 6, m);
	kernel.setArg( 7, n);
	__ocl__.execute(kernel, cl::NullRange, cl::NDRange(DIMPAD(n),DIMPAD(m)), cl::NullRange);
}

};  // struct

// actual interface via the static instance
static __maxpool2dgrad __maxpool2dgrad__;

template <typename Type>
void ocl_maxpool2d_grad(cl::Buffer& g, const cl::Buffer& ge, const cl::Buffer& idx, 
						int gpitch, int gepitch, int ipitch, int m, int n)
{
	__maxpool2dgrad__.grad<Type>(g, ge, idx, gpitch, gepitch, ipitch, m, n);
}
#endif


template <typename Type>
void maxpool2d_gradient_set(umat<Type>& __g, const umat<Type>& __ge, const umat<int>& __idx)
{
	int gpitch  = __g.xsize();
	int gepitch = __ge.xsize();
	int ipitch  = __idx.xsize();
	
	if (__g.dev() == device::GPU) {
	#if defined(__USE_CUDA__)
	cuda_maxpool2d_grad<Type>(__g.dmem(), __ge.cdmem(), __idx.cdmem(), gpitch, gepitch, ipitch, __idx.ydim(), __idx.xdim());
	#elif defined(__USE_OPENCL__)
	ocl_maxpool2d_grad<Type>(__g.dmem(), __ge.cdmem(), __idx.cdmem(), gpitch, gepitch, ipitch, __idx.ydim(), __idx.xdim());
	#endif
	return;
	}
	
	cpu_maxpool2d_grad(__g.mem(), __ge.cmem(), __idx.cmem(), gpitch, gepitch, ipitch, __idx.ydim(), __idx.xdim());
}


};     // namespace umml

#endif // UMML_MAXPOOL2D_INCLUDED
