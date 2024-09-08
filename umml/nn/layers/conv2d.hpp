#ifndef UMML_CONV2D_INCLUDED
#define UMML_CONV2D_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Artificial neural network layers.

 FILE:     conv2d.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: ann, neural networks
 
 Namespace
 ~~~~~~~~~
 mml
 
 Notes
 ~~~~~
 Do _NOT_ include this file directly, it is automatically included 
 by layers.hpp.

 Conv2DLayer
 ~~~~~~~~~~~
 2D Convolutional layer.
 Input can be a 2D square image or a 3D (multi-channel) square image.
 Kernel is a 2D square matrix.
 
 [1] Pavithra Solai: Convolutions and Backpropagations
 https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c

 [2] Jiangming Yao: Convolutional Neural Networks: Step by Step
 https://jmyao17.github.io/Machine_Learning/Neural_Network/CNN-1/CNN_Build.html    
*/ 

#include "../op/conv.hpp"


namespace umml {


// padding
enum {
	Valid,
	Same,
};


//template <typename Type=float, template<typename> class ConvFunc=conv2d>
template <typename Type=float>
class Conv2DLayer: public Layer<Type>
{
 using Vect   = uvec<Type>;
 using Matr   = umat<Type>;
 using Cube   = ucub<Type>;
 using Tensor = utensor<Type>;
 
 public:
	int    _k;             // kernel size
	int    _stride;        // kernel stride
	int    _pad;           // padding size
	int    _padding;       // Valid (no padding) or Same (output size=input size)
	Matr   _w180;          // rotated kernels
	Cube   _fp2c;          // patches2cols for a batch (forward pass)
	Cube   _dwp2c;         // patches2cols for a batch (backward pass, dw)
	Cube   _gp2c;          // patches2cols for a batch (backward pass, gradient)
	Matr   _dge;           // f'(a)*ge
	Matr   _dge_pad;       // padded f'(a)*ge (used only if padding=Same)
	Tensor _dge4;          // zero-padded f'(a)*ge (g calculation in GPU)
	Tensor _in4;           // zero-padded input (used only if padding=Same)
	int    _fnsamples;     // previous samples in forward pass
	int    _bnsamples;     // previous samples in backward pass
	using Layer<Type>::_n;
	using Layer<Type>::_c;
	using Layer<Type>::_shape;
	using Layer<Type>::_a;
	using Layer<Type>::_w;
	using Layer<Type>::_f;

	// constructors
	Conv2DLayer(): Layer<Type>(), _k(3), _stride(1), _pad(0), _padding(Valid) {
		this->_f    = fLinear;
		this->_name = "Conv2D";
		this->_type = Conv2DLayerType; 
	}
	Conv2DLayer(int __filters, int __f=fLinear): 
		Layer<Type>(__filters), _k(3), _stride(1), _pad(0), _padding(Valid) { 
		this->_f    = __f;
		this->_name = "Conv2D";
		this->_type = Conv2DLayerType; 
	}
	Conv2DLayer(int __filters, int __k, int __stride, int __padding=Valid, int __f=fLinear):
		Layer<Type>(__filters), _k(__k), _stride(__stride), _pad(0), _padding(__padding) {
		this->_f    = __f;
		this->_name = "Conv2D";
		this->_type = Conv2DLayerType; 
	}

	// sends layer's data to CPU or GPU memory. override to send extra data
	void to_device(device __dev) override {
		Layer<Type>::to_device(__dev);
		_fp2c.to_device(__dev);
		_dwp2c.to_device(__dev);
		_gp2c.to_device(__dev);
		_w180.to_device(__dev);
		_dge.to_device(__dev);
		_dge_pad.to_device(__dev);
		_dge4.to_device(__dev);
	}

	// clones itself
	Layer<Type>* clone() const override {
		Conv2DLayer* cl = new Conv2DLayer();
		this->clone_data(cl);
		cl->_k = _k;
		cl->_stride = _stride;
		cl->_pad = _pad;
		cl->_padding = _padding;
		return (Layer<Type>*)cl;
	}

	// allocation and initialization
	bool alloc() override {
		Layer<Type>* pl	= this->_prev;
		if (pl==nullptr) return false;
		if (_padding==Valid) _pad = 0;
		else _pad = padding_size(pl->_shape.x, _k, _stride);
		_shape.x = conv_size(pl->_shape.x, _k, _stride, _pad);
		_shape.y = conv_size(pl->_shape.y, _k, _stride, _pad);
		_shape.z = _n;
		this->_a_len  = _shape.x*_shape.y*_shape.z;
		this->_a_size = _w.xpadding(_shape.x*_shape.y) * _shape.z;
		// allocate weights, one row per kernel (filter)
		_w.resize(_n, pl->_shape.z*_k*_k);
		_w180.resize(pl->_shape.z, _n*_k*_k);
		return true;
	}
	
	void init(int __init_method) override {
		Layer<Type>* pl	= (Layer<Type>*)this->_prev;
		if (__init_method != DoNotInitWeights) {
			Type range = std::sqrt(Type(3.) / (pl->_shape.z*pl->_shape.x*pl->_shape.y));
			//Type range = std::sqrt(Type(3.) / (pl->_shape.z*pl->_shape.x*pl->_shape.y + _shape.z*_shape.x*_shape.y));
			//std::cout << this->name << " weights initialized to -+" << range << "\n";
			_w.random_reals(-range, range);
		}
		_fnsamples = _bnsamples = 0;
	}
	
	// layer's parameters (type, filters, kernel size, stride, padding, activation function)
	std::string get_parameters(char sep=',') const override {
		return 	Layer<Type>::get_parameters(sep) + std::to_string(_n) + sep + 
				std::to_string(_k) + sep + std::to_string(_stride) + sep + 
				std::to_string(_padding) + sep + std::to_string(_f) + sep;
	}
	
	// forward pass
	void forward(const Matr& __samples) override {
		dims4 pshape = ((Layer<Type>*)this->_prev)->_shape;
		_a.resize(__samples.ydim(), this->_a_len);

		if (_padding==Valid) {
			#if !defined(__USE_CUDA__) && !defined(__USE_OPENCL__)
			#pragma omp parallel for
			for (int sample=0; sample<__samples.ydim(); ++sample) {
				uc_ref<Type> in3 = __samples.row(sample).reshape(pshape.z, pshape.y, pshape.x, pshape.x, pshape.y);
				um_ref<Type> out = _a.row(sample).reshape(_shape.z, _shape.x*_shape.y, _shape.x*_shape.y, _shape.z);
				Matr fp2c(_a.dev());
				conv2d(in3, fp2c, _w, _k, _k, _stride, _pad, out);
				out.apply_function(_f);
			}
			#else
			ut_ref<Type> in4(__samples.active_mem(), __samples.dev(), __samples.ydim(), 
						 pshape.z, pshape.y, pshape.x, pshape.x, pshape.y, pshape.x*pshape.y, __samples.xsize());
			uc_ref<Type> out3(_a.active_mem(), _a.dev(), _a.ydim(), _shape.z, _shape.x*_shape.y,
						 _shape.x*_shape.y, _shape.z, _a.xsize());
			batched_conv2d(in4, _fp2c, _w, _k, _k, _stride, _pad, out3);
			out3.apply_function(_f);
			#endif
			
		} else {
			bool zero = (__samples.ydim() != _fnsamples);
			_fnsamples = __samples.ydim();
			ut_ref<Type> in4(__samples.active_mem(), __samples.dev(), __samples.ydim(), pshape.z, pshape.y, pshape.x,
						 pshape.x, pshape.y, pshape.x*pshape.y, __samples.xsize());
			_in4.zero_padded(in4, _pad, _pad, zero);
			uc_ref<Type> out3(_a.active_mem(), _a.dev(), _a.ydim(), _shape.z, _shape.x*_shape.y,
						 _shape.x*_shape.y, _shape.z, _a.xsize());
			batched_conv2d(_in4, _fp2c, _w, _k, _k, _stride, _pad, out3);
			out3.apply_function(_f);
		}
	}
	
	// dL/dF can be computed as a valid convolution of input and ge
	void backward(const Matr& __ge, Matr& __g, Vect& /*dbs*/, Vect& __dws) override {
		Layer<Type>* pl = this->_prev;
		dims4 pshape = pl->_shape;
		int nsamples = _a.ydim();
		int unpadx = _shape.x;
		int unpady = _shape.y;
		bool zero = (nsamples != _bnsamples);
		_bnsamples = nsamples;

		__dws.resize(nsamples*_w.ysize()*_w.xsize());
		if (zero) __dws.zero_active_device();
		uc_ref<Type> dws = __dws.reshape(nsamples, _w.ydim(), _w.xdim(), _w.xsize(), _w.ysize());

		if (this->_gprop) {
			__g.resize(nsamples, pl->_a_len);
			if (zero) __g.zero_active_device();
			rot180(_w, _k, pshape.z, _w180);
		}

		if (_padding==Valid) {
			_dge.resize_like(_a);
			_dge.set(_a);
			_dge.apply_function(_f+1);
			_dge.prod(_dge, __ge);
		} else {
			unpadx = conv_size(pl->_shape.x, _k, _stride, 0);
			unpady = conv_size(pl->_shape.y, _k, _stride, 0);
			_dge_pad.resize_like(_a);
			_dge_pad.set(_a);
			_dge_pad.apply_function(_f+1);
			_dge_pad.prod(_dge_pad, __ge);
			ut_ref<Type> dgepad_ref(_dge_pad.active_mem(), _dge_pad.dev(), nsamples, _n, _shape.y, _shape.x, 
						 _shape.x, _shape.y, _shape.x*_shape.y, _dge_pad.xsize());
			_dge.resize(nsamples, _n*unpady*unpadx);
			ut_ref<Type> dge_ref(_dge.active_mem(), _dge.dev(), nsamples, _n, unpady, unpadx, 
						 unpadx, unpady, unpadx*unpady, _dge.xsize());
			dge_ref.copy2d(dgepad_ref, _pad, _pad, 0, 0, unpady, unpadx);
		}

		#if !defined(__USE_CUDA__) && !defined(__USE_OPENCL__)
		#pragma omp parallel for
		for (int sample=0; sample<nsamples; ++sample) {
			// dL/dF (dw)
			uc_ref<Type> in3 = pl->_a.row(sample).reshape(pshape.z, pshape.y, pshape.x, pshape.x, pshape.y);
			um_ref<Type> ge  = _dge.row(sample).reshape(_n, unpadx*unpady, unpadx*unpady, _n);
			um_ref<Type> dw  = dws.slice(sample);
			Matr dwp2c(_a.dev());
			backconv2d(in3, dwp2c, ge, unpady, unpadx, 1, 0, dw);
			// dL/dX (g) full conv size(shape.y, k, 1) -> input.shape
			if (this->_gprop) {
				int h_padded = unpady + 2*(_k-1);
				int w_padded = unpadx + 2*(_k-1);
				uc_ref<Type> dge_cub = _dge.row(sample).reshape(_n, unpady, unpadx, unpadx, unpady);
				Cube ge_cub(_n, h_padded, w_padded, _dge.dev());
				ge_cub.zero_padded(dge_cub, _k-1, _k-1);
				Matr gp2c(ge_cub.dev());
				int p2c_rows, p2c_cols;
				conv2d_p2c_size(ge_cub.zdim(), ge_cub.ydim(), ge_cub.xdim(), _k, _k, 1, &p2c_rows, &p2c_cols);
				int gy = _w180.ydim();
				int gx = p2c_cols;
				um_ref<Type> g = __g.row(sample).reshape(gy, gx, gx, gy);
				conv2d(ge_cub, gp2c, _w180, _k, _k, 1, 0, g);
			}
		}

		#else
		// dL/dF (dw)
		ut_ref<Type> in4(pl->_a.active_mem(), pl->_a.dev(), nsamples, pshape.z, pshape.y, pshape.x, 
						 pshape.x, pshape.y, pshape.x*pshape.y, pl->_a.xsize());
		uc_ref<Type> dge(_dge.active_mem(), _dge.dev(), nsamples, _n, unpadx*unpady, unpadx*unpady, _n, _dge.xsize());
		batched_backconv2d(in4, _dwp2c, dge, unpady, unpadx, 1, 0, dws);
		
		// dL/dX (g) full conv size(shape.y, k, 1) -> input.shape
		if (this->_gprop) {
			int h_padded = unpady + 2*(_k-1);
			int w_padded = unpadx + 2*(_k-1);
			ut_ref<Type> dge4_ref(_dge.active_mem(), _dge.dev(), nsamples, _n, unpady, unpadx,
						 unpadx, unpady, unpadx*unpady, _dge.xsize());
			_dge4.resize(nsamples, _n, h_padded, w_padded);
			_dge4.zero_padded(dge4_ref, _k-1, _k-1);
			int p2c_rows, p2c_cols;
			conv2d_p2c_size(_dge4.zdim(), _dge4.ydim(), _dge4.xdim(), _k, _k, 1, &p2c_rows, &p2c_cols);
			int gy = _w180.ydim();
			int gx = p2c_cols;
			uc_ref<Type> g3(__g.active_mem(), __g.dev(), nsamples, gy, gx, gx, gy, __g.xsize());
			batched_conv2d(_dge4, _gp2c, _w180, _k, _k, 1, 0, g3);
		}
		#endif
	}

	// update trainable parameters
	void update(const Vect& __db, const Vect& __dw) override {
		Layer<Type>::update(__db, __dw);
	}

	// write to a file stream
	bool write(std::ofstream& __os) override {
		__os.write((char*)&_k, sizeof(int));
		__os.write((char*)&_stride, sizeof(int));
		return Layer<Type>::write(__os);
	}
	
	// read from a file stream
	bool read(std::ifstream& __is) override {
		int i;
		__is.read((char*)&i, sizeof(int));
		if (i != _k) return false;
		__is.read((char*)&i, sizeof(int));
		if (i != _stride) return false;
		return Layer<Type>::read(__is);
	}

	// returns a std::string with info about the layer
	std::string info() const override {
		size_t nparams = _w.len();
		size_t n = this->_a_size + _w.size() + _w180.size() + _fp2c.size() + _dwp2c.size() + _gp2c.size() +
					_dge.size() + _dge_pad.size() + _dge4.size() + _in4.size();
		size_t bytes = n * sizeof(Type);
		std::stringstream ss;
		ss << this->_name << "(" << function_name(_f) << "), shape:" << this->output_shape() << " (" 
		   << "k:" << _k << ",s:" << _stride << ",pad:" << (_pad ? "same" : "no") << "), " 
		   << "parameters:" << nparams << ", memory:" << memory_footprint(bytes);
		return ss.str();
	}
};


};     // namespace umml

#endif // UMML_CONV2D_INCLUDED
