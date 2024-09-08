#ifndef UMML_DENSE_INCLUDED
#define UMML_DENSE_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Artificial neural network layers.

 FILE:     dense.hpp
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

 DenseLayer
 ~~~~~~~~~~
 Dense layer of arificial neurons f(Σaiwi + b)
 This may be used as a hidden or an output layer.
 2d shape is used to support RNNs 
*/ 


namespace umml {


template <typename Type=float>
class DenseLayer: public Layer<Type>
{
 using Vect = uvec<Type>;
 using Matr = umat<Type>;
 using Cube = ucub<Type>;
 
 public:
	bool  _w_transposed;  // true if weights already transposed
	Matr  _wT;            // transposed weights
	int   _prev_nsamples; // previous batch size
	using Layer<Type>::_n;
	using Layer<Type>::_c;
	using Layer<Type>::_a;
	using Layer<Type>::_w;
	using Layer<Type>::_b;
	using Layer<Type>::_f;

	// constructors
	DenseLayer(): Layer<Type>() { 
		this->_f = fLinear;
		this->_name = "Dense";
		this->_type = DenseLayerType; 
	}
	DenseLayer(int __neurons, int __f=fLinear): Layer<Type>(__neurons) { 
		this->_f = __f;
		this->_name = "Dense";
		this->_type = DenseLayerType;
	}
	/*
	DenseLayer(dims3 __shape, int __f=fLinear): Layer<Type>() { 
		_n = __shape.x * __shape.y * __shape.z; // * shape.t; 
		this->_f = __f;
		this->_shape = __shape;
		this->_name  = "Dense";
		this->_type  = DenseLayerType;
	}
	*/
	
	// sends layer's data to CPU or GPU memory. override to send extra data
	void to_device(device __dev) override {
		Layer<Type>::to_device(__dev);
		_wT.to_device(__dev);
	}

	// clones itself
	Layer<Type>* clone() const override {
		DenseLayer* cl = new DenseLayer();
		this->clone_data(cl);
		return (Layer<Type>*)cl;
	}
	
	// allocation and initialization
	bool alloc() override {
		this->_a_len = _n;
		this->_a_size = _w.xpadding(_n);
		if (_c > 0) {
			_w.resize(_n, _c);
			_wT.resize(_c, _n);
		}
		_b.resize(_n);
		return true;
	}
	void init(int __init_method) override {
		if (__init_method != DoNotInitWeights) {
			Type range = std::sqrt(Type(6.) / (_n + this->_prev->_shape.x*this->_prev->_shape.y));
			_w.random_reals(-range, range);
			_b.zero_active_device();
			//std::cout << this->name << " weights initialized to -+" << range << " sum=" << w.sum() << "\n";
		}
		_w_transposed = false;
		_prev_nsamples = 0;
	}

	// layer's parameters (type, neurons, activation function)
	std::string get_parameters(char sep=',') const override {
		return Layer<Type>::get_parameters(sep) + std::to_string(_n) + sep + std::to_string(_f) + sep;
	}

	// forward pass a batch of samples.
	void forward(const Matr& __samples) override {
		int nsamples = __samples.ydim();
		_a.resize(nsamples, _n);
		if (!_w_transposed) {
			_wT.transpose(_w);
			_w_transposed = true;
		}
		_a.mul(__samples, _wT); 
		_a.apply_function(_f, Type(1), Type(1), _b);
	}
	
	// backward pass (batch of samples)
	// error gradient 'ge' from next layer, shape (nsamples,n)
	// back propagated gradient 'g', shape (nsamples,c)
	// deltas biases 'db': flattened matrix of shape (nsamples,n)
	// deltas weights 'dw': flattened cube of shape (nsamples,n,c)
	void backward(const Matr& __ge, Matr& __g, Vect& __dbs, Vect& __dws) override {
		Layer<Type>* pl = this->_prev;
		bool zero = (_a.ydim() != _prev_nsamples);
		_prev_nsamples = _a.ydim();
		
		// d = f'(a) * dL/dO, len=n
		__dbs.resize(_a.ysize()*_a.xsize());
		if (zero) __dbs.zero_active_device();
		um_ref<Type> d = __dbs.reshape(_a.ydim(), _a.xdim(), _a.xsize(), _a.ysize());
		assert(d.ysize()==__ge.ysize() && d.xsize()==__ge.xsize());
		d.set(_a);
		d.apply_function(_f+1);
		d.prod(d, __ge);
//std::cout << "a:\n" << _a.format(2,5) << "\n\nd:\n" << d.format(2,5) << "\n\nge:\n" << __ge.format(2,5) << "\n";		

		// dL/dX g = w^T.d, len=c (single sample)
		__dws.resize(_a.ydim()*_w.ysize()*_w.xsize());
		if (zero) __dws.zero_active_device();
		uc_ref<Type> dw = __dws.reshape(_a.ydim(), _w.ydim(), _w.xdim(), _w.xsize(), _w.ysize());
		if (this->_gprop) {	__g.resize(_a.ydim(), _c); if (zero) __g.zero_active_device(); }
		#if !defined(__USE_CUDA__) && !defined(__USE_OPENCL__)
		#pragma omp parallel for
		for (int sample=0; sample<_a.ydim(); ++sample) {
			uv_ref<Type> in_sample = pl->_a.row(sample);
			uv_ref<Type> d_sample = d.row(sample);
			dw.slice(sample).outer(d_sample, in_sample);
			if (this->_gprop) __g.row(sample).mul(_wT, d_sample);
		}
		#else
		dw.outer(d, pl->_a);
		if (this->_gprop) __g.mul(d, _w);
		#endif
	}

	// update trainable parameters
	void update(const Vect& __db, const Vect& __dw) override {
		Layer<Type>::update(__db, __dw);
		_w_transposed = false;
	}
	
	// write to a file stream
	bool write(std::ofstream& __os) override { return Layer<Type>::write(__os); }
	
	// read from a file stream
	bool read(std::ifstream& __is) override { return Layer<Type>::read(__is); }

	// returns a std::string with info about the layer
	std::string info() const override { 
		size_t nparams = _w.len() + _b.len();
		size_t bytes = (this->_a_size+2*_w.zsize()+_b.xsize()) * sizeof(Type);
		std::stringstream ss;
		ss << this->_name << "(" << function_name(_f) << "), shape:" << this->output_shape() << ", " 
		   << "parameters:" << nparams << ", memory:" << memory_footprint(bytes);
		return ss.str();
	}
};



/*
 SigmoidLayer
 ~~~~~~~~~~~~
 Dense layer of arificial neurons with logistic (sigmoid) activation: logistic(Σaiwi + b)
 This is a hidden or an output layer (usualy combined with logloss loss function).
 4d shape is used to support RNNs 
*/ 
template <typename Type=float>
class SigmoidLayer: public DenseLayer<Type>
{
 public:
	// constructors
	SigmoidLayer(): DenseLayer<Type>() { this->_f = fLogistic; }
	SigmoidLayer(int __neurons): DenseLayer<Type>(__neurons) { this->_f = fLogistic; }
	SigmoidLayer(dims4 __shape): DenseLayer<Type>(__shape) { this->_f = fLogistic; }
};


};     // namespace umml

#endif // UMML_DENSE_INCLUDED
