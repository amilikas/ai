#ifndef UMML_SOFTMAX_INCLUDED
#define UMML_SOFTMAX_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Artificial neural network layers.

 FILE:     softmax.hpp
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

 SoftmaxLayer
 ~~~~~~~~~~~~
 Dense (fully connected) layer with Softmax activation (classification)
 2d shape is used to support RNNs 
 
 [1] Peter Roelants: Softmax classification with cross-entropy
 https://peterroelants.github.io/posts/cross-entropy-softmax/
 
 [2] MLDawn: Back propagation through Cross Entropy and Softmax
 https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/
*/ 


namespace umml {


template <typename Type=float>
class SoftmaxLayer: public Layer<Type>
{
 using Vect = uvec<Type>;
 using Matr = umat<Type>;
 using Cube = ucub<Type>;

 public:
	bool  _w_transposed;  // transpose weights or not
	Matr  _wT;            // transposed weights
	Vect  _max;
	Vect  _sum;
	Type  _temperature;
	int   _prev_nsamples; // previous batch size
 	using Layer<Type>::_n;
	using Layer<Type>::_c;
	using Layer<Type>::_a;
	using Layer<Type>::_w;
	using Layer<Type>::_b;

	// constructors
	SoftmaxLayer(): Layer<Type>() { 
		this->_name  = "Dense(softmax)";
		this->_type  = SoftmaxLayerType; 
	}
	SoftmaxLayer(int __neurons, Type __temperature=Type(1.0)): Layer<Type>(__neurons) { 
		_temperature = __temperature;
		this->_name  = "Dense(softmax)";
		this->_type  = SoftmaxLayerType; 
	}
	/*
	SoftmaxLayer(dims4 __shape, Type __temperature=Type(1.0)): Layer<Type>() { 
		_n = __shape.x * __shape.y * __shape.z; // * shape.t;
		_temperature = __temperature;
		this->_shape = __shape;
		this->_name  = "Dense(Softmax)";
		this->_type  = SoftmaxLayerType; 
	}
	*/

	// sends layer's data to CPU or GPU memory. override to send extra data
	void to_device(device __dev) override {
		Layer<Type>::to_device(__dev);
		_wT.to_device(__dev);
		_max.to_device(__dev);
		_sum.to_device(__dev);
	}

	// clones itself
	Layer<Type>* clone() const override {
		SoftmaxLayer* cl = new SoftmaxLayer();
		this->clone_data(cl);
		cl->_temperature = _temperature;
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
			//std::cout << this->name << " weights initialized to -+" << range << "\n";
			_w.random_reals(-range, range);
			_b.zero_active_device();
		}
		_w_transposed = false;
		_prev_nsamples = 0;
	}
	
	// layer's parameters (type, neurons, temperature)
	std::string get_parameters(char sep=',') const override {
		return Layer<Type>::get_parameters(sep) + std::to_string(_n) + sep + std::to_string(_temperature) + sep;
	}
	
	// forward pass, batch of samples
	// y = in.w^T+b,  a = exp(y-max(y)) / Σexp(y-max(y))
	void forward(const Matr& __samples) override {
		int nsamples = __samples.ydim();
		_a.resize(nsamples, _n);
		_max.resize(nsamples);
		_sum.resize(nsamples);
		if (!_w_transposed) {
			_wT.transpose(_w);
			_w_transposed = true;
		}
		_a.mul(__samples, _wT);
		_a.plus_vector(_b);
		_a.reduce_max(_max, AxisY);
		_a.apply_function(fExp, 1/_temperature, Type(-1), _max, AxisY);
		_a.reduce_sum(_sum, AxisY);
		_a.divide(_sum, AxisY);
	}

	// backward pass
	// error gradient 'ge' from the next layer, shape (nsamples,n)
	// back prpagated gradient 'g', shape (nsamples,c)
	// deltas biases 'db': flattened matrix of shape (nsamples, n)
	// deltas weights 'dw': flattened cube of shape (nsamples, n, c)
	void backward(const Matr& __ge, Matr& __g, Vect& __dbs, Vect& __dws) override {
		Layer<Type>* pl = this->_prev;
		bool zero = (_a.ydim() != _prev_nsamples);
		_prev_nsamples = _a.ydim();

		// d = dL/dO, len=n
		__dbs.resize(_a.ysize()*_a.xsize());
		if (zero) __dbs.zero_active_device();
		um_ref<Type> d = __dbs.reshape(_a.ydim(), _a.xdim(), _a.xsize(), _a.ysize());
		d.set(__ge);
		d.mul(1/_temperature);
		
		// dL/dX g = w.T() * dL/dO, len=c
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
		size_t bytes = (this->_a_size+2*_w.zsize()+_b.xsize()+_max.zsize()+_sum.zsize()) * sizeof(Type);
		std::stringstream ss;
		ss << this->_name << ", ";
		if (_temperature != Type(1)) ss << "T=" << _temperature << ", ";
		ss << "shape:" << this->output_shape() << ", " 
		   << "parameters:" << nparams << ", memory:" << memory_footprint(bytes);
		return ss.str();
	}
};


}; // namespace umml

#endif // UMML_SOFTMAX_INCLUDED
