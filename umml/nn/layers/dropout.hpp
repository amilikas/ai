#ifndef UMML_DROPOUT_INCLUDED
#define UMML_DROPOUT_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Artificial neural network layers.

 FILE:     dropout.hpp
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

 DropoutLayer
 ~~~~~~~~~~~~
 'ratio' is the probability to exclude a neuron's activation from the output.
 uses averaging out *= 1.0 / (1.0 - ratio);
*/ 


namespace umml {


template <typename Type=float>
class DropoutLayer: public Layer<Type>
{
 using Vect = uvec<Type>;
 using Matr = umat<Type>;
 using Cube = ucub<Type>;

 public:
 	double _ratio;
	Matr   _mask;
	using  Layer<Type>::_n;
	using  Layer<Type>::_c;
	using  Layer<Type>::_a;
	using  Layer<Type>::_f;

	// constructors
	DropoutLayer(double __ratio=0.5): Layer<Type>() { 
		this->_ratio = __ratio;
		this->_f = fLinear;
		this->_name = "Dropout";
		this->_type = DropoutLayerType; 
	}

	// sends layer's data to CPU or GPU memory. override to send extra data
	void to_device(device __dev) override {
		Layer<Type>::to_device(__dev);
		_mask.to_device(__dev);
	}

	// clones itself
	Layer<Type>* clone() const override {
		DropoutLayer* cl = new DropoutLayer();
		this->clone_data(cl);
		cl->_ratio = _ratio;
		return (Layer<Type>*)cl;
	}
	
	// allocation and initialization
	bool alloc() override {
		Layer<Type>* pl	= this->_prev;
		if (pl==nullptr) return false;
		_n = pl->_n;
		_c = pl->_n;
		this->_a_len  = pl->_a_len;
		this->_a_size = pl->_a_size;
		this->_shape  = pl->_shape;
		return true;
	}
	void init(int __init_method) override {}

	// layer's parameters (type, ratio)
	std::string get_parameters(char sep=',') const override {
		return Layer<Type>::get_parameters(sep) + std::to_string(_ratio) + sep;
	}

	// forward pass a batch of samples.
	void forward(const Matr& __samples) override {
		_a.resize_like(__samples);
		_a.set(__samples);
	}
	
	// training: forward pass a batch of samples.
	void training_forward(const Matr& __samples) override {
		_a.resize_like(__samples);
		// randomize mask wrt ratio
		_mask.to_cpu();
		_mask.resize_like(__samples);
		_mask.set(1);
		for (int i=0; i<_mask.ydim(); ++i) _mask.row(i).random_ints(0, 0, _ratio);
		//_mask.random_ints(0, 1, (1-_ratio)*_mask.ydim());
		_mask.to_device(_a.dev());
		_a.prod(__samples, _mask);
		_a.mul(1.0 / (1.0-_ratio));
	}

	// backward pass (batch of samples)
	// error gradient 'ge' from the next layer, shape (nsamples,n)
	// back propagated gradient 'g', shape (nsamples,c)
	void backward(const Matr& __ge, Matr& __g, Vect& __dbs, Vect& __dws) override {
		if (this->_gprop) {
			__g.resize_like(__ge);
			__g.prod(__ge, _mask);
			__g.mul(1.0 / (1.0-_ratio));
		}
	}
	
	// returns a std::string with info about the layer
	std::string info() const override { 
		size_t nparams = 0;
		size_t bytes = this->_a_size*sizeof(Type);
		std::stringstream ss;
		ss << this->_name << ", ratio:" << _ratio << ", shape:" << this->output_shape() << ", " 
		   << "parameters:" << nparams << ", memory:" << memory_footprint(bytes);
		return ss.str();
	}
};


};     // namespace umml

#endif // UMML_DROPOUT_INCLUDED
