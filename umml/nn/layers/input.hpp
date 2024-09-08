#ifndef UMML_INPUT_INCLUDED
#define UMML_INPUT_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Artificial neural network layers.

 FILE:     input.hpp
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
*/


namespace umml {


template <typename Type=float>
class InputLayer: public Layer<Type>
{
 using Vect = uvec<Type>;
 using Matr = umat<Type>;
 using Cube = ucub<Type>;
 
 public:
	using Layer<Type>::_n;
	using Layer<Type>::_w;
	using Layer<Type>::_a;
	using Layer<Type>::_shape;

	// constructors
	InputLayer(): Layer<Type>() { 
		this->_gprop = this->_trainable = false;
		this->_name  = "Input";
		this->_type  = InputLayerType; 
	}
	InputLayer(int __neurons): Layer<Type>(__neurons) {
		this->_gprop = this->_trainable = false;
		this->_name  = "Input";
		this->_type  = InputLayerType; 
	}
	InputLayer(dims3 __shape): Layer<Type>() { 
		_n = __shape.x * __shape.y * __shape.z;
		this->_shape  = {__shape.x, __shape.y, __shape.z, 1};
		this->_gprop   = this->_trainable = false;
		this->_name  = "Input";
		this->_type  = InputLayerType; 
	}
	InputLayer(dims4 __shape): Layer<Type>() { 
		_n = __shape.x * __shape.y * __shape.z;
		this->_shape = __shape;
		this->_gprop   = this->_trainable = false;
		this->_name  = "Input";
		this->_type  = InputLayerType; 
	}

	// clones itself
	Layer<Type>* clone() const override {
		InputLayer* cl = new InputLayer();
		this->clone_data(cl);
		return (Layer<Type>*)cl;
	}
	
	bool alloc() override {
		this->_a_len = _n;
		this->_a_size = _w.xpadding(_n);
		return true;
	}
	
	// layer's parameters (type, shape)
	std::string get_parameters(char sep=',') const override {
		return	Layer<Type>::get_parameters(sep) +
				std::to_string(_shape.x) + sep + std::to_string(_shape.y) + sep + std::to_string(_shape.z) + sep;
	}
		
	// forward pass, copies the input samples from 'samples' to activations a
	// activations are allocated/resized properly
	void forward(const Matr& __samples) override { 
		_a.resize_like(__samples);
		_a.set(__samples);
	}

	// nothing to be done here
	void accumulate(Vect& __db, Vect& __dw, const Vect& __dbs, const Vect& __dws) override {}

	// nothing to be done here
	void update(const Vect& __db, const Vect& __dw) override {}
		
	// write to a file stream
	bool write(std::ofstream& __os) override {
		__os.write((char*)&_n, sizeof(int));
		__os.write((char*)&this->_shape, sizeof(dims3));
		return true;
	}
	
	// read from a file stream
	bool read(std::ifstream& __is) override {
		__is.read((char*)&_n, sizeof(int));
		__is.read((char*)&this->_shape, sizeof(dims3));
		return true;
	}

	// returns a std::string with info about the layer
	std::string info() const override {
		std::stringstream ss;
		size_t bytes = this->_a_size * sizeof(Type);
		ss << this->_name << ", shape:" << this->output_shape() << ", " 
		   << "parameters: 0, memory:" << memory_footprint(bytes);
		return ss.str();
	}
};


};     // namespace umml

#endif // UMML_INPUT_INCLUDED
