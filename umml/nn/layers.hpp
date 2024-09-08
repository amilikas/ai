#ifndef UMML_LAYERS_INCLUDED
#define UMML_LAYERS_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Artificial neural network layers.

 FILE:     layers.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: ann, neural networks
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 umml basic types
 umml uvec
 umml umat
 umml ucub
 
 Internal dependencies:
 STL strings
 STL file streams

 Classes
 ~~~~~~~
 * InputLayer: Input layer (activations hold the data)
 * DenseLayer: Dense layer (fully connected).
 * DropoutLayer: Dropout regularizer.
 * Conv2DLayer: 2D Convolutional layer.
 * Maxpool2DLayer: 2D Maxpooling layer.
 * Meanpool2DLayer: 2D Meanpooling layer.
 * SoftmaxLayer: Softmax output layer.
 * SigmoidLayer: for convenience (Dense layer with Logistic activation).
 * RNNCells: Simple recurrent cells (RNNs),
 * LSTMCells: LSTM cells (RNNs)

 Notes
 ~~~~~
 alloc() and init() methods are called by the FFNN, after layer's parameters have been determined.
 gprop member is set by backprop algorithm.
 
 forward(vectors) and backward(vectors) versions are added because in the new design,
 RNN's layers and FFNN's layers both inherit from the same base class (Layer). See notes
 in rnncells.hpp for more info.
  
 TODO
 ~~~~
 * Error handling in save() and load() methods.
*/

#include "../types.hpp"
#include "../umat.hpp"
#include "../ucub.hpp"
#include "../utils.hpp"

#include <fstream>


namespace umml {


// layer types
enum {
	InputLayerType,
	DenseLayerType,
	Conv2DLayerType,
	MaxPool2DLayerType,
	MeanPool2DLayerType,
	DropoutLayerType,
	SoftmaxLayerType,
	FFNN_Layers_Size,
	
	RNNCellType=FFNN_Layers_Size,
	LSTMCellType,
	GRUCellType,
	RNN_Cells_Size,
};

// weights initialization policy
enum {
	RandomWeights,    // init network's weights
	DoNotInitWeights, // do not init netrork's weights
};

// RNNs: stacked or not cells
enum {
	SingleOutput,
	EmitAllOutputs,
};



template <typename Type>
class Layer
{
 using Vect = uvec<Type>;
 using Matr = umat<Type>;
 using Cube = ucub<Type>;
 
 public:
	// member data
	int          _type;      // type of the layer
	int          _n;         // number of neurons
	int          _c;         // number of connections per neuron
	dims4        _shape;     // shape of the layer's output
	Matr         _a;         // activations (nsamples x n)
	int          _a_len;     // activations length (eg a_len=n for dense layers)
	int          _a_size;    // activations size including GPU padding (a_size=PAD(a_len) for dense layers)
	Matr         _w;         // weights matrix (n x c)
	Vect         _b;         // biases (n)
	bool         _gprop;     // propagate error gradient or not
	bool         _trainable; // if trainable=false, weights are not updated in backpropagation
	int          _f;         // activation function (f+1 is the derivative)
	std::string  _name;      // layer's name
	Layer*       _prev;      // previous layer in the neural network
	Layer*       _next;      // next layer in the neural network

	// constructors, destructor
	Layer(): _n(0), _c(0), _a_len(0), _a_size(0), _gprop(true), _trainable(true), _f(fLinear) { 
		_prev  = _next = nullptr;
		_shape = {0,0,0,0}; 
	} 
	Layer(int __neurons): _n(__neurons), _c(0), _a_len(_n), _gprop(true), _trainable(true), _f(fLinear) {
		_a_size = 0; // will be set later by alloc()
		_prev  = _next = nullptr;
		_shape = {_n,1,1,1}; 
	} 
	virtual ~Layer() {}

	// sends layer's data to CPU or GPU memory. override to send extra data
	virtual void to_device(device __dev) {
		_a.to_device(__dev);
		_w.to_device(__dev);
		_b.to_device(__dev);
	}

	// clone layer's common data
	void clone_data(Layer* __cl) const {
		__cl->_n = _n;
		__cl->_c = _c;
		__cl->_shape = _shape;
		__cl->_a.resize_like(_a);
		__cl->_a.set(_a);
		__cl->_a_len = _a_len;
		__cl->_a_size = _a_size;
		__cl->_w.resize_like(_w);
		__cl->_w.set(_w);
		__cl->_b.resize_like(_b);
		__cl->_b.set(_b);
		__cl->_gprop = _gprop;
		__cl->_trainable = _trainable;
		__cl->_f = _f;
		__cl->_name = _name;
	}
	
	std::string get_name() const { return _name; }
	
	// trainable layer
	bool    is_trainable() const { return _trainable; }
	void    set_trainable(bool __trainable) { _trainable = __trainable; }

	// clones itself, clone_data() must be called in the derrived class, plus extra data
	virtual Layer* clone() const = 0;
	
	// parameter allocation and initialization
	virtual bool alloc() { return true; }
	virtual void init(int init_method) {}

	// reset layer's state (used for RNNs)
	virtual void reset(int __timesteps) {}
	
	// forward pass: batch version (samples) and single sample version (in, out)
	virtual void forward(const Matr& __samples) = 0;
	virtual void training_forward(const Matr& __samples) { forward(__samples); }

	// returns a vector that references a part of layer's output
/*	
	virtual Vect output_copy(int __sample, int __startpos, int __len) const {
		int sz = _shape.x * _shape.y * _shape.z;
		if (__startpos < 0) __startpos = _a.xdim()/sz + __startpos;
		assert((__startpos+__len)*sz <= _a.xdim());
Vect out; out.const_reference(a.row_cptr(sample)+startpos*sz, len*sz);
		return out;
	}
*/

	// back propagates the error gradients 'ge' through gradients 'g'.
	// calculates deltas for weights and biases.
	virtual void backward(const Matr& __ge, Matr& __g, Vect& __dbs, Vect& __dws) {}
	
	// parameters
	virtual int  grad_len() const { return _prev != nullptr ? _prev->_a_len : 0; }
	virtual int  grad_size() const { return _prev != nullptr ? _prev->_a_size : 0; }
	virtual int  weights_size() const { return _w.len(); }
	virtual int  biases_size() const { return _b.len(); }

	// return a vector with weights dimentions (RNN cells have multiple weights)
	virtual std::vector<dims4> weights_dims() const { return std::vector<dims4>(1, _w.dims()); }
	
	// layer's parameters (type, shape, activation function, etc)
	virtual std::string get_parameters(char sep=',') const {
		return std::to_string(_type) + sep;
	}
	
	// get weights, biases and their dimensions
	// any padding for GPU memory is _not_ preserved
	virtual void get_trainable_parameters(Vect& __weights, Vect& __biases) const {
		if (_w.len()) {
			__weights.resize(_w.len());
			_w.flatten(__weights, false);
		}
		if (_b.len()) {
			__biases.resize(_b.len());
			__biases.set(_b);
		}
	}

	// set weights and biases
	virtual void set_trainable_parameters(const Vect& __weights, const Vect& __biases) {
		assert(_w.len()==__weights.len());
		assert(_b.len()==__biases.len());
		if (_w.len()) _w.inflate(__weights, false);
		if (_b.len()) _b.set(__biases);
	}

	// accumulate deltas for biases and weights calculated from a batch of samples
	virtual void accumulate(Vect& __db, Vect& __dw, const Vect& __dbs, const Vect& __dws) {
		int nsamples = _a.ydim();
		__db.resize(_b.len());
		__dbs.reshape(nsamples, _b.len(), _b.xsize(), 0).reduce_sum(__db, AxisX);
		__dw.resize(_w.zsize());
		__dws.reshape(nsamples, _w.ysize()*_w.xsize(), _w.ysize()*_w.xsize(), nsamples).reduce_sum(__dw, AxisX);

//std::cout << "*** dbs: " << __dbs.format(2, 5, ' ', 24) << "\n";
//std::cout << "*** dws: " << __dws.format(2, 5, ' ', 24) << "\n";
	}
	
	// update weights and biases using deltas 'dw' and 'db'
	virtual void update(const Vect& __db, const Vect& __dw) {
		_b.plus(__db, -1);
		_w.plus(__dw.reshape(_w.ydim(), _w.xdim(), _w.xsize(), _w.ysize()), -1);
	}
	
	// write to a file stream
	virtual bool write(std::ofstream& __os) {
		int len;
		__os.write((char*)&_n, sizeof(int));
		//os.write((char*)&c, sizeof(int));
		//os.write((char*)&shape, sizeof(dim3));
		len = _w.ydim();
		__os.write((char*)&len, sizeof(int));
		len = _w.xdim();
		__os.write((char*)&len, sizeof(int));
		len = _b.len();
		__os.write((char*)&len, sizeof(int));
		double val;
		int dev = _w.dev();
		_w.to_cpu();
		_b.to_cpu();
		for (int i=0; i<_w.ydim(); ++i) {
			for (int j=0; j<_w.xdim(); ++j) {
				val = static_cast<double>(_w(i,j));
				__os.write((char*)&val, sizeof(double));
			}
		}
		for (int i=0; i<_b.len(); ++i) {
			val = static_cast<double>(_b(i));
			__os.write((char*)&val, sizeof(double));
		}
		_w.to_device(dev);
		_b.to_device(dev);
		return true;
	}
	
	// read from a file stream
	virtual bool read(std::ifstream& __is) {
		int len, w_rows, w_cols, b_len;
		__is.read((char*)&len, sizeof(int));
		if (len != _n) return false;
		//is.read((char*)&c, sizeof(int));
		//is.read((char*)&shape, sizeof(dim3));
		__is.read((char*)&w_rows, sizeof(int));
		__is.read((char*)&w_cols, sizeof(int));
		__is.read((char*)&b_len, sizeof(int));
		if (_w.ydim() != w_rows || _w.xdim() != w_cols || _b.len() != b_len) return false;
		int dev = _w.dev();
		_w.to_cpu();
		_b.to_cpu();
		double val;
		for (int i=0; i<_w.ydim(); ++i) {
			for (int j=0; j<_w.xdim(); ++j) {
				__is.read((char*)&val, sizeof(double));
				_w(i,j) = static_cast<Type>(val);
			}
		}
		for (int i=0; i<_b.len(); ++i) {
			__is.read((char*)&val, sizeof(double));
			_b(i) = static_cast<Type>(val);
		}
		_w.to_device(dev);
		_b.to_device(dev);
		return true;
	}
	
	// returns a std::string with layer's output shape
	std::string output_shape() const {
		std::stringstream ss;
		std::string dim_t, dim_z, dim_y;
		ss << "[" << _shape.x;
		if (_shape.t > 1) dim_t = std::string(",") + std::to_string(_shape.t);
		if (_shape.z > 1) dim_z = std::string(",") + std::to_string(_shape.z);
		if (_shape.y > 1) dim_y = std::string(",") + std::to_string(_shape.y);
		ss << dim_y << dim_z << dim_t << "]";
		return ss.str();
	}	

	// returns a std::string with info about the layer
	virtual std::string info() const = 0;
};


}; // namespace umml


#include "layers/input.hpp"
#include "layers/dense.hpp"
#include "layers/softmax.hpp"
#include "layers/dropout.hpp"
#include "layers/maxpool2d.hpp"
#include "layers/conv2d.hpp"

/*
#include "layers/rnncells.hpp"
*/

#endif // UMML_LAYERS_INCLUDED
