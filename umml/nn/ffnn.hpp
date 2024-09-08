#ifndef UMML_FFNN_INCLUDED
#define UMML_FFNN_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Feedforward Neural Network.

 FILE:     ffnn.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: ann, neural networks
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~

 Dependencies
 ~~~~~~~~~~~~
 umml uvec
 umml umat
 umml activations
 umml layers
 STL string
 
 Internal dependencies:
 STL vector
 STL file streams
  
 Usage example
 ~~~~~~~~~~~~~
 FFNN<> net;
 net.add(new InputLayer(16));           // input layer with 16 values
 net.add(new DenseLayer(8, fLogistic)); // 1st hidden layer with 8 neurons and logistic activation
 net.add(new DenseLayer(6, fLogistic)); // 2nd hidden layer with 6 neurons and logistic activation
 net.add(new DenseLayer(2));            // output layer with 2 neurons (2 outputs) and linear activation
  
 TODO
 ~~~~
  
*/

#include "layers.hpp"
#include <fstream>


namespace umml {


template <typename Type=float>
class FFNN
{
 public:
	// save/load error codes
	int _err_id;
	int _err_layer;
	device _dev;
	
	// error codes
	enum {
		OK = 0,
		NotExists,
		BadMagic,
		BadCount,
		BadType,
		BadDimensions,
		BadParameters,
		BadAlloc,
		BadIO,
	};
	
	// constructor, destructor
	FFNN(): _err_id(OK), _err_layer(-1), _dev(device::CPU) {}
	FFNN(const std::string& __name): _err_id(OK), _err_layer(-1), _dev(device::CPU), _name(__name) {}
	virtual ~FFNN() { clear(); }

	// checks if no errors occured during build/load/save
	explicit operator bool() const { return _err_id==OK; }

	void set_name(const std::string& __name) { _name = __name; }
	std::string get_name() const { return _name; }
	
	void clear() {
		for (size_t i=0; i<_layers.size(); ++i) delete _layers[i];
		_layers.clear();
	}

	// get network's active device
	device dev() const { return _dev; }

	// sends the network to CPU or GPU memory. 
	void to_device(device __dev) {
		if (_dev==__dev) return;
		#if !defined(__USE_CUDA__) && !defined(__USE_OPENCL__)
		if (__dev==device::GPU) return;
		#endif

		_dev = __dev;
		for (size_t i=0; i<_layers.size(); ++i) _layers[i]->to_device(_dev);
	}

	FFNN& operator =(const FFNN& __other) = delete;

	// clones itself to other FFNN
	void clone_to(FFNN& __to) const {
		__to.clear();
		__to._dev = _dev;
		__to._err_id = _err_id;
		__to._err_layer = _err_layer;
		for (size_t i=0; i<_layers.size(); ++i)
			__to.add(_layers[i]->clone(), DoNotInitWeights);
	}
	
	// adds a layer with 'neurons' neurons and activation function 'actfunc'
	bool add(Layer<Type>* __l, int __init_method=RandomWeights) {
		__l->to_device(_dev);
		__l->_c = 0;
		if (!_layers.empty()) {
			__l->_c = _layers.back()->_a_len;
			__l->_prev = _layers.back();
			_layers.back()->_next = __l;
		}
		bool ok = __l->alloc();
		if (ok) {
			__l->init(__init_method);
		} else {
			_err_id = BadAlloc;
			_err_layer = (int)_layers.size();
		}
		_layers.push_back(__l);
		return ok;
	}

	// reset state for every layer
	void reset() {
		Layer<Type>* l = _layers[1];
		while (l != nullptr) {
			l->reset();
			l = l->_next;
		}
	}

	// feed raw input samples through the network
	void forward(const umat<Type>& __samples) {
		Layer<Type>* l = _layers[0];
		l->forward(__samples);
		l = l->_next;
		while (l != nullptr) {
			l->forward(l->_prev->_a);
			l = l->_next;
		}
	}

	// feed input samples from X, indexed by idx, through the network
	void forward(const umat<Type>& __X, const std::vector<int>& __idx) {
		Layer<Type>* l = _layers[0];
		l->_a.resize(__idx.size(), __X.xdim());
		l->_a.copy_rows(__X, __idx);
		l = l->_next;
		while (l != nullptr) {
			l->forward(l->_prev->_a);
			l = l->_next;
		}
	}

	// feed raw input samples through the network
	void training_forward(const umat<Type>& __samples) {
		Layer<Type>* l = _layers[0];
		l->training_forward(__samples);
		l = l->_next;
		while (l != nullptr) {
			l->training_forward(l->_prev->_a);
			l = l->_next;
		}
	}
	
	// feed input samples from X, indexed by idx, through the network
	void training_forward(const umat<Type>& __X, const std::vector<int>& __idx) {
		Layer<Type>* l = _layers[0];
		l->_a.resize(__idx.size(), __X.xdim());
		l->_a.copy_rows(__X, __idx);
		l = l->_next;
		while (l != nullptr) {
			l->training_forward(l->_prev->_a);
			l = l->_next;
		}
	}
	
	// copies output's layer activations to matrix 'out'
	void output(umat<Type>& __output) {
		__output.set(_layers.back()->_a);
	}
	
	void predict(const umat<Type>& __X, umat<Type>& __Y) {
		int xrows = __X.ydim();
		int xcols = __X.xdim();
		int ycols = output_layer()->_a_len;
		__Y.resize(xrows, ycols);
		//int threads = openmp<>::threads;
		int threads = 256;
		int batch_size = threads >= 1 && threads < xrows ? threads : xrows;
		int nbatches = xrows / batch_size;
		int nleftover = xrows % batch_size;
		umat<Type> Xbatch(batch_size+nleftover, xcols, __X.dev());
		for (int b=0; b<nbatches; ++b) {
			int bs = batch_size;
			if (b==nbatches-1) bs += nleftover;
			Xbatch.resize(bs, xcols);
			Xbatch.copy_rows(__X, b*batch_size, bs, 0);
			forward(Xbatch);
			__Y.copy_rows(_layers.back()->_a, 0, bs, b*batch_size);
		}
	}
	
	// layers 
	int  nlayers() const { return (int)_layers.size(); }
	Layer<Type>* input_layer() const { return _layers[0]; }
	Layer<Type>* output_layer() const { return _layers.back(); }
	Layer<Type>* get_layer(int __idx) const {
		if (__idx < 0) __idx = nlayers() + __idx;
		assert(__idx >= 0 && __idx <= nlayers());
		return _layers[__idx]; 
	}

	// saves the weights of the network in a disk file
	bool save(const std::string& __filename);
	
	// loads the weights of the network from a disk file
	bool load(const std::string& __filename);

	// returns a std::string with the description of the error in err_id
	std::string error_description() const;
	
	// returns a std::string with information about the network
	std::string info() const;
	
 protected:
	// member data
	std::string _name;
	std::vector<Layer<Type>*> _layers;
};



////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////


template <typename Type>
bool FFNN<Type>::save(const std::string& __filename)
{
	int n;
	std::ofstream os;
	os.open(__filename, std::ios::out|std::ios::binary);
	if (!os.is_open()) { _err_id=NotExists; return false; }
	os << "FFNN." << "00"/*version*/ << ".";
	// number of layers
	n = nlayers(); 
	os.write((char*)&n, sizeof(int));
	// save layers
	Layer<Type>* l = _layers[0];
	while (l != nullptr) {
		os.write((char*)&l->_type, sizeof(int));
		l->write(os);
		l = l->_next;
	}
	os.close();	
	if (!os) { _err_id=BadIO; return false; }
	return true;
}
	
template <typename Type>
bool FFNN<Type>::load(const std::string& __filename)
{
	char buff[8];
	int n;
	std::ifstream is;
	is.open(__filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { _err_id=NotExists; return false; }
	is.read(buff, 5);
	buff[5] = '\0';
	if (std::string(buff) != "FFNN.") { _err_id=BadMagic; return false; }
	is.read(buff, 3);
	if (buff[2] != '.') { _err_id=BadMagic; return false; }
	//version = ('0'-buff[0])*10 + '0'-buff[1];
	// number of layers
	is.read((char*)&n, sizeof(int));
	if (n != (int)_layers.size()) { _err_id=BadCount; return false; }
	// layers' neurons and type of activation
	for (int i=0; i<n; ++i) {
		// layer's type (and kernel size)
		int type, ltype;
		bool ok;
		is.read((char*)&type, sizeof(int));
		ltype = _layers[i]->_type;
		ok = (type==ltype || (type==DenseLayerType && ltype==SoftmaxLayerType) || (type==SoftmaxLayerType && ltype==DenseLayerType));
		if (!ok) { 
			_err_id=BadType; 
			_err_layer = i;
			return false; 
		}
		ok = _layers[i]->read(is);
		if (!ok) {
			_err_id = BadParameters;
			_err_layer = i;
			return false;
		}
	}
	is.close();
	if (!is) { _err_id=BadIO; return false; }
	return true;
}


template <typename Type>
std::string FFNN<Type>::error_description() const 
{
	std::stringstream ss;
	switch (_err_id) {
		case NotExists: return "File not exists.";
		case BadMagic: return "File has unknown magic number.";
		case BadCount: return "Unexpected number of layers.";
		case BadType: ss << "Wrong layer type"; break;
		case BadDimensions: ss << "Wrong dimensions."; break;
		case BadParameters: ss << "Error reading weights/biases"; break;
		case BadAlloc: ss << "Memory allocation error"; break;
		case BadIO: return "File IO error.";
		default: return "";
	};
	ss << " (layer: " << _err_layer << ")";
	return ss.str();
}


template <typename Type>
std::string FFNN<Type>::info() const 
{
	std::stringstream ss;
	ss << "Network: " << _name << "\n";
	for (int lx=0; lx<nlayers(); ++lx) 
		ss << get_layer(lx)->info() << "\n";
	return ss.str();
}


};     // namespace umml

#endif // UMML_FFNN_INCLUDED
