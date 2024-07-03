#ifndef UMML_BACKPROP_INCLUDED
#define UMML_BACKPROP_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Backpropagation with stochastic gradient descent.

 FILE:     backprop.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: supervised, gradient descent, neural networks
 
 Namespace
 ~~~~~~~~~
 mml
 
 Notes
 ~~~~~
 It is often combined with minmaxscaler (scaler.hpp) and onehot_enc (preproc.hpp)
  
 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 umml activations
 umml feedforward neural network
 umml::gdstep gradient descent step methods (optimizers)
 umml::lossfunc loss functions
 STL string
 
 Internal dependencies:
 STL vector
 STL file streams
 umml algo
 umml utils
 umml rand
 umml logger
 umml preprocessing (1hot encoder)
 umml metrics (accuracy)
  
 Usage example
 ~~~~~~~~~~~~~
 FFNN<> net;
 net.add(input layer);
 net.add(hidden layer, activation);
 net.add(output layer, activation);
 Backprop<> bp;
 gdstep::learnrate<> st(learnrate);
 lossfunc::rss<> loss;
 bp.train(net, loss, st, X_train, Y_train);
 net.predict(X_test, Y_pred);
 
 * Dataset spliting and scaling:
 dataset<> ds(Xdata, Ytargets);
 ds.split_train_test_sets(0.75);
 ds.get_splits(X_train, X_test, Y_train, Y_test);
 minmaxscaler<> xscaler;
 xscaler.fit_transform(X_train);
 xscaler.transform(X_test);
 
 * For OpenMP parallelism, compile with -D__USE_OPENMP__ -fopenmp
 mml_set_openmp_threads(-1); // -1 for std::thread::hardware_concurrency
 bp.train(net, loss, st, X_train, Y_train);
  
 TODO
 ~~~~ 
*/

#include "ffnn.hpp"
#include "lossfunc.hpp"
#include "gdstep.hpp"
#include "shuffle.hpp"
#include "../algo.hpp"
#include "../utils.hpp"
#include "../preproc.hpp"
#include "../metrics.hpp"
#include "../logger.hpp"


namespace umml {


/*
 Backprop
 
 Uses the delta rule.
 Gradiend descent step methods: 
 * learnrate: typical value: 0.01
 * momentum: typical value: 0.9
 * adam: see [2]

 Stopping conditions:
 * max_iters reached.
 * total error < tolerance.
 
 [1] Matt Mazur: A Step by Step Backpropagation Example
 https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ 
 
 [2] Kingma, Lei Ba: Adam: A Method for Stochastic Optimization.
 https://arxiv.org/pdf/1412.6980.pdf 
*/  

template <typename Type=float>
class Backprop {
 using Vect = uvec<Type>;
 using Matr = umat<Type>;
 using Cube = ucub<Type>;
 
 public:
	// training parameters
	struct params {
		Type        threshold;  // stopping: determines when network will be considered as trained [default: 1e-6]
		size_t      max_iters;  // stopping: max number of iterations [default: 0 - until convergence]
		int         batch_size; // SGD mini batch size [default: 1]
		size_t      info_iters; // display progress every info_iters iterations [default: 1]
		bool        verbose;    // display info every batch [default: false]
		size_t      autosave;   // autosave iterations [default: 0 - no autosave]
		bool        multisave;  // save weights in incremental files [defalut: false]
		std::string filename;   // filename for saving the network's weights
		
		params() {
			threshold  = Type(1e-6);
			max_iters  = 0;
			batch_size = 1;
			info_iters = 0;
			verbose    = false;
			autosave   = 0;
			multisave  = false;
		}
	};
	
	// constructors, destructor
	Backprop(): _log(nullptr), _speed(0) {}
	~Backprop() {}
	
	void   set_params(const params& __opt) { _opt = __opt; }
	params get_params() const { return _opt; }
	void   set_logging_stream(Logger* __log) { _log = __log; }

	std::string info() const {
		std::stringstream ss;
		ss << "batch_size:" << _opt.batch_size << " epochs:" << _opt.max_iters;
		return ss.str();
	}
	
	// trains the feedforward network 'net' using input-target samples from 'X' and 'Y'
	// with stochastic gradient descent.
	template <class Loss, class GStep, class Shuffle>
	void train(FFNN<Type>& __net, Loss& __loss, GStep& __gs, Shuffle& __sh, const Matr& __X, const Matr& __Y) {
		_dev = __X.dev();
		assert(__Y.dev()==_dev);
		__net.to_device(_dev);
		__gs.to_device(_dev);
		__loss.to_device(_dev);
		Matr Xvalid(_dev), Yvalid(_dev);
		gradient_descent(__net, __loss, __gs, __sh, __X, __Y, Xvalid, Yvalid); 
	}

	// trains the feedforward network 'net' using input-target samples from 'X' and 'Y'
	// with stochastic gradient descent. Displays the accuracy in the validation set
	// at the end of each epoch.
	template <class Loss, class GStep, class Shuffle>
	void train(FFNN<Type>& __net, Loss& __loss, GStep& __gs, Shuffle& __sh, 
			   const Matr& __X, const Matr& __Y, const Matr& __Xvalid, const Matr& __Yvalid) {
		_dev = __X.dev();
		assert(__Y.dev()==_dev && __Xvalid.dev()==_dev && __Yvalid.dev()==_dev);
		__net.to_device(_dev);
		__gs.to_device(_dev);
		__loss.to_device(_dev);
		gradient_descent(__net, __loss, __gs, __sh, __X, __Y, __Xvalid, __Yvalid);
	}

	// returns the total loss after trainining completed
	Type total_loss() const { return _tot_loss; }

	// called when autosave is on (>0)
	virtual void save_weights(FFNN<Type>& __net);
	
	// displays progress info every 'params.info_iters' iterations in stdout.
	// override to change that, eg when used with a GUI. 
	virtual void show_progress_info();

 protected:
	template <class Loss, class GStep, class Shuffle>
	void gradient_descent(FFNN<Type>& __net, Loss& __loss, GStep& __gs, Shuffle& __sh,
						  const Matr& __X, const Matr& __Y,
						  const Matr& __Xvalid, const Matr& __Yvalid);

 protected:
	// private data
	device        _dev;
	params        _opt;
	size_t        _iter;
	Type          _tot_loss;
	Type          _acc_validation;
	std::string   _loss_name;
	Logger*       _log;
	double        _speed;
	std::chrono::steady_clock::time_point _time0, _time1, _time2;
};



/////////////////////////////////////////////////////////////////////////////////
//                        I M P L E M E N T A T I O N                          //
/////////////////////////////////////////////////////////////////////////////////


template <typename Type>
template <class Loss, class GStep, class Shuffle>
void Backprop<Type>::gradient_descent(FFNN<Type>& __net, Loss& __loss, GStep& __gs, Shuffle& __sh,
									  const Matr& __X, const Matr& __Y, 
									  const Matr& __Xvalid, const Matr& __Yvalid)
{
	int nlyrs = __net.nlayers();

	umml_assert(nlyrs > 1, "FFNN is not properly built.");
	umml_assert(__X.ydim()==__Y.ydim(), "Training data/targets dimensions mismatch.");

	_loss_name = __loss.name;

	Layer<Type>* inl = __net.input_layer();
	Layer<Type>* l;
	int lx;
	
	// per layer gradients and deltas
	std::vector<Matr> grads;
	std::vector<Vect> dws;
	std::vector<Vect> dbs;

	// per layer gradient step for weights and biases
	std::vector<GStep> gsw;
	std::vector<GStep> gsb;

	// optimize error gradient propagation
	// il->h1->h2->ol, (X)h1
	// il->mpl->cl->ol, (X)cl
	for (l=inl->_next; l != nullptr; l=l->_next) {
		if (!l->_prev->_gprop && (l->_prev->_w.empty() || !l->_prev->_trainable)) l->_gprop = false;
		else break;
	}

	// set trainable layers
	bool trainable = true;
	for (l=__net.output_layer(); l != nullptr; l=l->_prev) {
		if (!l->_trainable) trainable = false;
		l->set_trainable(trainable);
	}

	// gradients (+1 for loss) and gradient step methods
	for (l=inl; l!=nullptr; l=l->_next) {
		grads.push_back(Matr(_dev));
		dbs.push_back(Vect(_dev));
		dws.push_back(Vect(_dev));
		gsw.push_back(__gs);
		gsb.push_back(__gs);
	}
	grads.push_back(Matr(_dev));

	// seed shuffle object
	__sh.seed();

	// training epochs
	std::vector<int> trnidx(__X.ydim());
	umat<Type> Yb(_dev);
	_time0 = _time1 = std::chrono::steady_clock::now();
	for (_iter=1;; ++_iter) {
		_tot_loss = 0;

		// shuffle training set
		__sh.shuffle(trnidx, __X.ydim());

		// TODO: Remove when debugging UMML is done!!!
		//for (size_t i=0; i<trnidx.size(); ++i) trnidx[i] = i;

		// minibatch stochastic gradient descent
		int batch_size = _opt.batch_size < __X.ydim() ? _opt.batch_size : __X.ydim();
		int nbatches = __X.ydim() / batch_size;
		int nleftover = __X.ydim() % batch_size;
		for (int b=0; b<nbatches; ++b) {
			int bs = batch_size;
			if (b==nbatches-1) bs += nleftover;
			std::vector<int> idx(bs); 
			for (int k=0; k<bs; ++k) idx[k] = trnidx[b*batch_size+k];

			// forward pass
			__net.training_forward(__X, idx);

			// create the ground truth matrix Yb for the selected batch
			Yb.resize(bs, __Y.xdim());
			Yb.copy_rows(__Y, idx);

			// calculate loss and error gradient
			_tot_loss += __loss.calculate(grads[nlyrs], Yb, __net.output_layer()->_a);

			// backpropagate gradients. stops when the first non trainable layer is found
			for (l=__net.output_layer(), lx=nlyrs-1; l != nullptr && l->_trainable; l=l->_prev, --lx) {
				//std::cout << "gradient " << l->_name << " " << grads[lx+1].shape() << " ";
				l->backward(grads[lx+1], grads[lx], dbs[lx], dws[lx]);
				//std::cout << "--> " << grads[lx].shape() << "\n";
			}
			//std::abort();

			// adjust weights
			for (l=inl->_next, lx=1; l != nullptr; l=l->_next, ++lx) {
				if (l->_trainable) {
					Vect db(_dev), dw(_dev);
					l->accumulate(db, dw, dbs[lx], dws[lx]);
					// gradient step
					db.mul(Type(1)/bs);
					gsb[lx].step(db);
					dw.mul(Type(1)/bs);
					gsw[lx].step(dw);
					// update
					l->update(db, dw);
				}
			}

			if (_opt.verbose) {
				_time2 = std::chrono::steady_clock::now();
				if (duration_milli(_time0, _time2) > 1000) {
					unsigned long time_diff = duration_milli(_time1, _time2);
					int nsamples = (b+1)*batch_size;
					std::cout << "epoch: " << _iter << " (" << nsamples << "/" << __X.ydim() << ") ";
					if (time_diff > 1) {
						_speed = 1000. * ((double)nsamples / time_diff);
						std::stringstream ss;
						ss << "speed: " << std::fixed << std::setprecision(1) << _speed << " ETA: "
						   << format_duration((unsigned long)((__X.ydim()-nsamples)*time_diff/nsamples));
						std::cout.width(79); std::cout << std::left << ss.str();
					}
					std::cout << "\r" << std::flush;
					_time0 = _time2;
				}
			}
		}
		
		// calculate accuracy in validation set
		_acc_validation = Type(-1);
		if (__Xvalid.ydim() > 0 && __Xvalid.ydim()==__Yvalid.ydim()) {
			Matr Ypred(_dev), Ypred_1hot(_dev);
			__net.predict(__Xvalid, Ypred);
			Ypred_1hot.resize_like(Ypred);
			Ypred_1hot.argmaxto1hot(Ypred);
			_acc_validation = umml::accuracy(__Yvalid, Ypred_1hot);
		}
		
		// show info
		_time2 = std::chrono::steady_clock::now();
		if (_opt.info_iters > 0 && _iter % _opt.info_iters==0) show_progress_info();
		_time1 = _time2;
		// autosave
		if (_opt.autosave > 0 && _iter % _opt.autosave==0) save_weights(__net);
		// check stopping conditions
		if (_opt.max_iters > 0 && _iter >= _opt.max_iters) break;
		if (_tot_loss < _opt.threshold) break;
	}

	_time2 = std::chrono::steady_clock::now();
	if (_opt.info_iters > 0 && _iter % _opt.info_iters != 0) show_progress_info();
	if (_opt.autosave > 0 && _iter % _opt.autosave != 0) save_weights(__net);
}


template <typename Type>
void Backprop<Type>::save_weights(FFNN<Type>& __net) 
{
	if (_opt.multisave) {
		std::stringstream ss;
		ss << _opt.filename << "." << std::setfill('0') << std::setw(3) << _iter;
		__net.save(ss.str());
	}
	__net.save(_opt.filename);
}

template <typename Type>
void Backprop<Type>::show_progress_info() 
{
	if (_opt.verbose) {
		std::stringstream ss;
		ss << std::fixed;
		ss << "epoch: " << std::setw(3) << _iter << ", time: " << format_duration(_time1, _time2);
		ss << " (speed " << std::setprecision(1) << _speed << ")";
		ss << std::setprecision(4);
   		ss << ", loss (" << _loss_name << "): " << _tot_loss;
   		if (_acc_validation != Type(-1)) ss << ", validation accuracy: " << _acc_validation;
   		ss << "\n";
		if (_log) *_log << ss.str(); else std::cout << ss.str();
	}
}


};     // namespace umml

#endif // UMML_BACKPROP_INCLUDED
