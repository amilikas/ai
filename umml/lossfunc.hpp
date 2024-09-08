#ifndef UMML_LOSSFUNC_INCLUDED
#define UMML_LOSSFUNC_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Loss functions.

 FILE:     lossfunc.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 - RSS
 - MSE
 - Softmax Cross Entropy
 
 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 STL string

 Internal dependencies:
  
 Usage example
 ~~~~~~~~~~~~~

 TODO
 ~~~~
 Huber loss
 https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3

*/

#include <string>
#include <cmath>
#include <cstdarg>


namespace umml {
namespace lossfunc {


/*
 Residual Sum of Squares loss
 loss = 1/2 Σ(y-t)^2
 derrivative(loss) = y-t
*/
template <typename Type=float>
struct rss {
	using Matr = umat<Type>;
	std::string name;
	
	rss(): name("RSS") {}

	void to_device(device __dev) {}

	std::string info() const { return name; }
	
	// calculate total loss and error gradient 'g' from ground truth targets 't'
	// and network's output 'y'.
	Type calculate(Matr& g, const Matr& t, const Matr& y, ...) {
		assert(t.len() == y.len());
		g.resize(y.ydim(), y.xdim());
		g.set(y);
		g.plus(t, Type(-1));
		return g.sum_squared();
		/*
		Type tot_loss = 0;
		for (int sample=0; sample<nsamples; ++sample) {
			for (int j=0; j<__y.xdim(); ++j) {
				Type d = y(sample,j) - t(sample,j);
				tot_loss += d*d;
				g(sample,j) = d;
			}
		}
		return tot_loss;
		*/
	}
};


/*
 Mean Squarred Error loss
 loss = 1/n Σ(y-t)^2
 derrivative(loss) = 2*(y-t)/n
 n: number of elements in the output
*/
template <typename Type=float>
struct mse {
	using Matr = umat<Type>;
	std::string name;
	
	mse(): name("MSE") {}

	void to_device(device __dev) {}

	std::string info() const { return name; }
	
	// calculate total loss and error gradient 'g' from ground truth targets 't'
	// and network's output 'y'.
	Type calculate(Matr& g, const Matr& t, const Matr& y, ...) {
		assert(t.len() == y.len());
		Type loss;
		g.resize(y.ydim(), y.xdim());
		g.set(y);
		g.plus(t, Type(-1));
		loss = g.sum_squared();
		g.mul(Type(2)/y.xdim());
		return loss/y.xdim();

		/*
		Type tot_loss = 0;
		int nsamples = y.nrows();
		assert(t.nelem()==y.nelem());
		g.resize(nsamples, y.ncols());
		for (int sample=0; sample<nsamples; ++sample) {
			for (int i=0; i<y.ncols(); ++i) {
				Type d = y(sample,i) - t(sample,i);
				g(sample,i) = d*2/y.ncols();
				tot_loss += d*d;
			}
		}
		return tot_loss / y.ncols();
		*/
	}
};


/*
 Softmax Cross Entropy loss (classification)
 loss = -log(y-t)
 derrivative(loss) = y - t
 n: number of elements in the output
*/
template <typename Type=float>
struct softmaxce {
	using Vect = uvec<Type>;
	using Matr = umat<Type>;
	std::string name;
	
	softmaxce(): name("CCE") {}

	void to_device(device __dev) {}

	std::string info() const { return name; }
	
	// calculate total loss and error gradient 'g' from ground truth targets 't'
	// and network's output 'y'. 
	Type calculate(Matr& g, const Matr& t, const Matr& y, ...) {
		assert(t.len() == y.len());
		g.resize(y.ydim(), y.xdim());
		g.set(y);
		g.plus(t, Type(-1));
		Type loss = 0;
		for (int sample=0; sample<y.ydim(); ++sample) {
			int pos1 = uv_ref<Type>(t.row_offset(sample), t.dev(), t.xdim(), t.xsize()).argmax();
			loss += -std::log(y.get_element(sample, pos1));
		}
		return loss;
		
		/*
		Type tot_loss = 0;
		assert(t.nelem()==y.nelem());
		for (int sample=0; sample<y.nrows(); ++sample) {
			Vect t_ref; t_ref.const_reference(t.row_cptr(sample), y.ncols());
			tot_loss += -std::log(y(sample, t_ref.find_first(1)));
		}
		g = y;
		g -= t;
		return tot_loss;
		*/
	}
};


/*
 Logloss (Binary Cross Entropy loss) (classification)
 loss = -log(y-t)
 loss = -1/N Σ[yi*log(p(yi)) + (1-yi)*log(1-p(yi))]
 derrivative(loss) = y - t
 n: number of elements in the output
*/
/*
template <typename Type=float>
struct logloss {
	using Vect = vect<Type>;
	using Matr = matr<Type>;
	std::string name;
	Type eps;
	
	logloss(): name("Logloss"), eps(1e-8) {}

	void to_device(device __dev) {}

	std::string info() const { return name; }
	
	// calculate total loss and error gradient 'g' from ground truth targets 't'
	// and network's output 'y'.
	Type calculate(Matr& g, const Matr& t, const Matr& y, ...) {
		Type tot_loss = 0;
		int nsamples = y.nrows();
		assert(t.nelem()==y.nelem());
		for (int i=0; i<nsamples; ++i) {
			for (int j=0; j<y.ncols(); ++j)
				tot_loss += t(i,j)*std::log(y(i,j)+eps) + (1.-t(i,j)*std::log(1.-y(i,j)+eps));
		}
		g = (1.-t)/(1.-y+eps) - t/(y+eps);
		return tot_loss / nsamples;
	}
};
*/


template <typename Type=float>
struct kdloss {
	using Vect = uvec<Type>;
	using Matr = umat<Type>;
	std::string name;

	Type  _alpha;  // α hyperparameter 
	Type  _tau;    // temperature T of softmax
	Matr  _t;      // ground truth (one-hot encoded)
	Matr  _y;      // softmax of student logits
	Matr  _ys;     // softmax with temperature of student logits
	Matr  _yt;     // softmax with temperature of teacher logits
	Matr  _ytys;   // yt/ys
	Vect  _max;
	Vect  _sum;
	Vect  _mu;

	
	kdloss(): name("KDloss"), _alpha(0.2), _tau(5.0) {}
	kdloss(Type __alpha, Type __tau): name("KDloss"), _alpha(__alpha), _tau(__tau) {}

	void to_device(device __dev) {
		_t.to_device(__dev);
		_y.to_device(__dev);
		_ys.to_device(__dev);
		_yt.to_device(__dev);
		_ytys.to_device(__dev);
		_max.to_device(__dev);
		_sum.to_device(__dev);
		_mu.to_device(__dev);
	}

	std::string info() const { 
		std::stringstream ss;
		ss << name << " (" << "α:" << _alpha << " T:" << _tau << ")";
		return ss.str(); 
	}

	// calculate total loss and error gradient 'g' 
	// 't' is ground truth (one-hot encoded) concatanation with teacher logits.
	// 'y' is student's model logits (pre softmax activations)
	// L = α∗H(t,softmax(y;T=1)) + (1-α)∗H(softmax(yt;T=τ),softmax(y,T=τ))
	// dL = α∗(softmax(y;T=1)-t) + (1-α)∗(softmax(y;T=τ)-softmax(yt,T=τ))
	Type calculate(Matr& g, const Matr& t, const Matr& y, ...) {
		assert(t.len() == 2*y.len());
		int nsamples = y.ydim();

		/*
		va_list list;
		va_start(list, y);
		Matr& yt = va_arg(list, Matr);
		*/

		_max.resize(nsamples);
		_sum.resize(nsamples);
		_mu.resize(y.xdim());

		// ground truth 
		_t.resize_like(y);
		_t.copy_cols(t, 0, y.xdim());

		// teacher's centered soft predictions = softmax(yt;T)
		_yt.resize_like(y);
		_yt.copy_cols(t, y.xdim(), y.xdim());
		/*
		_yt.reduce_sum(_mu, AxisX);
		_mu.mul(1.0/nsamples);
		_yt.plus_vector(_mu, AxisX, -1);
		*/
		_yt.reduce_max(_max, AxisY);
		_yt.apply_function(fExp, 1/_tau, Type(-1), _max, AxisY);
		_yt.reduce_sum(_sum, AxisY);
		_yt.divide(_sum, AxisY);

		// student's centered soft predictions = softmax(y;T)
		_ys.resize_like(y);
		_ys.set(y);
		/*
		_ys.reduce_sum(_mu, AxisX);
		_mu.mul(1.0/nsamples);
		_ys.plus_vector(_mu, AxisX, -1);
		*/
		_ys.reduce_max(_max, AxisY);
		_ys.apply_function(fExp, 1/_tau, Type(-1), _max, AxisY);
		_ys.reduce_sum(_sum, AxisY);
		_ys.divide(_sum, AxisY);

		// student's hard predictions = softmax(y)
		_y.resize_like(y);
		_y.set(y);
		_y.apply_function(fExp, Type(1), Type(-1), _max, AxisY);
		_y.reduce_sum(_sum, AxisY);
		_y.divide(_sum, AxisY);

		Type loss1 = 0;
		Type loss2 = 0;

		// loss1: categorical cross entropy of hard predictions (T=1)
		for (int sample=0; sample<nsamples; ++sample) {
			int pos1 = uv_ref<Type>(_t.row_offset(sample), _t.dev(), _t.xdim(), _t.xsize()).argmax();
			loss1 += -std::log(_y.get_element(sample, pos1));
		}

		// loss2: categorical cross entropy of soft predictions (T=τ) -Σyt*log(ys)
		_ytys.resize_like(_ys);
		_ytys.set(_ys);
		_ytys.apply_function(fLog);
		_ytys.prod(_yt, _ytys);
		loss2 += -_ytys.sum();
		
		/*
		// Kullback–Leibler divergence loss = yt*log(yt/ys)
		_ytys.resize_like(_ys);
		_ytys.reciprocal(_ys);
		_ytys.prod(_yt, _ytys);
		_ytys.apply_function(fLog);
		_ytys.prod(_yt, _ytys);
		loss2 = _ytys.sum();// / _yt.len();
		*/

		// g = α(y - t) + (1-α)/CT^2(ys-yt)
		g.resize(nsamples, _y.xdim());
		g.set(_y);
		g.plus(_t, Type(-1));
		g.mul(_alpha);
		_ys.plus(_yt, Type(-1));
		_ys.mul((1-_alpha)/_tau);
		g.plus(_ys);

		return _alpha*loss1 + (1-_alpha)*loss2;
	}
};


}; // namespace lossfunc
}; // namespace umml

#endif // UMML_LOSSFUNC_INCLUDED
