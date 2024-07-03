#ifndef UMML_GDSTEP_INCLUDED
#define UMML_GDSTEP_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Gradient Descent step methods.

 FILE:     gdstep.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: supervised, classification, regression, neural networks
 
 Namespace
 ~~~~~~~~~
 umml::gdstep
 
 Notes
 ~~~~~
 * learnrate
 * lrdecay
 * momentum
 * adam
  
 Dependencies
 ~~~~~~~~~~~~
 umml vector
 
 Internal dependencies:
  
 Usage example
 ~~~~~~~~~~~~~
  
 TODO
 ~~~~ 
*/


#include "../uvec.hpp"
#include "op/momentum.hpp"
#include "op/adam.hpp"


namespace umml {
namespace gdstep {


// constant learning rate gradient descent step method
// r: learning rate
// step = r * gradient
template <typename Type=float>
struct learnrate {
	Type r;

	learnrate(Type __r=0.01): r(__r) {}

	void step(uvec<Type>& g) { 
		g.mul(r); 
	}

	void to_device(device __dev) {}

	std::string info() const { 
		std::stringstream ss;
		ss << "learnrate (α:" << r << ")";
		return ss.str(); 
	}
};


// decaying learning rate gradient descent step method
// r: learning rate
// c; learning rate decay
// steps: steps that need to be completed to perform learning rate decay
// t: current timestep
// step = r * gradient
template <typename Type=float>
struct lrdecay {
	Type r; 
	Type c;
	int  steps;
	int  t;
	
	lrdecay(Type __r=0.1, Type __c=0.999, int __steps=1000): 
		r(__r), c(__c), steps(__steps), t(0) {}

	void step(uvec<Type>& g) { 
		++t;
		if (t==steps) { 
			t = 0;
			r = r*c + 1e-6;
		}
		g.mul(r);
	}

	void to_device(device __dev) {}

	std::string info() const { 
		std::stringstream ss;
		ss << "decaying learnrate (α:" << r << " c:" << c << "/" << steps << ")";
		return ss.str(); 
	}	
};


// momentum gradient descent step
// r: learning rate
// b: momentum
// m: previous gradient
//
// m = b*m + g; 
// g = r*m;
template <typename Type=float>
struct momentum {
	Type r;
	Type b;
	uvec<Type> m;
	
	momentum(Type __r=0.01, Type __b=0.9): r(__r), b(__b) {}
	
	void step(uvec<Type>& g) {
		if (m.empty()) { m.resize(g.len()); m.zero_active_device(); }
		apply_momentum(g, r, b, m);
	}
	
	void to_device(device __dev) { m.to_device(__dev); }

	std::string info() const { 
		std::stringstream ss;
		ss << "momentum (α:" << r << " β:" << b << ")";
		return ss.str(); 
	}	
};


// Adam
// r: stepsize (learning rate)
// b1, b2: exponential decay rates for the moment estimates
// m, v: 1st and 2nd moment vector
// e: regularizer to control divisions by zero
// t: timestep
//
// m = b1*m + (1-b1)*g
// v = b2*v + (1-b2)*g^2
// mh = m / (1-b1^t)
// vh = v / (1-b2^t)
// g = (r*mh) / (vh.sqrt()+e)
template <typename Type=float>
struct adam {
	Type r;
	Type b1, b2;
	Type e; 
	uvec<Type> m, v;
	int t;
	
	adam(Type __r=0.001, Type __b1=0.9, Type __b2=0.999, Type __e=1e-8):
		r(__r), b1(__b1), b2(__b2), e(__e), t(0) {}

	void step(uvec<Type>& g) {
		if (m.empty()) {
			m.resize(g.len()); m.zero_active_device();
			v.resize(g.len()); v.zero_active_device();
		}
		++t;
		Type b1t = 1 - powi(b1,t);
		Type b2t = 1 - powi(b2,t);
		apply_adam(g, r, b1, b2, b1t, b2t, e, m, v);
	}
	
	void to_device(device __dev) { m.to_device(__dev); v.to_device(__dev); }

	std::string info() const { 
		std::stringstream ss;
		ss << "Adam (α:" << r << " β1:" << b1 << " β2:" << b2 << ")";
		return ss.str(); 
	}	

};


};     // namespace gdstep 
};     // namespace umml

#endif // UMML_GDSTEP_INCLUDED
