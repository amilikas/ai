#ifndef UMML_PSO_INCLUDED
#define UMML_PSO_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Particle Swarm Optimization.

 FILE:     pso.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2024
 KEYWORDS: pso, swarm
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 - fitness=0 is an optimal solution.


 Dependencies
 ~~~~~~~~~~~~
 umml::initializer
 STL string
 STL vector
 
 Internal dependencies:
 STL file streams
 umml algo
 umml rand
 umml openmp
  
 Usage example
 ~~~~~~~~~~~~~
 
 1. Define a fitness function (Type=float in this example)
	struct Fitness {
	  double operator ()(const std::vector<float>& chain) {
		decode chain and calculcate fitness
		return fitness;
	  }
	};

 2. Construct a PSO object, set parameters and call solve():

	Binary encoding (bool) example:
	PSO<float> pso;
	initializer::values::random<float> init(minval, maxval);
	pso.solve(nparticles, ndimensions, minval, maxval, init);

	or, instead of calling solve():

	pso.init(nparticles, ndimensions, minval, maxval, init);
	for (;;) {
	  pso.evaluate(ff);
	  pso.update();
	  if (pso.done()) break;
	  pso.step();
	}
	pso.finish();

 * For OpenMP parallelism, compile with -D__USE_OPENMP__ -fopenmp (GCC)
   umml_set_openmp_threads(-1); // -1 for std::thread::hardware_concurrency

 TODO
 ~~~~
  
*/


#include "../logger.hpp"
#include "../utils.hpp"
#include "../rand.hpp"
#include "initializer.hpp"
#include <chrono>
#include <fstream>


namespace umml {


// minimum=0 is an optimal solution.
template <typename Type=float>
class PSO {
 using time_point = std::chrono::steady_clock::time_point;

 public:
	// error status (see error codes)
	int err_id;
	
	// error codes
	enum {
		OK = 0,
		NotExists,
		BadMagic,
		BadCount,
		BadAlloc,
		BadIO,
	};

	// parallel modes
	enum {
		SingleThread,
		OpenMP,
	};

	// particle
	// use umml::uvec 
	struct particle {
		bool evaluated;
		double curr_minimum;
		double best_minimum;
		std::vector<Type> x;
		std::vector<Type> v;
		std::vector<Type> pbest;
	};

	// PSO parameters
	struct params {
		Type        c1;         // cognitive parameter [default: 1.5]
		Type        c2;         // social parameter [default: 1.5]
		Type        w;          // inertia weight [default: 0.999]
		double      threshold;  // stopping: fitness better than threshold [default: 0.999]
		size_t      max_iters;  // stopping: max number of iterations [default: 0 - until convergence]
		size_t      info_iters; // display info every info_iters iterations [default 100]
		std::string filename;   // filename for saving the population's parameters
		size_t      autosave;   // autosave iterations [default: 0 - no autosave]
		int         parallel;   // parallel mode [default: OpenMP]
		params() {
			c1         = 1.5;
			c2         = 1.5;
			w          = 0.5;
			threshold  = 1e-6;
			max_iters  = 0;
			info_iters = 100;
			autosave   = 0;
			parallel   = OpenMP;
		}
	};

	PSO(): err_id(OK), _np(0), _ndims(0), _iter(0), _log(nullptr) {}
	virtual ~PSO() {}

	// checks if no errors occured during allocation/load/save
	explicit operator bool() const { return err_id==OK; }

	// parameters and logging
	void      set_params(const params& opt) { _opt = opt; }
	params    get_params() const { return _opt; }
	void      set_logging_stream(Logger* log) { _log = log; }
	size_t    iterations() const { return _iter; }

	// get a specific particle
	// in first place (idx=0) is the best particle
	particle  get_particle(int num=0) const { 
		assert(num >= 0 && num < _np);
		return _particles[_sorted[num]]; 
	}

	particle& particle_at(int num=0) { 
		assert(num >= 0 && num < _np);
		return _particles[_sorted[num]]; 
	}

	// set a particle's minimum
	void   set_current_minimum(int idx, double f) { _particles[idx].curr_minimum = f; }

	// returns the fitness of the global best particle
	double gbest_minimum() const { return _gbest_minimum; }

	// returns the solution
	std::vector<Type> solution() const { return _gbest; }

	// reset iteration counter
	void reset() { _iter = 0; }

	// initialize particles
	template <class Initializer>
	void   init(int nparticles, int ndims, Type minval, Type maxval, Initializer& initializer);

	template <class Initializer>
	void   init(int nparticles, int ndims, Initializer& initializer);

	// init only min/max (used after a swarm is loaded from disk)
	void   init_minmax(Type minval, Type maxval);

	// evaluate a single member
	template <class Fitness>
	void   evaluate_particle(int idx, Fitness& ff);

	// evaluate all particles
	template <class Fitness>
	void   evaluate(Fitness& ff);

	// sorts particles and save current state
	void   update();

	// move particles to new positions
	void   step();

	// save current state
	void   save_state();

	// save final state
	void   finish();

	// for convenience, runs the evolution cycle
	template <class Initializer, class Fitness>
	void   solve(int nparticles, int ndims, Type minval, Type maxval, 
				 Initializer& initializer, Fitness& ff, bool step_first=false);

	template <class Initializer, class Fitness>
	void   solve(int nparticles, int ndims, 
				 Initializer& initializer, Fitness& ff, bool step_first=false);

	void   show_progress_info();

	bool   save_values(std::ofstream& os, const std::vector<Type>& values);
	bool   load_values(std::ifstream& is, std::vector<Type>& values);
	bool   save(const std::string& filename);
	bool   load(const std::string& filename);

	// returns a std::string with the description of the error in err_id
	std::string error_description() const;

	// returns info about ga's parameters
	std::string info() const;
	
	// returns the total time elapsed since ga started
	std::string total_time() const;

	// early stopping
	bool   early_stop() const { return get_particle(0).best_minimum <= _opt.threshold; }

	// determines if PSO is done
	bool   done() const { return early_stop() || (_opt.max_iters > 0 && _iter > _opt.max_iters); }


 private:
 	int      _np;
 	int      _ndims;
	size_t   _iter;
 	params   _opt;
	Logger*  _log;
 	std::vector<particle> _particles;
	std::vector<int> _sorted;
	std::vector<Type> _xmin;
	std::vector<Type> _xmax;
	std::vector<Type> _gbest;
	double _gbest_minimum;
	time_point _time0, _time1;
};


////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////

template <typename Type>
template <class Initializer>
void PSO<Type>::init(int nparticles, int ndims, Type minval, Type maxval, Initializer& initializer) 
{
	_np = nparticles;
	_ndims = ndims;
	_xmin.resize(_ndims);
	_xmax.resize(_ndims);
	std::fill(_xmin.begin(), _xmin.end(), minval);
	std::fill(_xmax.begin(), _xmax.end(), maxval);
	_particles.resize(_np);
	for (int i=0; i<_np; ++i) {
		particle& p = _particles[i];
		p.x.resize(_ndims);
		p.v.resize(_ndims);
		p.pbest = p.x;
		initializer.apply(p.x);
		std::fill(p.v.begin(), p.v.end(), Type(0.0));
		p.curr_minimum = p.best_minimum = std::numeric_limits<Type>::max();
		p.evaluated = false;
		// update _xmin and _xmax boundary constraints
		/*
		for (int j=0; j<_ndim; ++j) {
			if (p.x[j] < _xmin[j]) _xmin[j] = p.x[j];
			if (p.x[j] > _xmax[j]) _xmax[j] = p.x[j];
		}
		*/
	}
	_gbest_minimum = std::numeric_limits<Type>::max(); 

	// start timer
	_time0 = _time1 = std::chrono::steady_clock::now();

	// advance iteration counter
	++_iter;
}

template <typename Type>
template <class Initializer>
void PSO<Type>::init(int nparticles, int ndims, Initializer& initializer) 
{
	Type minval = std::numeric_limits<Type>::min();
	Type maxval = std::numeric_limits<Type>::max();
	init(nparticles, ndims, minval, maxval, initializer);
}

template <typename Type>
void PSO<Type>::init_minmax(Type minval, Type maxval) 
{
	_xmin.resize(_ndims);
	_xmax.resize(_ndims);
	std::fill(_xmin.begin(), _xmin.end(), minval);
	std::fill(_xmax.begin(), _xmax.end(), maxval);

	// start timer
	_time0 = _time1 = std::chrono::steady_clock::now();

	// advance iteration counter
	++_iter;
}

// evaluate a single particle
template <typename Type>
template <class Fitness>
void PSO<Type>::evaluate_particle(int idx, Fitness& ff) 
{
	if (!_particles[idx].evaluated) {
		_particles[idx].curr_minimum = ff(_particles[idx].x);
		_particles[idx].evaluated = true;
	}
}

template <typename Type>
template <class Fitness>
void PSO<Type>::evaluate(Fitness& ff) {
	if (_opt.parallel==SingleThread) {
		for (int i=0; i<_np; ++i) evaluate_particle(i, ff);
	} else if (_opt.parallel==OpenMP) {
		#pragma omp parallel for num_threads(openmp<>::threads)
		for (int i=0; i<_np; ++i) evaluate_particle(i, ff);
	}
}

template <typename Type>
void PSO<Type>::update() 
{
	// update best_fitness and globalbest fitness
	int gbest_idx = -1;
	for (size_t i=0; i<_particles.size(); ++i) {
		particle& p = _particles[i];
		if (p.curr_minimum < p.best_minimum) {
			p.best_minimum = p.curr_minimum;
			p.pbest = p.x;
		}
		if (p.best_minimum < _gbest_minimum) {
			_gbest_minimum = p.best_minimum;
			gbest_idx = i;
		}
	}
	if (gbest_idx >= 0) _gbest = _particles[gbest_idx].x;

	// sort particles by fitness
	_sorted.resize(_particles.size());
	for (size_t i=0; i<_particles.size(); ++i) _sorted[i] = i;
	struct orderby_fitness {
		const std::vector<particle>& _pop;
		orderby_fitness(const std::vector<particle>& pop): _pop(pop) {}
		bool operator() (int a, int b) const { return (_pop[a].best_minimum < _pop[b].best_minimum); }
	};
	orderby_fitness order_by(_particles);
	std::sort(_sorted.begin(), _sorted.end(), order_by);

	// show progress info and autosave
	save_state();

	// advance iteration counter
	++_iter;
}

template <typename Type>
void PSO<Type>::step() 
{
	// update velocity p.v and position p.x
	for (particle& p : _particles) {
		p.evaluated = false;
		for (int j=0; j<_ndims; ++j) {
	    	Type r[2];
		    uniform_random_reals(r, 2, Type(0), Type(1));
		    p.v[j] = _opt.w*p.v[j] + _opt.c1*r[0]*(p.pbest[j]-p.x[j]) + _opt.c2*r[1]*(_gbest[j]-p.x[j]);
		    p.x[j] += p.v[j];
		    // apply boundary constraints and possibly reverse direction of velocity
		    if (p.x[j] < _xmin[j]) {
    			p.x[j] = _xmin[j];
			    p.v[j] *= -0.5;
		    } else if (p.x[j] > _xmax[j]) {
    			p.x[j] = _xmax[j];
			    p.v[j] *= -0.5;
		    }
		}
	}
}

template <typename Type>
void PSO<Type>::save_state()
{
	if (_opt.info_iters > 0 && _iter % _opt.info_iters==0) show_progress_info();
	if (_opt.autosave > 0 && _iter % _opt.autosave==0) save(_opt.filename);
}

template <typename Type>
void PSO<Type>::finish()
{
	if (_opt.info_iters > 0 && _iter % _opt.info_iters != 0) show_progress_info();
	if (_opt.autosave > 0 && _iter % _opt.autosave != 0) save(_opt.filename);
}

template <typename Type>
template <class Initializer, class Fitness>
void PSO<Type>::solve(int nparticles, int ndims, Type minval, Type maxval, 
					  Initializer& initializer, Fitness& ff, bool step_first)
{
	_iter = 0;
	init(nparticles, ndims, minval, maxval, initializer);
	if (step_first) step();
	for (;;) {
		evaluate(ff);
		update();
		if (done()) break;
		step();
	}
	finish();
}

template <typename Type>
template <class Initializer, class Fitness>
void PSO<Type>::solve(int nparticles, int ndims, Initializer& initializer, Fitness& ff, bool step_first)
{
	Type minval = std::numeric_limits<Type>::min();
	Type maxval = std::numeric_limits<Type>::max();
	solve(nparticles, ndims, minval, maxval, initializer, ff, step_first);
}

template <typename Type>
bool PSO<Type>::save_values(std::ofstream& os, const std::vector<Type>& values)
{
	for (Type v : values) {
		double d = (double)v;
		os.write((char*)&d, sizeof(double));
	}
	return true;
}

template <typename Type>
bool PSO<Type>::load_values(std::ifstream& is, std::vector<Type>& values)
{
	values.resize(_ndims);
	for (int j=0; j<_ndims; ++j) {
		double d;
		is.read((char*)&d, sizeof(double));
		values[j] = Type(d);
	}
	return true;
}

template <typename Type>
bool PSO<Type>::save(const std::string& filename)
{
	int n;
	std::ofstream os;
	os.open(filename, std::ios::out|std::ios::binary);
	if (!os.is_open()) { err_id=NotExists; return false; }
	os << "PSO." << "00"/*version*/ << ".";
	// iteration count
	n = (int)_iter;
	os.write((char*)&n, sizeof(int));
	// dimensions
	n = _ndims;
	os.write((char*)&n, sizeof(int));
	// swarm size
	n = _np;
	os.write((char*)&n, sizeof(int));
	// save members
	for (int i=0; i<n; ++i) {
		particle& p = _particles[_sorted[i]];
		os.write((char*)&p.curr_minimum, sizeof(double));
		os.write((char*)&p.best_minimum, sizeof(double));
		save_values(os, p.x);
		save_values(os, p.v);
		save_values(os, p.pbest);
	}
	// save gbest
	os.write((char*)&_gbest_minimum, sizeof(double));
	save_values(os, _gbest);
	// save xmin and xmax
	save_values(os, _xmin);
	save_values(os, _xmax);

	os.close();
	if (!os) { err_id=BadIO; return false; }
	return true;
}

template <typename Type>
bool PSO<Type>::load(const std::string& filename)
{
	char buff[8];
	int n;

	std::ifstream is;
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { err_id=NotExists; return false; }
	is.read(buff, 4);
	buff[4] = '\0';
	if (std::string(buff) != "PSO.") { err_id=BadMagic; return false; }
	is.read(buff, 3);
	if (buff[2] != '.') { err_id=BadMagic; return false; }
	//version = ('0'-buff[0])*10 + '0'-buff[1];
	// iteration count
	is.read((char*)&n, sizeof(int));
	_iter = (size_t)n;
	// chain size
	is.read((char*)&n, sizeof(int));
	_ndims = n;
	// population size
	is.read((char*)&n, sizeof(int));
	_np = n;
	// allocate memory for the stored population
	_particles.clear(); _particles.reserve(_np);
	_sorted.clear(); _sorted.reserve(_np);
	for (int i=0; i<_np; ++i) _sorted.push_back(i);
	// load members 
	for (int i=0; i<_np; ++i) {
		particle p;
		is.read((char*)&p.curr_minimum, sizeof(double));
		is.read((char*)&p.best_minimum, sizeof(double));
		load_values(is, p.x);
		load_values(is, p.v);
		load_values(is, p.pbest);
		p.evaluated = true;
		_particles.push_back(p);
	}
	// load gbest
	is.read((char*)&_gbest_minimum, sizeof(double));
	load_values(is, _gbest);
	// load xmin and xmax
	load_values(is, _xmin);
	load_values(is, _xmax);

	is.close();	
	if (!is) { err_id=BadIO; return false; }
	return true;
}

template <typename Type>
std::string PSO<Type>::error_description() const 
{
	switch (err_id) {
		case NotExists: return "File not exists.";
		case BadMagic:  return "File has unknown magic number.";
		case BadCount:  return "Unexpected number.";
		case BadAlloc:  return "Memory allocation error.";
		case BadIO:     return "File IO error.";
	};
	return "";
}

template <typename Type>
std::string PSO<Type>::info() const 
{
	std::stringstream ss;
	ss << "Swarm size: " << _np << " particless\nDimensions: " << _ndims << "\n"
	   << "Iterations: " << (_iter > 0 ? _iter : 0) << " of " << _opt.max_iters << "\n"
	   << "Early stop threshold: " << _opt.threshold << "\n" << "Elitism ratio: " << _opt.elitism << "\n";
	return ss.str();
}

template <typename Type>
std::string PSO<Type>::total_time() const 
{
	time_point time_now = std::chrono::steady_clock::now();
	return std::string("Total time: ") + format_duration(_time0, time_now);
}

template <typename Type>
void PSO<Type>::show_progress_info() 
{
	time_point time_now = std::chrono::steady_clock::now();
	double max_fitn = _particles[_sorted[0]].best_minimum;
	double min_fitn = _particles[_sorted[_np-1]].best_minimum;
	double sum = 0.0;
	for (int i=0; i<_np; ++i) sum += _particles[i].best_minimum;
	double mean_fitn = sum / _np;
	std::stringstream ss;
	ss << std::fixed;
	ss << std::setprecision(6);
	// TODO: speed, ETA (like ETA in backprop)
	ss << "Iteration " << std::setw(3) << _iter << ", time: " << format_duration(_time1, time_now) << ", " 
	   << "best minima: " << max_fitn << ", worst: " << min_fitn << ", mean: " << mean_fitn << "\n";
	if (_log) *_log << ss.str(); else std::cout << ss.str();
	_time1 = std::chrono::steady_clock::now();
}


};     // namespace umml

#endif // UMML_PSO_INCLUDED