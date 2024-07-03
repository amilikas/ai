#ifndef UMML_GA_INCLUDED
#define UMML_GA_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Genetic Algorithms.

 FILE:     ga.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2022-2024
 KEYWORDS: genetic algorithms
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 - fitness must be in [0..1], fitness=1 is an optimal solution.
 - a common way to scale fitness to [0..1] is f = 1.0 / (1.0 + std::abs(value))


 Dependencies
 ~~~~~~~~~~~~
 umml::initializer
 umml::crossover
 umml::mutation
 STL string
 STL vector
 
 Internal dependencies:
 STL file streams
 umml algo
 umml rand
 umml openmp
  
 Usage example
 ~~~~~~~~~~~~~
 
 1. Define a fitness function (greater the fitness, better the solution):
	struct Fitness {
	  double operator ()(const std::vector<bool>& chain) {
		decode chain and calculcate fitness
		return fitness;
	  }
	};

 2. Construct a GA object, set parameters and call solve():

	Binary encoding (bool) example:
	GA<bool> ga;
	initializer::values::random<> init;
	crossover::values::onepoint<> xo;
	mutation::values::flip<> mut(0.01, 0.001);  // mutation probability, percentage of bits to flip
	ga.solve(tot_bits, pop_size, init, ff, xo, mut);

	or, instead of calling solve():

	ga.init(tot_bits, pop_size, init);
	for (;;) {
	  ga.evaluate(ff);
	  ga.sort();
	  if (ga.done()) break;
	  ga.evolve(xo, mut);
	}
	ga.finish();

	Permutations encoding example:
	GA<int> ga;
	initializer::permut::random<int> init;
	crossover::permut::onepoint<int> xo;
	mutation::permut::swap<int> mut(0.01, 0.001);  // mutation prob, percentage of indeces to swap
	...

 * For OpenMP parallelism, compile with -D__USE_OPENMP__ -fopenmp (GCC)
   umml_set_openmp_threads(-1); // -1 for std::thread::hardware_concurrency

 TODO
 ~~~~
  
*/


#include "../logger.hpp"
#include "../utils.hpp"
#include "../rand.hpp"
#include "initializer.hpp"
#include "crossover.hpp"
#include "mutation.hpp"
#include <chrono>
#include <fstream>


namespace umml {


template <typename Type=bool>
class GA
{
 using Chain = std::vector<Type>;
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
	
	// member type
	struct member {
		double fitness;
		Chain  chain;
		bool   evaluated;
	};

	// GA parameters
	struct params {
		double      threshold;    // stopping: fitness better than threshold [default: 0.999]
		size_t      max_iters;    // stopping: max number of iterations [default: 0 - until convergence]
		double      elitism;      // elitism: percent of best members that pass to the next generation [default: 0.0]
		size_t      children;     // maximum number of childrenn from two parents [default: 2]
		size_t      info_iters;   // display info every info_iters iterations [default 100]
		std::string filename;     // filename for saving the population's parameters
		size_t      autosave;     // autosave iterations [default: 0 - no autosave]
		int         parallel;     // parallel mode [default: OpenMP]
		params() {
			threshold  = 0.999;
			max_iters  = 0;
			elitism    = 0.0;
			children   = 2;
			info_iters = 100;
			autosave   = 0;
			parallel   = OpenMP;
		}
	};

	
	// constructor, destructor
	GA(): err_id(OK), _iter(0), _popsize(0), _log(nullptr) {}
	virtual ~GA() {}

	// checks if no errors occured during allocation/load/save
	explicit operator bool() const { return err_id==OK; }

	// parameters and logging
	void    set_params(const params& opt) { _opt = opt; }
	params  get_params() const { return _opt; }
	void    set_logging_stream(Logger* log) { _log = log; }
	size_t  generations() const { return _iter; }
	
	// get a specific member of the population
	// in first place (idx=0) is the best member of the population.
	member  get_member(int num=0) const { 
		assert(num >= 0 && num < _popsize);
		return _population[_sorted[num]]; 
	}

	member& member_at(int num=0) { 
		assert(num >= 0 && num < _popsize);
		return _population[_sorted[num]]; 
	}

	// return the greatest of all time (G.O.A.T)
	member  solution() const { return _goat; }

	// number of elites
	int  elites() const { return (int)(1+_popsize*_opt.elitism+0.5); }

	// reset generation counter
	void reset() { _iter = 0; }

	// initialize population
	template <class Initializer>
	void init(int chainsize, int popsize, Initializer& initializer);

	// set member's fitness
	void set_member_fitness(int idx, double f);

	// evaluate a single member
	template <class Fitness>
	void evaluate_member(int idx, Fitness& ff);

	// evaluate current generation
	template <class Fitness>
	void evaluate(Fitness& ff);

	// sort population and save current state
	void sort();

	// evolve next generation
	template <class Crossover, class Mutation>
	void evolve(Crossover& xo, Mutation& mut);

	// save state during evolution
	void save_state();
	
	// save final state
	void finish();

	// for convenience, runs the evolution cycle
	template <class Initializer, class Fitness, class Crossover, class Mutation>
	void solve(int chainsize, int popsize, Initializer& initializer, 
			   Fitness& ff, Crossover& xo, Mutation& mut, bool evolve_first=false);

	// saves the population's parameters in a disk file
	bool save(const std::string& filename);
	
	// loads the population's parameters from a disk file
	bool load(const std::string& filename);

	// returns a std::string with the description of the error in err_id
	std::string error_description() const;

	// returns info about ga's parameters
	std::string info() const;
	
	// returns the total time elapsed since ga started
	std::string total_time() const;
	
	// displays progress info every 'params.info_iters' iterations in stdout.
	// override to change that, eg in a GUI. 
	virtual void show_progress_info();

	// early stopping
	bool early_stop() const { return solution().fitness >= _opt.threshold; }

	// determines if GA is done
	bool done() const { return early_stop() || (_opt.max_iters > 0 && _iter > _opt.max_iters); }

 protected:
	// mating process, returns the next generation
	template <class Crossover> 
	void mating(std::vector<member>& nextgen, Crossover& xo);
	
	// save/load member's chain
	bool save_chain(std::ostream& os, const std::vector<Type>& chain);
	bool load_chain(std::istream& is, std::vector<Type>& chain);
	
 protected:
	// private data
	params               _opt;
	size_t               _iter;
	int                  _chainsize;
	int                  _popsize;
	Logger*              _log;
	member               _goat;
	std::vector<member>  _population;
	std::vector<int>     _sorted;
	time_point           _time0, _time1;
};



////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////

template <typename Type>
template <class Initializer>
void GA<Type>::init(int chainsize, int popsize, Initializer& initializer)
{
	assert(chainsize > 0);

	_chainsize = chainsize;
	_popsize = popsize;
	
	// allocate and initialize population
	_sorted.resize(_popsize);
	for (int i=0; i<_popsize; ++i) _sorted[i] = i;
	_population.resize(_popsize);
	for (int i=0; i<_popsize; ++i) {
		_population[i].evaluated = false;
		_population[i].fitness = 0.0;
		_population[i].chain.resize(_chainsize);
		initializer.apply(_population[i].chain);
	}
	_goat.fitness = 0.0;

	// start timer
	_time0 = _time1 = std::chrono::steady_clock::now();

	// set iteration counter to 1
	_iter++;
}

template <typename Type>
void GA<Type>::set_member_fitness(int idx, double f) 
{
	_population[idx].fitness = f;
	_population[idx].evaluated = true;
}

template <typename Type>
template <class Fitness>
void GA<Type>::evaluate_member(int idx, Fitness& ff) 
{
	if (!_population[idx].evaluated) {
		_population[idx].fitness = ff(_population[idx].chain);
		_population[idx].evaluated = true;
	}
}

template <typename Type>
template <class Fitness>
void GA<Type>::evaluate(Fitness& ff) {
	if (_opt.parallel==SingleThread) {
		for (int i=0; i<_popsize; ++i) evaluate_member(i, ff);
	} else if (_opt.parallel==OpenMP) {
		#pragma omp parallel for num_threads(openmp<>::threads)
		for (int i=0; i<_popsize; ++i) evaluate_member(i, ff);
	}
}

template <typename Type>
void GA<Type>::sort()
{
	// sort members by fitness
	for (int i=0; i<_popsize; ++i) _sorted[i] = i;

	struct orderby_fitness {
		const std::vector<member>& _pop;
		orderby_fitness(const std::vector<member>& pop): _pop(pop) {}
		bool operator() (int a, int b) const { return (_pop[a].fitness > _pop[b].fitness); }
	};
	orderby_fitness order_by(_population);
	std::sort(_sorted.begin(), _sorted.end(), order_by);
	if (get_member(0).fitness > _goat.fitness) _goat = get_member(0);

	// show progress info and autosave
	save_state();

	// advance iteration counter
	_iter++;
}

template <typename Type>
template <class Crossover, class Mutation>
void GA<Type>::evolve(Crossover& xo, Mutation& mut)
{
	std::vector<member> nextgen;
	
	// mating (crossover)
	mating(nextgen, xo);
		
	// mutation
	for (int i=elites(); i<_popsize; ++i) mut.apply(nextgen[i].chain);

	// check for duplicates between last generation and the next
	for (int i=0; i<_popsize; ++i) {
		if (nextgen[i].evaluated) continue;
		for (int j=0; j<_popsize; ++j) {
			if (nextgen[i].chain==_population[j].chain) {
				nextgen[i].fitness = _population[j].fitness;
				nextgen[i].evaluated = true;
				break;
			}
		}
	}

	// replace the current population with the next generation
	for (int i=0; i<_popsize; ++i) _population[i] = nextgen[i];
}

template <typename Type>
void GA<Type>::save_state()
{
	if (_opt.info_iters > 0 && _iter % _opt.info_iters==0) show_progress_info();
	if (_opt.autosave > 0 && _iter % _opt.autosave==0) save(_opt.filename);
}

template <typename Type>
void GA<Type>::finish()
{
	if (_opt.info_iters > 0 && _iter % _opt.info_iters != 0) show_progress_info();
	if (_opt.autosave > 0 && _iter % _opt.autosave != 0) save(_opt.filename);
}

template <typename Type>
template <class Initializer, class Fitness, class Crossover, class Mutation>
void GA<Type>::solve(int chainsize, int popsize, Initializer& initializer, 
					 Fitness& ff, Crossover& xo, Mutation& mut, bool evolve_first)
{
	_iter = 0;
	init(chainsize, popsize, initializer);
	if (evolve_first) evolve(xo, mut);
	for (;;) {
		evaluate(ff);
		sort();
		if (done()) break;
		evolve(xo, mut);
	}
	finish();
}

template <typename Type>
template <class Crossover>
void GA<Type>::mating(std::vector<member>& nextgen, Crossover& xo)
{
	// total fitness of the population
	double sum = 0.0;
	for (int i=0; i<_popsize; ++i) {
		sum += _population[i].fitness;
	}

	// create the next generation
	nextgen.clear();
	nextgen.reserve(_popsize);
	
	// elitism: copy the best members to the new population
	int elts = elites();
	for (int i=0; i<elts; ++i) nextgen.push_back(_population[_sorted[i]]);

	// wheel selection
	while (nextgen.size() < (size_t)_popsize) {
		// select parents
		int p1=-1, p2=-1;
		while (p1==-1 || p2==-1) {
			double slice = uniform_random_real(0.0, sum);
			int p = 0;
			double cumulative = 0.0;
			for (int i = 0; i < _popsize; ++i) {
				cumulative += _population[i].fitness;
				if (cumulative >= slice) { p = i; break; }
			}
			if (p1==-1) p1 = p;
			else if (p != p1) p2 = p;
		}
		// generate children
		size_t nchildren = _opt.children;
		if (nchildren > 2) nchildren = 2;
		if (nchildren < 1) nchildren = 1;
		if (nchildren==1 || nextgen.size()+nchildren > (size_t)_popsize) {
			// generate one child
			member c1;
			c1.evaluated = false;
			c1.chain.resize(_chainsize);
			xo.apply(_population[p1].chain, _population[p2].chain, c1.chain);
			nextgen.push_back(c1);
		} else {
			// generate two children
			member c1, c2;
			c1.evaluated = c2.evaluated = false;
			c1.chain.resize(_chainsize);
			c2.chain.resize(_chainsize);
			xo.apply(_population[p1].chain, _population[p2].chain, c1.chain, c2.chain);
			nextgen.push_back(c1);
			nextgen.push_back(c2);
		}
	}
}

template <typename Type>
bool GA<Type>::save_chain(std::ostream& os, const std::vector<Type>& chain)
{
	for (int j=0; j<_chainsize; ++j) {
		double d = (double)(chain[j]);
		os.write((char*)&d, sizeof(double));
	}
	return true;
}

template <typename Type>
bool GA<Type>::load_chain(std::istream& is, std::vector<Type>& chain)
{
	for (int j=0; j<_chainsize; ++j) {
		double d;
		is.read((char*)&d, sizeof(double));
		chain[j] = Type(d);
	}
	return true;
}

// vector<bool> specialization
template <>
bool GA<bool>::save_chain(std::ostream& os, const std::vector<bool>& chain)
{
	unsigned char b = 0;
	int bitpos = 0;
	for (int j=0; j<_chainsize; ++j) {
		if (bitpos <= 7) {
			if (chain[j]) b |= (1 << bitpos);
			++bitpos;
			if (bitpos==8 || j==_chainsize-1) {
				os.write((char*)&b, sizeof(char));
				b = 0;
				bitpos = 0;
			}
		}
	}
	return true;
}

// vector<bool> specialization
template <>
bool GA<bool>::load_chain(std::istream& is, std::vector<bool>& chain)
{
	int bytes = (_chainsize+7) / 8;
	int j = 0;
	for (int i=0; i<bytes; ++i) {
		unsigned char b;
		is.read((char*)&b, sizeof(char));
		for (int k=0; k < 8 && j < _chainsize; ++k, ++j)
			chain[j] = b & (1 << k);
	}
	return true;
}


template <typename Type>
bool GA<Type>::save(const std::string& filename)
{
	int n;
	std::ofstream os;
	os.open(filename, std::ios::out|std::ios::binary);
	if (!os.is_open()) { err_id=NotExists; return false; }
	os << "GA." << "00"/*version*/ << ".";
	// iteration count
	n = (int)_iter;
	os.write((char*)&n, sizeof(int));
	// chain size
	n = _chainsize;
	os.write((char*)&n, sizeof(int));
	// population size
	n = _popsize; 
	os.write((char*)&n, sizeof(int));
	// save members
	for (int i=0; i<_popsize; ++i) {
		member& m = _population[_sorted[i]];
		os.write((char*)&m.fitness, sizeof(double));
		save_chain(os, m.chain);
	}
	// save GOAT
	os.write((char*)&_goat.fitness, sizeof(double));
	save_chain(os, _goat.chain);

	os.close();
	if (!os) { err_id=BadIO; return false; }
	return true;
}

// load allocates the memory needed to store the saved population
template <typename Type>
bool GA<Type>::load(const std::string& filename)
{
	char buff[8];
	int n;
	double d;

	std::ifstream is;
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { err_id=NotExists; return false; }
	is.read(buff, 3);
	buff[3] = '\0';
	if (std::string(buff) != "GA.") { err_id=BadMagic; return false; }
	is.read(buff, 3);
	if (buff[2] != '.') { err_id=BadMagic; return false; }
	//version = ('0'-buff[0])*10 + '0'-buff[1];
	// iteration count
	is.read((char*)&n, sizeof(int));
	_iter = (size_t)n;
	// chain size
	is.read((char*)&n, sizeof(int));
	_chainsize = n;
	// population size
	is.read((char*)&n, sizeof(int));
	_popsize = n;
	// allocate memory for the stored population
	_population.clear(); _population.reserve(_popsize);
	_sorted.clear(); _sorted.reserve(_popsize);
	for (int i=0; i<_popsize; ++i) _sorted.push_back(i);
	// load members 
	for (int i=0; i<_popsize; ++i) {
		member m;
		is.read((char*)&d, sizeof(double));
		m.fitness = d;
		m.evaluated = true;
		m.chain.resize(_chainsize);
		load_chain(is, m.chain);
		_population.push_back(m);
	}
	// load GOAT
	is.read((char*)&d, sizeof(double));
	_goat.fitness = d;
	_goat.chain.resize(_chainsize);
	load_chain(is, _goat.chain);

	is.close();	
	if (!is) { err_id=BadIO; return false; }
	return true;
}

template <typename Type>
std::string GA<Type>::error_description() const 
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
std::string GA<Type>::info() const 
{
	std::stringstream ss;
	ss << "Population size: " << _popsize << " members\nChain size: " << _chainsize << "\n"
	   << "Generations: " << (_iter > 0 ? _iter-1 : 0) << " of " << _opt.max_iters << "\n"
	   << "Early stop threshold: " << _opt.threshold << "\n" << "Elitism ratio: " << _opt.elitism << "\n";
	return ss.str();
}

template <typename Type>
std::string GA<Type>::total_time() const 
{
	time_point time_now = std::chrono::steady_clock::now();
	return std::string("Total time: ") + format_duration(_time0, time_now);
}

template <typename Type>
void GA<Type>::show_progress_info() 
{
	time_point time_now = std::chrono::steady_clock::now();
	double max_fitn = _population[_sorted[0]].fitness;
	double min_fitn = _population[_sorted[_popsize-1]].fitness;
	double sum = 0.0;
	for (int i=0; i<_popsize; ++i) sum += _population[i].fitness;
	double mean_fitn = sum / _popsize;
	std::stringstream ss;
	ss << std::fixed;
	ss << std::setprecision(6);
	// TODO: speed, ETA (like ETA in backprop)
	ss << "Generation " << std::setw(3) << _iter << ", time: " << format_duration(_time1, time_now) << ", " 
	   << "fitness max: " << max_fitn << ", min: " << min_fitn << ", mean: " << mean_fitn << "\n";
	if (_log) *_log << ss.str(); else std::cout << ss.str();
	_time1 = std::chrono::steady_clock::now();
}


};     // namespace umml

#endif // UMML_GA_INCLUDED
