#ifndef UMML_SYMVOL_INCLUDED
#define UMML_SYMVOL_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Genetic Algorithms: Symvol genetic algorithm.

 FILE:     symvol.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     1995, 2024
 
 Namespace
 ~~~~~~~~~
 mml::mutation::values
 mml::mutation::permut
 
 Notes
 ~~~~~
 fitness must be in [0..1], fitness=1 is an optimal solution.
 
 Dependencies
 ~~~~~~~~~~~~
 umml algo
 umml rand
 umml bitvec
 STL string
 STL vector
 STL algorithm
  
 Usage example
 ~~~~~~~~~~~~~

*/

/*
 Symvol class implementation.
 
 Artificial Intelligence - Genetic Algorithms
 Copyright (c) 1995 by Anastasios Milikas
*/

#include "../compiler.hpp"
#include "../logger.hpp"
#include "../utils.hpp"
#include "../rand.hpp"
#include "bitvec.hpp"
#include "initializer.hpp"
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>


namespace umml {


class Symvol
{
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
		std::vector<int>    chain;
		std::vector<double> attractions;
		std::vector<int>    replaced;
		double              fitness;
		double              prev_fitness;
		bool                evaluated;
	};

	// Symvol parameters
	struct params {
		double      threshold;    // stopping: fitness better than threshold [default: 0.999]
		size_t      max_iters;    // stopping: max number of iterations [default: 0 - until convergence]
		double      amin;         // attraction minimum value 0..1 [default: 0.02]
		double      agrow;        // attraction growth factor 0..1 [default: 0.1 (10%)]
		double      elitism;      // elitism: percent of best members that pass to the next generation [default: 0.0]
		double      randomness;   // mutation, probability to choose a random symbol 0..0.5 [default: 0.2]
		size_t      info_iters;   // display info every info_iters iterations [default 0 - no info]
		std::string filename;     // filename for saving the population's parameters
		size_t      autosave;     // autosave iterations [default: 0 - no autosave]
		int         parallel;     // parallel mode [default: OpenMP]

		params() {
			threshold  = 0.999;
			max_iters  = 0;
			amin       = 0.02;
			agrow      = 0.1;
			elitism    = 0.0;
			randomness = 0.2;
			info_iters = 0;
			autosave   = 0;
			parallel   = OpenMP;
		}
	};

 public:
	Symvol(): err_id(OK), _iter(0), _symsize(0), _chainsize(0), _popsize(0), _log(nullptr) {}
	virtual ~Symvol() {};

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
	int  elites() const { return (int)(_popsize*_opt.elitism+0.5); }

	// reset generation counter
	void reset() { _iter = 0; }

	// generate symbols
	std::vector<bitvec> generate_symbols(int symbits) const;

	// initialize population
	template <class Initializer>
	void init(int chainsize, int popsize, Initializer& initializer, const std::vector<bitvec>& symbols);

	// set member's fitness
	void set_member_fitness(int idx, double f);

	// convert member's chain to a standard bitstring (bool)
	std::vector<bool> to_bitstring(const std::vector<int>& chain) const;

	// evaluate a single member
	template <class Fitness>
	void evaluate_member(int idx, Fitness& ff);

	// evaluate current generation
	template <class Fitness>
	void evaluate(Fitness& ff);

	// sort population and save current state
	void sort();

	// evolve next generation
	void evolve();

	// save state during evolution
	void save_state();
	
	// save final state
	void finish();

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
	bool early_stop() const {	return solution().fitness >= _opt.threshold; }

	// determines if GA is done
	bool done() const { return early_stop() || (_opt.max_iters > 0 && _iter > _opt.max_iters); }

 protected:
	bool save_member(std::ostream& os, const member& m);
	bool load_member(std::istream& is, member& m);

 protected:
	// private data
	params       _opt;
	size_t       _iter;
	int          _symsize;
	int          _chainsize;
	int          _popsize;
	member       _goat;
	Logger*      _log;
	time_point   _time0, _time1;
	std::vector<bitvec>  _symbols;
	std::vector<member>  _population;
	std::vector<int>     _sorted;
	std::vector<double>  _probtable;
};



////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////

std::vector<bitvec> Symvol::generate_symbols(int symbits) const
{
	// allocate and initialize symbols
	int nsymbols = std::pow(2, symbits);
	assert (nsymbols > 0 && nsymbols <= 65536); // at most 16-bit symbols
	std::vector<bitvec> symbols;
	symbols.reserve(nsymbols);
	for (int s=0; s<nsymbols; ++s) {
		bitvec v(symbits);
		for (int j=0; j<symbits; ++j) v.set(j, s & (1 << j));
		symbols.push_back(v);
		//std::cout << "s=" << s << ", v=" << v.format() << "\n";
	}
	return symbols;
}

template <class Initializer>
void Symvol::init(int chainsize, int popsize, Initializer& initializer, const std::vector<bitvec>& symbols)
{
	assert(chainsize > 0 && symbols.size() > 0);
	assert(_opt.randomness >= 0.0 && _opt.randomness <= 0.5);

	_symsize = symbols[0].size();
	_chainsize = chainsize;
	_popsize = popsize;
	_symbols = symbols;

	// allocate and initialize population
	_sorted.resize(_popsize);
	for (int i=0; i<_popsize; ++i) _sorted[i] = i;
	_population.resize(_popsize);
	for (int i=0; i<_popsize; ++i) {
		member& m = _population[i];
		m.evaluated = false;
		m.chain.resize(_chainsize);
		initializer.apply(m.chain);
		m.attractions.resize(_chainsize);
		std::fill(m.attractions.begin(), m.attractions.end(), 0.5);
		m.fitness = m.prev_fitness = 0.0;
		m.replaced.clear();
	}
	_goat.fitness = _goat.prev_fitness = 0.0;

	// start timer
	_time0 = _time1 = std::chrono::steady_clock::now();

	// advance iteration counter
	++_iter;
}

void Symvol::set_member_fitness(int idx, double f) 
{
	_population[idx].fitness = f;
	_population[idx].evaluated = true;
}

std::vector<bool> Symvol::to_bitstring(const std::vector<int>& chain) const 
{
	std::vector<bool> v;
	for (int i: chain) {
		std::vector<bool> s = _symbols[i].to_vector();
		v.insert(v.end(), s.begin(), s.end());
	}
	return v;
}

template <class Fitness>
void Symvol::evaluate_member(int idx, Fitness& ff) 
{
	if (!_population[idx].evaluated) {
	    _population[idx].prev_fitness = _population[idx].fitness;
		_population[idx].fitness = ff(to_bitstring(_population[idx].chain));
		_population[idx].evaluated = true;
	}
}

template <class Fitness>
void Symvol::evaluate(Fitness& ff) {
	if (_opt.parallel==SingleThread) {
		for (int i=0; i<_popsize; ++i) evaluate_member(i, ff);
	} else if (_opt.parallel==OpenMP) {
		#pragma omp parallel for num_threads(openmp<>::threads)
		for (int i=0; i<_popsize; ++i) evaluate_member(i, ff);
	}
}

void Symvol::sort()
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
	++_iter;
}

void Symvol::save_state()
{
	if (_opt.info_iters > 0 && _iter % _opt.info_iters==0) show_progress_info();
	if (_opt.autosave > 0 && _iter % _opt.autosave==0) save(_opt.filename);
}

void Symvol::finish()
{
	if (_opt.info_iters > 0 && _iter % _opt.info_iters != 0) show_progress_info();
	if (_opt.autosave > 0 && _iter % _opt.autosave != 0) save(_opt.filename);
}


bool Symvol::save_member(std::ostream& os, const member& m)
{
	for (int i=0; i<_chainsize; i++) {
		if (!os.write((char*)&m.chain[i], sizeof(int))) return false;
		if (!os.write((char*)&m.attractions[i], sizeof(double))) return false;
	}
	if (!os.write((char*)&m.fitness, sizeof(double))) return false;
	if (!os.write((char*)&m.prev_fitness, sizeof(double))) return false;
	int len = (int)m.replaced.size();
	if (!os.write((char*)&len, sizeof(int))) return false;
	for (std::vector<int>::const_iterator it=m.replaced.begin(); it!=m.replaced.end(); it++) {
		int p = *it;
		if (!os.write((char*)&p, sizeof(int))) return false;
	}
	return true;
}

bool Symvol::load_member(std::istream& is, member& m)
{
	m.chain.resize(_chainsize);
	m.attractions.resize(_chainsize);
	for (int i=0; i<_chainsize; i++) {
		if (!is.read((char*)&m.chain[i], sizeof(int))) return false;
		if (!is.read((char*)&m.attractions[i], sizeof(double))) return false;
	}
	if (!is.read((char*)&m.fitness, sizeof(double))) return false;
	if (!is.read((char*)&m.prev_fitness, sizeof(double))) return false;
	int len = 0;
	if (!is.read((char*)&len, sizeof(int))) return false;
	if (len > _chainsize) return false;
	m.replaced.clear();
	while (len > 0) {
		int p=0;
		if (!is.read((char*)&p, sizeof(int))) return false;
		m.replaced.push_back(p);
		len--;
	}
	return true;
}

bool Symvol::save(const std::string& filename)
{
	int n;
	std::ofstream os;
	os.open(filename, std::ios::out|std::ios::binary);
	if (!os.is_open()) { err_id=NotExists; return false; }
	os << "SV." << "00"/*version*/ << ".";
	// chain size
	n = _chainsize;
	os.write((char*)&n, sizeof(int));
	// population size
	n = _popsize; 
	os.write((char*)&n, sizeof(int));
	// generations
	n = (int)_iter; 
	os.write((char*)&n, sizeof(int));
	// save members
	for (int i=0; i<_popsize; ++i) {
		if (!save_member(os, _population[_sorted[i]])) return false;
	}
	// save GOAT
	if (!save_member(os, _goat)) return false;

	os.close();
	if (!os) { err_id=BadIO; return false; }
	return true;
}

bool Symvol::load(const std::string& filename)
{
	char buff[8];
	int n;

	std::ifstream is;
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { err_id=NotExists; return false; }
	is.read(buff, 3);
	buff[3] = '\0';
	if (std::string(buff) != "SV.") { err_id=BadMagic; return false; }
	is.read(buff, 3);
	if (buff[2] != '.') { err_id=BadMagic; return false; }
	//version = ('0'-buff[0])*10 + '0'-buff[1];
	// chain size
	is.read((char*)&n, sizeof(int));
	_chainsize = n;
	// population size
	is.read((char*)&n, sizeof(int));
	_popsize = n;
	// iteration count
	is.read((char*)&n, sizeof(int));
	_iter = (size_t)n;
	// allocate memory for the stored population
	_population.clear(); _population.reserve(_popsize);
	_sorted.clear(); _sorted.reserve(_popsize);
	for (int i=0; i<_popsize; ++i) _sorted.push_back(i);
	// load members 
	for (int i=0; i<_popsize; ++i) {
		member m;
		if (!load_member(is, m)) return false;
		_population.push_back(m);
	}
	// load GOAT
	if (!load_member(is, _goat)) return false;

	is.close();	
	if (!is) { err_id=BadIO; return false; }
	return true;
}

std::string Symvol::error_description() const 
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

std::string Symvol::info() const 
{
	std::stringstream ss;
	ss << "Population size: " << _popsize << " members\nChain size: " << _chainsize << "\n"
	   << "Number of symbols: " << _symbols.size() << "\nSymbol size: " << _symbols[0].size() << " bits\n"
	   << "Generations: " << (_iter > 0 ? _iter-1 : 0) << " of " << _opt.max_iters << "\n"
	   << "Early stop threshold: " << _opt.threshold << "\n";
	return ss.str();
}

std::string Symvol::total_time() const 
{
	time_point time_now = std::chrono::steady_clock::now();
	return std::string("Total time: ") + format_duration(_time0, time_now);
}

void Symvol::show_progress_info() 
{
	time_point time_now = std::chrono::steady_clock::now();
	double max_fitn = member_at(0).fitness;
	double min_fitn = member_at(_popsize-1).fitness;
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


void Symvol::evolve()
{
	// evaluate the changes from previous generation and update symbol's attractions
	if (_iter > 0) {
		for (int i=0; i<_popsize; ++i) {
			member& m = _population[i];
			// grow all attractions by the growing value agrow 
			for (int j=0; j<_chainsize; ++j) m.attractions[j] *= (1+_opt.agrow);

			// increase/decrease attraction according to the amount of fitness improvement/deterioration
			for (int pos: m.replaced) {
			    m.attractions[pos] *= (std::exp(2.7*(m.fitness*(m.prev_fitness-m.fitness))));
			}

			// crop attractions to [amin..1]
			for (int j=0; j<_chainsize; ++j) {
				if (m.attractions[j] > 1.0) m.attractions[j] = 1.0;
				if (m.attractions[j] < _opt.amin) m.attractions[j] = _opt.amin;
			}
		}
	}
	
	// calculate wheel from of members fitnesses
	std::vector<double> wheel(_popsize);
	double totf = 0.0;
	for (int i=0; i<_popsize; ++i) {
		totf += _population[i].fitness;
		wheel[i] = totf;
	}
	
	std::vector<double> probtable(_chainsize);
	//for (int i=0; i<_popsize; ++i) {
	for (int i=elites(); i<_popsize; ++i) {
		member& m = member_at(i); //_population[i];
		m.replaced.clear();
		m.evaluated = false;
		// build the probtable using member's attractions
		double sum = 0.0;
		for (int j=0; j<_chainsize; ++j) {
			sum += m.attractions[j];
			probtable[j] = sum;
		}

		// change N% of chain symbols
		// N is proportional of member's fitness (f) and randomness (r): N=1-f+r
		/*while (m.replaced.empty())*/ {
			int sym=0, pos=-1;
			//int nperc = std::max(1, (int)(3.0*(1.0-m.fitness)*(0.0+_opt.randomness)*_chainsize+0.5));
			int nperc = std::max(1, (int)((1.0-m.fitness+_opt.randomness)*_chainsize+0.5));
			if (nperc > _chainsize) nperc = _chainsize;
			for (int k=0; k<nperc; ++k) {
				// find a symbol to change (using probtable)
				int tries = 0;
				pos = -1;
				while (pos==-1 && tries < 5) {
					double slice = uniform_random_real<double>(0.0, sum);
					for (int j=0; j<_chainsize; ++j) {
						if (probtable[j] >= slice && m.attractions[j] > _opt.amin &&
							std::find(m.replaced.begin(), m.replaced.end(), j)==m.replaced.end()) { 
							pos = j; 
							break; 
						}
					}
					tries++;
				}

				if (pos==-1) continue;
			
				// get a symbol to replace the one in sympos
				//sym = replace_symbol_strategy(sympos, i);
				double r = uniform_random_real(0.0, 1.0);
				if (r < _opt.randomness) {
					// mutation
					int slen = (int)_symbols.size();
					if (r < _opt.randomness/2) sym = uniform_random_int<int>(0, slen-1);
					else sym = m.chain[pos] + 1;
					if (sym >= slen) sym = 0;
				} else {
					/*
					if (r > 1.0-_opt.randomness) {
						// choose from GOAT
						sym = _goat.chain[pos];
					} else */{
						// choose from members
						double slice = uniform_random_real<double>(0.0, totf);
						for (int j=0; j<_popsize; ++j) {
							if (wheel[j] >= slice) { 
								sym = _population[j].chain[pos]; 
								break; 
							}
						}
					}
				}
			
				// place the symbol into the chain
				m.chain[pos] = sym;
				m.replaced.push_back(pos);
			}
		}
	}
}


}; // namespace umml

#endif
