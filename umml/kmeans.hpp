#ifndef UMML_KMEANS_INCLUDED
#define UMML_KMEANS_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 K-Means clustering.

 FILE:     kmeans.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: unsupervised, clustering
 
 Namespace
 ~~~~~~~~~
 mml
 
 Notes
 ~~~~~
 The class KMeans implements the K-Means clustering algorithm.

 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 STL string
 
 Internal dependencies:
 umml algorithm
 umml rand
 STL limits
  
 Usage example
 ~~~~~~~~~~~~~
 KMeans<> km;
 int n_clusters = 2;
 ivec cl = km.cluster(n_clusters, X);
  
 * Set parameters before cluster() is called, eg:
 KMeans<>::params opt;
 opt.max_iters = 1000;
 km.set_params(opt);
  
 TODO
 ~~~~
*/


#include "umat.hpp"
#include "logger.hpp"
#include <fstream>
#include <limits>


namespace umml {


/*
 KMeans clustering
 
 1. initialize k centroids (clusters) choosing randomly k points of X
 2. for every Xi calculate distances from the k centroids and assign to the closest centroid
 3. calculate the center of mass for each cluster
 4. compute new centroids and reassign Xi 
*/ 
template <typename XT=float, typename YT=int>
class KMeans
{
 public:
	// training parameters
	struct params {
		double tolerance;  // determines when the algorithm converged [default: 0.0001]
		size_t max_iters;  // maximum number of iterations [default: 0 - until convergence]
		size_t info_iters; // display info every info_iters iterations [default 10]
		int    rndseed;    // seed rng [default: -1 (random seed if using random device)]
		params() {
			tolerance  = 0.0001;
			max_iters  = 0;
			info_iters = 10;
			rndseed    = -1;
		}
	};
 
	// constructor, destructor
	KMeans(): _initialized(false), _log(nullptr) {}
	virtual ~KMeans() {}
	
	void     set_name(const std::string& __name) { _name = __name; }
	void     set_params(const params& __opt) { _opt = __opt; }
	params   get_params() const { return _opt; }
	void     set_logging_stream(Logger* __log) { _log = __log; }
	
	// seeds the k-means algorithm with initial centroids from 'X_seed'
	// which must have n_clusters rows.
	void     seed(int nclusters, const umat<XT>& X_seed);
	
	// returns the number of clusters found
	int      n_clusters() const { return _centroids.ydim(); }

	// 
	void     unitialize() { _initialized = false; }

	// performs k-means clustering in the dataset 'X' and returns their assigned labels.
	uvec<YT> fit(int nclusters, const umat<XT>& X);

	// clusters the dataset 'X' using learned centroids
	uvec<YT> cluster(const umat<XT>& X);

	// get the centroids found by cluster() and samples per cluster.
	void     clustering_info(umat<XT>& centroids, uvec<int>& samples_per_cluster);

	// displays progress info every 'params.info_iters' iterations in stdout.
	// override to change that, eg GUI. 
	virtual void show_progress_info();

	// saves the state of the classifier in a disk file
	bool    save(const std::string& filename);
	
	// loads the state of the classifier from a disk file
	bool    load(const std::string& filename);

 protected:
	// private methods
	template <template <typename> class Vect1, template <typename> class Vect2>
	XT      distance(const Vect1<XT>& v, const Vect2<XT>& u) const;

 protected:
	// member data
	params      _opt;
	std::string _name;
	bool        _initialized;
	umat<XT>    _centroids;
	uvec<int>   _spc;
	size_t      _iter;
	XT          _convergence;
	Logger*     _log;
};



////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////


template <typename XT, typename YT>
void KMeans<XT,YT>::seed(int nclusters, const umat<XT>& X_seed)
{
	assert(X_seed.dev()==device::CPU && "KMeans operates in CPU memory only");
	if (X_seed.ydim() != nclusters) return;
	_centroids.resize(nclusters, X_seed.xdim());
	for (int k=0; k<nclusters; k++) _centroids.set_row(k, X_seed.row_offset(k), X_seed.xdim());
	_initialized = true;
}

template <typename XT, typename YT>
uvec<YT> KMeans<XT,YT>::fit(int nclusters, const umat<XT>& X)
{
	assert(X.dev()==device::CPU && "KMeans operates in CPU memory only");
	
	int nrows = X.ydim();
	int ncols = X.xdim();
	constexpr XT inf = std::numeric_limits<XT>::max();
	
	// cluster predictions for dataset 'X'
	uvec<YT> pred(nrows);
	pred.zero_active_device();
	
	// distance (squared) of the X points from their associated cluster
	uvec<XT> dists(nrows);
	dists.set(inf);
	
	// center of mass (com) for each cluster and samples per cluster (spc)
	_spc.resize(nclusters);
	umat<XT> com(nclusters, ncols);
	umat<XT> prev_com(nclusters, ncols);
	prev_com.zero_active_device();
	
	// 1. initialize centroids by choosing randomly 'nclusters' points of X if not
	// already initialized with KMeans::seed() method 
	if (!_initialized || _centroids.ydim() != nclusters || _centroids.xdim() != ncols) {
		_centroids.resize(nclusters, ncols);
		std::vector<int> ptidcs(nrows);
		build_shuffled_indeces(ptidcs, nrows);
		for (int k=0; k<nclusters; k++)
			_centroids.set_row(k, X.row_offset(ptidcs[k]), X.xdim());
	}

	// k-means algorithm
	_iter = 0;
	for (;;) {
		// 2. calculate distances from centroids
		for (int k=0; k<nclusters; ++k) {
			#pragma omp parallel for default(shared) num_threads(openmp<>::threads) 
			for (int i=0; i<nrows; ++i) {
				XT d = distance(_centroids.row(k), X.row(i));
				if (d < dists(i)) {
					dists(i) = d;
					pred(i) = k;
				}
			}
		}

		// 3. calculate the center of mass and reset distances
		com.zero_active_device();
		_spc.zero_active_device();
		for (int i=0; i<nrows; ++i) {
			dists(i) = inf;
			int k = pred(i);
			for (int j=0; j<ncols; ++j) com(k,j) += X(i,j);
			++_spc(k);
		}

		// 4. move centroids to the center of mass for each cluster
		for (int k=0; k<nclusters; ++k) {
			int n = _spc(k);
			if (n==0) n = 1;
			for (int j=0; j<ncols; ++j) _centroids(k,j) = com(k,j) / n;
		}		
		
		// check if algorithm has converged
		_convergence = 0;
		for (int k=0; k<nclusters; k++)
			_convergence += std::sqrt(distance(com.row(k), prev_com.row(k)));
		if (_convergence <= _opt.tolerance) break;

		_iter++;
		if (_opt.max_iters > 0 && _iter >= _opt.max_iters) break;
		if (_opt.info_iters > 0 && _iter % _opt.info_iters==0) show_progress_info();
	
		// update center of mass
		prev_com = com;
	}
	
	if (_opt.info_iters > 0 && _iter % _opt.info_iters!=0) show_progress_info();
	return pred;
}

template <typename XT, typename YT>
uvec<YT> KMeans<XT,YT>::cluster(const umat<XT>& X)
{
	assert(X.dev()==device::CPU && "KMeans operates in CPU memory only");
	int nrows = X.ydim();
	int ncols = X.xdim();
	int k = _centroids.ydim();
	constexpr XT inf = std::numeric_limits<XT>::max();
	uvec<YT> pred(nrows);
	pred.zero_active_device();
	if (_centroids.xdim() != ncols) return pred;
	uvec<XT> dists(k);
	
	// cluster predictions for dataset 'X' using previously calculated centroids
	for (int i=0; i<nrows; ++i) {
		dists.set(inf);
		for (int c=0; c<k; c++) dists(c) = distance(X.row(i), _centroids.row(c));
		pred(i) = dists.argmin();
	}
	
	return pred;
}

template <typename XT, typename YT>
void KMeans<XT,YT>::clustering_info(umat<XT>& centr, uvec<int>& spc)
{
	assert(centr.dev()==device::CPU && spc.dev()==device::CPU && "KMeans operates in CPU memory only");
	centr = _centroids;
	spc = _spc;
}

template <typename XT, typename YT>
void KMeans<XT,YT>::show_progress_info()
{
	std::stringstream ss;
	if (!_name.empty()) ss << "[" << _name << "] ";
	ss << "Iteration: " << _iter << ", convergence: " << _convergence << "\n";
	if (_log) *_log << ss.str(); 
	else std::cout << ss.str();
}

template <typename XT, typename YT>
bool KMeans<XT,YT>::save(const std::string& filename)
{
	int n;
	double val;
	std::ofstream os;
	os.open(filename, std::ios::out|std::ios::binary);
	if (!os.is_open()) return false;
	os << "KMEANS." << '0' << ".";
	n = _centroids.ydim();
	os.write((char*)&n, sizeof(int));
	n = _centroids.xdim();
	os.write((char*)&n, sizeof(int));
	for (int i=0; i<_centroids.ydim(); ++i) {
		for (int j=0; j<_centroids.xdim(); ++j) {
			val = static_cast<double>(_centroids(i,j));
			os.write((char*)&val, sizeof(double));
		}
	}
	for (int i=0; i<_centroids.ydim(); ++i) os.write((char*)&_spc(i), sizeof(int));
	os.close();	
	return true;
}
	
template <typename XT, typename YT>
bool KMeans<XT,YT>::load(const std::string& filename)
{
	char buff[8];
	double val;
	std::ifstream is;
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) return false;
	is.read(buff, 7);
	buff[7] = '\0';
	if (std::string(buff) != "KMEANS.") return false;
	is.read(buff, 2);
	if (buff[1] != '.') return false;
	int nrows, ncols;
	is.read((char*)&nrows, sizeof(int));
	is.read((char*)&ncols, sizeof(int));
	_centroids.resize(nrows,ncols);
	for (int i=0; i<nrows; ++i) {
		for (int j=0; j<ncols; ++j) {
			is.read((char*)&val, sizeof(double));
			_centroids(i,j) = static_cast<XT>(val);
		}
	}
	_spc.resize(nrows);
	for (int i=0; i<nrows; ++i) is.read((char*)&_spc(i), sizeof(int));
	is.close();	
	return true;
}

template <typename XT, typename YT>
template <template <typename> class Vect1, template <typename> class Vect2>
XT KMeans<XT,YT>::distance(const Vect1<XT>& v, const Vect2<XT>& u) const
{
	assert(v.len()==u.len());
	XT sum = 0;
	for (int i=0; i<v.len(); ++i) {
		XT diff = v(i) - u(i);
		sum += diff*diff;
	}
	return sum;
}


};     // namespace umml

#endif // UMML_KMEANS_INCLUDED
