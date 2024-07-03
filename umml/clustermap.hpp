#ifndef UMML_CLUSTERMAP_INCLUDED
#define UMML_CLUSTERMAP_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Use clustering algorithm for classification.

 FILE:     clustermap.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: supervised, classification, regression
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~

 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 STL string
 
 Internal dependencies:
 umml preprocessing
  
 Usage example
 ~~~~~~~~~~~~~
 [Clustering eg Kmeans] cls;
 cls.cluster(n_clusters, X_train);
 cls.clustering_info(prototypes, ...);  
 ClusterMap<> cmap;
 cmap.map(prototypes, X_train, y_train);
 pred = cmap.classify(X_test);
  
 TODO
 ~~~~
*/

#include "umat.hpp"
#include "preproc.hpp"


namespace umml {


// ClusterMap Classifier based on clustering
template <typename XT=float, typename YT=int>
class ClusterMap
{
 public:
	// methods for computing dinstances
	enum { Manhattan, Euclidean };

	// training parameters
	struct params {
		int method;     // method used to compute the nearest neighbors [default: Euclidean]
		params() {
			method = Euclidean;
		}
	};
 
	// constructor, destructor
	ClusterMap(int method=Euclidean) { 
		opt.method = method;
		threads = 1;
	}
	virtual  ~ClusterMap() {}
	
	void      set_name(const std::string& _name) { name = _name; }
	void      set_params(const params& _opt) { opt = _opt; }
	params    get_params() const { return opt; }
	void      set_openmp_threads(int nthreads) { threads = nthreads; }
	
	
	// trains the classifier using the 'X_train' set and the 'y_train' labels
	void      map(const umat<XT>& proto, const umat<XT>& X, const uvec<YT>& y);
	uvec<int> get_mapping() const { return assignments; }

	// classifies a single sample 'unknown' or a set of samples 'X_unknown' and returns 
	// its/their label(s)
	uvec<YT>  classify(const umat<XT>& X_unknown);

	template <template <typename> class Vect>
	YT        classify(const Vect<XT>& unknown);

 protected:
	// private methods
	template <template <typename> class Vect1, template <typename> class Vect2>
	XT       euclidean_distance(const Vect1<XT>& v, const Vect2<XT>& u) const;
	template <template <typename> class Vect1, template <typename> class Vect2>
	XT       manhattan_distance(const Vect1<XT>& v, const Vect2<XT>& u) const;

 protected:
	// member data
	params      opt;
	std::string name;
	int         threads;
	umat<XT>    protos;
	uvec<YT>    unique;
	int         n_classes;
	uvec<int>   assignments;
};



////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////

template <typename XT, typename YT>
void ClusterMap<XT,YT>::map(const umat<XT>& proto, const umat<XT>& X, const uvec<YT>& y)
{
	protos = proto;
	int nrows = X.ydim();
	int k = protos.ydim();
	unique = uniques(y);
	n_classes = unique.len();
	umat<int> counts(k,n_classes);
	counts.zero_active_device();
	uvec<XT> dists(k);
	assignments.resize(k);
	for (int i=0; i<nrows; ++i) {
		if (opt.method==Manhattan) {
			#pragma omp parallel for default(shared) num_threads(threads)
			for (int c=0; c<k; c++) 
				dists(c) = manhattan_distance(X.row(i), protos.row(c));
		} else if (opt.method==Euclidean) {
			#pragma omp parallel for default(shared) num_threads(threads)
			for (int c=0; c<k; c++) 
				dists(c) = euclidean_distance(X.row(i), protos.row(c));
		}
		int cl = dists.argmin();
		int lb = unique.find_first(y(i));
		++counts(cl, lb);
	}
	for (int c=0; c<k; c++) {
		assignments(c) = unique(counts.row(c).argmax());
		//std::cerr << "cluster " << c << ": " << r.format() << "\n";
	}
}

template <typename XT, typename YT>
uvec<YT> ClusterMap<XT,YT>::classify(const umat<XT>& X_unknown)
{
	int nrows = X_unknown.ydim();
	uvec<YT> pred(nrows);
	#pragma omp parallel for default(shared) num_threads(threads)
	for (int i=0; i<nrows; ++i) {
		pred(i) = classify(X_unknown.row(i));
	}
	return pred;
}

template <typename XT, typename YT>
template <template <typename> class Vect>
YT ClusterMap<XT,YT>::classify(const Vect<XT>& unknown)
{
	int k = protos.ydim();
	uvec<XT> dists(k);
	switch (opt.method) {
		case Manhattan:
		for (int c=0; c<k; c++) 
			dists(c) = manhattan_distance(unknown, protos.row(c));
		break;
		case Euclidean:
		for (int c=0; c<k; c++) 
			dists(c) = euclidean_distance(unknown, protos.row(c));
		break;
	}
	return assignments(dists.argmin());
	//return unique(assignments(dists.argmin()));
}

template <typename XT, typename YT>
template <template <typename> class Vect1, template <typename> class Vect2>
XT ClusterMap<XT,YT>::euclidean_distance(const Vect1<XT>& v, const Vect2<XT>& u) const
{
	XT sum = 0;
	for (int i=0; i<v.len(); ++i) {
		XT diff = v(i) - u(i);
		sum += diff*diff;
	}
	return sum;
}

template <typename XT, typename YT>
template <template <typename> class Vect1, template <typename> class Vect2>
XT ClusterMap<XT,YT>::manhattan_distance(const Vect1<XT>& v, const Vect2<XT>& u) const
{
	XT sum = 0;
	for (int i=0; i<v.len(); ++i) sum += static_cast<XT>(std::abs(v(i) - u(i)));
	return sum;
}


};     // namespace umml

#endif // UMML_CLUSTERMAP_INCLUDED
