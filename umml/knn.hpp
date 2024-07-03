#ifndef UMML_KNN_INCLUDED
#define UMML_KNN_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 K-Nearest Neighbors (KNN) Classifier.

 FILE:     knn.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: supervised, classification, regression
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 When one feature has very large values, that feature will dominate the distance 
 hence the outcome of the KNN. So the data must be normalized (scaled) using a scaler
 prior to feeding them in the classifier. See scaler.hpp
 If the number of features (X.ncols) is greater than 15, BruteForce method works better
 than kd-Tree. The method 'Auto' automatically chooses the best method for computing
 the nearest neighbors of a point.

 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 STL string
 
 Internal dependencies:
 umml kdtree
 STL unordered_map
  
 Usage example
 ~~~~~~~~~~~~~
 KNN clf;
 clf.train(training_set, training_labels);
 results = clf.classify(test_set);
  
 * Set parameters before train() is called, eg:
 KNN::params opt; 
 opt.k = 10;
 clf.set_params(opt);
 
 * Dataset spliting and scaling:
 dataset<> ds(Xdata, Ylabels);
 ds.split_train_test_sets(0.75);
 ds.get_splits(X_train, X_test, y_train, y_test);
 minmaxscaler<> scaler;
 scaler.fit_transform(X_train);
 scaler.transform(X_test);
  
 TODO
 ~~~~
 * implement weighted distances
 * implement weighted features
 * test cosine similarity: 
   https://booking.ai/k-nearest-neighbours-from-slow-to-fast-thanks-to-maths-bec682357ccd
*/

#include "umat.hpp"
#include "kdtree.hpp"
#include <unordered_map>


namespace umml {


// k-Nearest Neighbors Classifier
template <typename XT=float, typename YT=int>
class KNN
{
 public:
	// methods for computing nearest neighbors
	enum { Auto, KDTree, BruteForce };

	// training parameters
	struct params {
		int    k;          // number of neighbors to use [default: 5]
		int    method;     // method used to compute the nearest neighbors [default: Auto]
		params() {
			k = 5;
			method = Auto;
		}
	};
 
	// constructor, destructor
	KNN(int k=5, int method=Auto) { 
		opt.k = k; 
		opt.method = method;
	}
	virtual ~KNN() {}
	
	void     set_name(const std::string& _name) { name = _name; }
	void     set_params(const params& _opt) { opt = _opt; }
	params   get_params() const { return opt; }
	
	
	// trains the KNN classifier using the 'X_train' set and the 'y_train' labels
	// returns the accuracy of the model in the training set.
	XT       train(const umat<XT>& X_train, const uvec<YT>& y_train);

	// classifies a single sample 'unknown' or a set of samples 'X_unknown' and returns 
	// its/their label(s)
	uvec<YT> classify(const umat<XT>& X_unknown);

	template <template <typename> class Vect>
	YT       classify(const Vect<XT>& unknown);

	// finds the k-nearest neighbors of the point v
	// returns the indeces and the distances of these neighbors
	template <template <typename> class Vect>
	void     k_nearest(const Vect<XT>& v, uvec<int>& indeces, uvec<XT>& dists);

 protected:
	// private methods
	bool     use_kdtree() const;
	
	template <template <typename> class Vect1, template <typename> class Vect2>
	XT       distance(const Vect1<XT>& v, const Vect2<XT>& u) const;

 protected:
	// member data
	params      opt;
	std::string name;
	kdtree<XT>  kd;
	umat<XT>    X;
	uvec<YT>    y;
};



////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////

template <typename XT, typename YT>
XT KNN<XT,YT>::train(const umat<XT>& X_train, const uvec<YT>& y_train)
{
	assert(X.dev()==y.dev() && X.dev()==device::CPU && "KNN operates in CPU memory only");	
	X = X_train;
	y = y_train;
	if (use_kdtree()) {
		kd.use(X);
		kd.build();
	}
	return static_cast<XT>(1.0);
}

template <typename XT, typename YT>
uvec<YT> KNN<XT,YT>::classify(const umat<XT>& X_unknown)
{
	assert(X_unknown.dev()==device::CPU && "KNN operates in CPU memory only");
	int n = X_unknown.ydim();
	uvec<YT> pred(n);
	#pragma omp parallel for default(shared) num_threads(openmp<>::threads)
	for (int k=0; k<n; k++)
		pred(k) = classify(X_unknown.row(k));
	return pred;
}

template <typename XT, typename YT>
template <template <typename> class Vect>
YT KNN<XT,YT>::classify(const Vect<XT>& unknown)
{
	uvec<int> idcs(opt.k);
	uvec<XT> dists(opt.k);
	k_nearest(unknown, idcs, dists);

	// count frequencies
	std::unordered_map<YT,int> count;
	for (int i=0; i<idcs.len(); ++i) ++count[y(idcs(i))];
	YT pred = 0;
	int cnt = 0;
	for (const auto& p : count) {
		if (p.second > cnt) {
			pred = p.first;
			cnt = p.second;
		}
	}
	return pred;
}

template <typename XT, typename YT>
template <template <typename> class Vect>
void KNN<XT,YT>::k_nearest(const Vect<XT>& v, uvec<int>& indeces, uvec<XT>& dists)
{
	if (use_kdtree()) {
		kd.k_nearest(opt.k, v, indeces, dists);
	} else {
		int n = X.ydim();
		typedef std::pair<XT,int> PT;
		std::priority_queue<PT, std::vector<PT>, std::greater<PT>> pq;
		for (int i=0; i<n; ++i)
			pq.push(std::make_pair(distance(X.row(i),v), i));
		for (int i=0; i<opt.k; ++i) {
			indeces(i) = pq.top().second;
			dists(i) = pq.top().first;
			pq.pop();
		}
	}
}

template <typename XT, typename YT>
template <template <typename> class Vect1, template <typename> class Vect2>
XT KNN<XT,YT>::distance(const Vect1<XT>& v, const Vect2<XT>& u) const
{
	XT sum = 0;
	for (int i=0; i<v.len(); ++i) {
		XT diff = v(i) - u(i);
		sum += diff*diff;
	}
	return sum;
}

template <typename XT, typename YT>
bool KNN<XT,YT>::use_kdtree() const
{
	return opt.method==KDTree || (opt.method==Auto && X.xdim() < 15);
}


};     // namespace umml

#endif // UMML_KNN_INCLUDED
