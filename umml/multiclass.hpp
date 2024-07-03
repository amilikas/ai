#ifndef UMML_MULTICLASS_INCLUDED
#define UMML_MULTICLASS_INCLUDED

/*
 WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!
 * OVR works ok
 * OVO: TODO
 
 Machine Learning Artificial Intelligence Library
 Multi-class classifiers.

 FILE:     multiclass.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: clustering
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 These template classes implement multi-class classification by combining _binary_ classifiers
 with One-vs-Rest (OVR) and One-vs-One (OVO) methods.
 Only binary classifiers that have 'output' and 'output_label(s)' methods (like LSVM) can be
 used with these templates.
 
 [1] Jason Brownlee: One-vs-Rest and One-vs-One for Multi-Class Classification
 https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/

 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 STL string
 
 Internal dependencies:
 umml preprocessing
 umml openmp
 STL algorithm
 STL file streams
  
 Usage example
 ~~~~~~~~~~~~~
 For a 10-class linear SVM classifier One vs One:
 OVO<double,int,LSVM,10> clf;
 clf.train(training_set, training_labels);
 results (labels) = clf.classify(test_set);
  
 * Set parameters before train() is called, eg:
   LSVM<>::params params; 
   params.learnrate = 0.1;
   clf.set_params(params);

 * OVR and OVO converts the labels for binary (2-class) classification
   internally with the binary_conv class.
   
 * Dataset spliting and scaling:
   dataset<> ds(Xdata, Ylabels);
   ds.split_train_test_sets(0.75);
   ds.get_splits(X_train, X_test, y_train, y_test);
   minmaxscaler<> scaler;
   scaler.fit_transform(X_train);
   scaler.transform(X_test);
*/

#include "umat.hpp"
#include "preproc.hpp"
#include "logger.hpp"
#include <fstream>
#include <iostream>


namespace umml {


/// One vs Rest multi-class classifier
/// XT:  dataset type
/// YT:  labels type
/// CLF: classifier
/// N:   number of classes
template <typename XT, typename YT, template <typename, typename> class CLF, int N>
class OVR
{
 public: 
	// constructor, destructor
	OVR() {}
	virtual ~OVR() {}
	
	void     set_name(const std::string& _name) { name = _name; }
	void     set_params(const typename CLF<XT,YT>::params& _opt);
	void     set_logging_stream(Logger* _log);
	
	// trains the classifiers using the 'X_train' set and the 'y_train' labels
	// returns the accuracy of the model in the training set.
	XT       train(const umat<XT>& X_train, const uvec<YT>& y_train);

	// classifies a sample 'x' or a set of samples 'X_unknown' and returns 
	// the label(s)
	template <template <typename> class Vector>
	YT       classify(const Vector<XT>& unknown);
	vect<YT> classify(const umat<XT>& X_unknown);

	// displays progress info every 'params.info_iters' iterations in stdout.
	// override to change that, eg GUI. 
	virtual void show_progress_info();

	// saves the state of the classifier in a disk file
	bool save(const std::string& filename);
	
	// loads the state of the classifier from a disk file
	bool load(const std::string& filename);

 protected:
	// member data
	std::string name;
	CLF<XT,YT>  clfs[N];
	uvec<YT>    yc_train[N];
	uvec<YT>    u;
	Logger*     log;
};



////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////

template <typename XT, typename YT, template <typename, typename> class CLF, int N>
void OVR<XT,YT,CLF,N>::set_params(const typename CLF<XT,YT>::params& _opt)
{
	for (int i=0; i<N; ++i) clfs[i].set_params(_opt);
}

template <typename XT, typename YT, template <typename, typename> class CLF, int N>
void  OVR<XT,YT,CLF,N>::set_logging_stream(Logger* _log) 
{ 
	log = _log; 
	for (int i=0; i<N; ++i) clfs[i].set_logging_stream(log);
}

template <typename XT, typename YT, template <typename, typename> class CLF, int N>
XT OVR<XT,YT,CLF,N>::train(const umat<XT>& X_train, const uvec<YT>& y_train)
{
	u = uniques(y_train);
	
	// if calculated classes is not the same as N, return
	if (u.len() != N) return 0.0;
	
	// preprocess
	for (int i=0; i<N; ++i) 
		binary_conv<YT>(u(i)).convert(y_train, yc_train[i]);
		
	// training
	XT acc[N];
	#pragma omp parallel for default(shared) num_threads(openmp<>::threads)
	for (int i=0; i<N; ++i) acc[i] = clfs[i].train(X_train, yc_train[i]);

	// return mean training set accuracy
	XT sum = 0;
	for (int i=0; i<N; ++i) sum += acc[i];
	return sum / N;
}

template <typename XT, typename YT, template <typename, typename> class CLF, int N>
template <template <typename> class Vector>
YT OVR<XT,YT,CLF,N>::classify(const Vector<XT>& unknown)
{
	uvec<XT> outputs(N);
	for (int i=0; i<N; ++i) outputs(i) = clfs[i].output(unknown);
	return u(outputs.argmax());
}

template <typename XT, typename YT, template <typename, typename> class CLF, int N>
uvec<YT> OVR<XT,YT,CLF,N>::classify(const umat<XT>& X_unknown)
{
	int nrows = X_unknown.ydim();
	int ncols = X_unknown.xdim();
	uvec<YT> pred(nrows);
	for (int k=0; k<nrows; k++) {
		pred.set_element(k, classify(X_unknown.row(k)));
	}
	return pred;
}

template <typename XT, typename YT, template <typename, typename> class CLF, int N>
void OVR<XT,YT,CLF,N>::show_progress_info()
{
	for (int i=0; i<N; ++i) clfs[i].show_progress_info();
}


};     // namespace umml

#endif // UMML_MULTICLASS_INCLUDED
