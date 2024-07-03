#ifndef UMML_METRICS_INCLUDE
#define UMML_METRICS_INCLUDE

/*
 Machine Learning Artificial Intelligence Library
 Metrics for classification and regression tasks.

 FILE:     metrics.h
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 
 Namespace
 ~~~~~~~~~
 umml

 Dependencies
 ~~~~~~~~~~~~
 mml uvec
 mml umat
 
 Internal dependencies:
 STL vector
 STL algorithm
 STL functional
  
 Classification
 ~~~~~~~~~~~~~~
 * confusion_matrix(vector real, vector predicted, mode)
 * accuracy(vector real, vector predicted)
 * accuracy(confusion_matrix)
 * precision(confusion_matrix)
 * recall(confusion_matrix)
 * F1(confusion_matrix)

 Regression
 ~~~~~~~~~~
 * mean_squared_error (MSE)
 * root_mean_squared_error (RMSE)
 * mean_absolute_error (MAE)

 Clustering
 ~~~~~~~~~~
 * silhouette_coefficient

  
 Usage
 ~~~~~
 confmat<> cm = confusion_matrix<>(y_test, y_pred);
 prec = precision<>(cm);
 ...etc.
 
 mse = mean_squared_error(vector real, vector predicted)
 ...etc.
*/

#include <vector>
#include <algorithm>
#include <functional>
#include <limits>


namespace umml {


// confusion matrix evaluation mode
enum {
	CM_Binary,
	CM_Macro,
	CM_Micro,
};

// confusion matrix type
// set binary to true for a binary classifier
//   PREDICTED  
//   ---------- R 
//    +1  -1  | E
// +1 TP  FN  | A
// -1 FP  TN  | L

template <typename Type=int>
struct confmat 
{
	uvec<Type> l;
	umat<int>  m;
	double     accuracy;
	int        mode;
};

template <typename Type>
int __confmat_find_label(const std::vector<Type>& l, const Type& val)
{
	int n = (int)l.size();
	for (int i=0; i<n; ++i) if (l[i]==val) return i;
	return -1;
}

// Builds the confusion (error) matrix needed for precision, recall and F1 metrics.
template <typename Type=int>
confmat<Type> confusion_matrix(const uvec<Type>& real, const uvec<Type>& predicted, int mode=CM_Binary)
{
	assert(real.dev()==predicted.dev());
	assert(real.dev()==device::CPU && "confusion_matrix requires data stored on CPU memory");
	std::vector<Type> ll;
	
	// build the labels list ll and sort it
	for (int i=0; i<real.len(); ++i) {
		int r = __confmat_find_label(ll, real(i));
		// if label not found in already seen labels, append it
		if (r==-1) ll.push_back(real(i));
	}
	for (int i=0; i<predicted.len(); ++i) {
		int r = __confmat_find_label(ll, predicted(i));
		// if label not found in already seen labels, append it
		if (r==-1) ll.push_back(predicted(i));
	}
	std::sort(ll.begin(), ll.end());
	
	// compute the confusion matrix
	confmat<Type> cm;
	cm.mode = mode;
	int n = (int)ll.size();
	cm.l.resize(n);
	cm.l.set(ll);
	cm.m.resize(n, n);
	cm.m.zero_active_device();
	size_t correct= 0;
	for (int i=0; i<real.len(); ++i) {
		int r = __confmat_find_label(ll, real(i));
		if (real(i)==predicted(i)) {
			cm.m(r,r)++;
			correct++;
		} else {
			int p = __confmat_find_label(ll, predicted(i));
			cm.m(r,p)++;
		} 
	}
	cm.accuracy = (double)correct / predicted.len();
	
	return cm;
}


// Accouracy describes the fraction of predictions that the model predicts correctly.
template <typename Type=int>
double accuracy(const uvec<Type>& real, const uvec<Type>& predicted)
{
	int correct = real.count_equal(predicted);
	return (double)correct / real.len();
}

// Multi-label accouracy (one-hot encoded).
template <typename Type=int>
double accuracy(const umat<Type>& real, const umat<Type>& predicted, Type novalue=Type(0))
{
	int correct = real.count_equal(predicted, novalue);
	return (double)correct / real.sum();
}

template <typename Type=int>
double accuracy(const confmat<Type>& cm)
{
	return cm.accuracy;
}


// Precision is a good evaluation metric to use when the cost of a false positive is very high 
// and the cost of a false negative is relatively low.
// binary: tp/(tp+fp)
// multiclass (macro): average tp0/(tp0+fp0), tp1/(tp1+fp1),..., tpn/(tpn+fpn)
// multiclass (micro): (tp0+tp1+...+tpn) / (tp0+...+tpn+fp0+...+fpn)
template <typename Type=int>
double precision(const confmat<Type>& cm)
{
	if (cm.mode==CM_Binary) {
		// binary mode
		return (double)cm.m(0,0)/(cm.m(0,0)+cm.m(1,0));
	} else if (cm.mode==CM_Macro) {
		// multiclass (macro mode)
		double sum= 0.0;
		int n= cm.l.len();
		for (int i=0; i<n; ++i) {
			double tp= cm.m(i,i);
			double tpfp= 0.0;
			for (int k=0; k<n; k++) tpfp += cm.m(k,i);
			if (tpfp) sum += tp/tpfp;
		}
		return sum/n;
	} else {
		// multiclass (micro mode)
		double tp= 0.0;
		for (int i=0; i<cm.l.len(); ++i) tp += cm.m(i,i);
		double fp= 0.0;
		for (int i=1; i<cm.l.len(); ++i) {
			for (int j=0; j<i; ++j) fp += cm.m(i,j);
		}
		return tp/(tp+fp);
	}
}


// Recall calculates the percentage of actual positives (True Positives) that the classification 
// model correctly identified.
// binary: tp/(tp+fp)
// multiclass: average tp0/(tp0+fn0), tp1/(tp1+fn1),..., tpn/(tpn+fnn)
// multiclass (micro): (tp0+tp1+...+tpn) / (tp0+...+tpn+fn0+...+fnn)
template <typename Type=int>
double recall(const confmat<Type>& cm)
{
	if (cm.mode==CM_Binary) {
		// binary mode
		return (double)cm.m(0,0)/(cm.m(0,0)+cm.m(0,1));
	} else if (cm.mode==CM_Macro) {
		// multiclass (macro mode)
		double sum= 0.0;
		int n= cm.l.len();
		for (int i=0; i<n; ++i) {
			double tp= cm.m(i,i);
			double tpfn= 0.0;
			for (int k=0; k<n; k++) tpfn += cm.m(i,k);
			if (tpfn) sum += tp/tpfn;
		}
		return sum/n;
	} else {
		// multiclass (micro mode)
		double tp= 0.0;
		for (int i=0; i<cm.l.len(); ++i) tp += cm.m(i,i);
		double fn= 0.0;
		for (int i=0; i<cm.l.len()-1; ++i) {
			for (int j=i+1; j<cm.l.len(); ++j) {
				fn += cm.m(i,j);
			}
		}
		return tp/(tp+fn);
	}
}


// F1-score employs both Precision and Recall to determine the actual power of the model.
template <typename Type=int>
double F1(const confmat<Type>& cm)
{
	double p = precision<Type>(cm);
	double r = recall<Type>(cm);
	return 2.0*(p*r)/(p+r);
}


// Mean Squared Errorr (MSE) is a popular error metric for regression problems.
// MSE is calculated as the mean or average of the squared differences between 
// predicted and expected target values
// MSE = 1/N * sum(y_i – ypred_i)^2
template <typename Type>
double mean_squared_error(const uvec<Type>& y_real, const uvec<Type>& y_pred) 
{
	return (double)y_real.distance_squared(y_pred) / y_real.len();
}

template <typename Type>
double mean_squared_error(const umat<Type>& Y_real, const umat<Type>& Y_pred) 
{
	return (double)Y_real.distance_squared(Y_pred) / Y_real.len();
}


// The Root Mean Squared Error (RMSE), is an extension of the mean squared error.
// The RMSE can be calculated as: RMSE = sqrt(MSE)
template <typename Type>
double root_mean_squared_error(const uvec<Type>& y_real, const uvec<Type>& y_pred) 
{
	return std::sqrt(mean_squared_error(y_real, y_pred));
}

template <typename Type>
double root_mean_squared_error(const umat<Type>& Y_real, const umat<Type>& Y_pred) 
{
	return std::sqrt(mean_squared_error(Y_real, Y_pred));
}


// Mean Absolute Error (MAE) is a popular metric.
// Unlike the RMSE, the changes in MAE are linear and therefore intuitive.
// MAE score is calculated as the average of the absolute error values: 
// MAE = 1/N * sum(abs(y_i – ypred_i))
template <typename Type>
double mean_absolute_error(const uvec<Type>& y_real, const uvec<Type>& y_pred) 
{
	return (double)y_real.manhattan_distance(y_pred) / y_real.len();
}

template <typename Type>
double mean_absolute_error(const umat<Type>& Y_real, const umat<Type>& Y_pred) 
{
	return (double)Y_real.manhattan_distance(Y_pred) / Y_real.len();
}


// The silhouette coefficient is used to evaluate the quality of clusters formed  by a 
// clustering algorithm. It measures how similar an object is to its own cluster (cohesion) 
// compared to other clusters (separation).
template <typename Type>
double silhouette_coefficient(const umat<Type>& X, const umat<Type>& centroids) 
{
	assert(X.dev()==centroids.dev());
	uvec<int> cluster_labels(X.ydim(), X.dev());
	
	// assign each point to the nearest centroid
	for (int i=0; i<X.ydim(); ++i) {
		double min_distance = X.row(i).distance_squared(centroids.row(0));
		cluster_labels(i) = 0;
		for (int j=1; j<centroids.ydim(); ++j) {
			double distance = X.row(i).distance_squared(centroids.row(j));
			if (distance < min_distance) {
				min_distance = distance;
				cluster_labels(i) = j;
			}
		}
	}
	
	// compute cohesion (a) and separation (b) for each point
	uvec<double> a(X.ydim());
	uvec<double> b(X.ydim());
	for (int i=0; i<X.ydim(); ++i) {
		int cluster = cluster_labels(i);

		// cohesion
		double sum_a = 0.0;
		int count = 0;
		for (int j=0; j<X.ydim(); ++j) {
			if (cluster_labels(j) == cluster && j != i) {
				sum_a += std::sqrt(X.row(i).distance_squared(X.row(j)));
				count++;
			}
		}
		a(i) = (count > 0) ? sum_a/count : 0.0;
		
		// separation
		double min_b = std::numeric_limits<double>::max();
		for (int j=0; j<centroids.ydim(); ++j) {
			if (j != cluster) {
				double sum_b = 0.0;
				int count_b = 0;
				for (int k=0; k<X.ydim(); ++k) {
					if (cluster_labels(k) == j) {
						sum_b += std::sqrt(X.row(i).distance_squared(X.row(k)));
						count_b++;
					}
				}
				double avg_b = (count_b > 0) ? sum_b/count_b : 0.0;
				if (avg_b < min_b) min_b = avg_b;
			}
		}
		b(i) = min_b;
	}
	
	// calculate silhouette coefficient for each point
	double silhouette_sum = 0.0;
	for (int i=0; i<X.ydim(); ++i) {
		double s = (b(i)-a(i)) / std::max(a(i),b(i));
		silhouette_sum += s;
	}
	
	// average silhouette coefficient
	return silhouette_sum / X.ydim();
}


};     // namespace umml

#endif // UMML_METRICS_INCLUDE
