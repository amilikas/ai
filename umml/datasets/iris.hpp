#ifndef UMML_IRIS_INCLUDED
#define UMML_IRIS_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 FILE:   iris.hpp
 AUTHOR: Anastasios Milikas (amilikas@csd.auth.gr)
 
 Namespace
 ~~~~~~~~~
 umml
 
 Requirements
 ~~~~~~~~~~~~
 umml vector
 umml matrix
  
 Description
 ~~~~~~~~~~~
 Iris Plants Dataset
 https://archive.ics.uci.edu/ml/datasets/iris
 
 Features (columns):
 1. sepal length in cm
 2. sepal width in cm
 3. petal length in cm
 4. petal width in cm
 5. class (0, 1 or 2)
   
 Class: 
 0 -- Iris Setosa (50)
 1 -- Iris Versicolour (50)
 2 -- Iris Virginica (50)
  
 Total samples:  150 
 Missing values: 0
*/

#include "../umat.hpp"

#include <vector>
#include <algorithm>


namespace umml {

	
/// load_iris
/// loads the iris data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void load_iris(umat<XT>& X, uvec<YT>& y)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);
	
	std::vector<XT> iris_data = {
	5.1,3.5,1.4,0.2,0,
	4.9,3,1.4,0.2,0,
	4.7,3.2,1.3,0.2,0,
	4.6,3.1,1.5,0.2,0,
	5,3.6,1.4,0.2,0,
	5.4,3.9,1.7,0.4,0,
	4.6,3.4,1.4,0.3,0,
	5,3.4,1.5,0.2,0,
	4.4,2.9,1.4,0.2,0,
	4.9,3.1,1.5,0.1,0,
	5.4,3.7,1.5,0.2,0,
	4.8,3.4,1.6,0.2,0,
	4.8,3,1.4,0.1,0,
	4.3,3,1.1,0.1,0,
	5.8,4,1.2,0.2,0,
	5.7,4.4,1.5,0.4,0,
	5.4,3.9,1.3,0.4,0,
	5.1,3.5,1.4,0.3,0,
	5.7,3.8,1.7,0.3,0,
	5.1,3.8,1.5,0.3,0,
	5.4,3.4,1.7,0.2,0,
	5.1,3.7,1.5,0.4,0,
	4.6,3.6,1,0.2,0,
	5.1,3.3,1.7,0.5,0,
	4.8,3.4,1.9,0.2,0,
	5,3,1.6,0.2,0,
	5,3.4,1.6,0.4,0,
	5.2,3.5,1.5,0.2,0,
	5.2,3.4,1.4,0.2,0,
	4.7,3.2,1.6,0.2,0,
	4.8,3.1,1.6,0.2,0,
	5.4,3.4,1.5,0.4,0,
	5.2,4.1,1.5,0.1,0,
	5.5,4.2,1.4,0.2,0,
	4.9,3.1,1.5,0.1,0,
	5,3.2,1.2,0.2,0,
	5.5,3.5,1.3,0.2,0,
	4.9,3.1,1.5,0.1,0,
	4.4,3,1.3,0.2,0,
	5.1,3.4,1.5,0.2,0,
	5,3.5,1.3,0.3,0,
	4.5,2.3,1.3,0.3,0,
	4.4,3.2,1.3,0.2,0,
	5,3.5,1.6,0.6,0,
	5.1,3.8,1.9,0.4,0,
	4.8,3,1.4,0.3,0,
	5.1,3.8,1.6,0.2,0,
	4.6,3.2,1.4,0.2,0,
	5.3,3.7,1.5,0.2,0,
	5,3.3,1.4,0.2,0,
	7,3.2,4.7,1.4,1,
	6.4,3.2,4.5,1.5,1,
	6.9,3.1,4.9,1.5,1,
	5.5,2.3,4,1.3,1,
	6.5,2.8,4.6,1.5,1,
	5.7,2.8,4.5,1.3,1,
	6.3,3.3,4.7,1.6,1,
	4.9,2.4,3.3,1,1,
	6.6,2.9,4.6,1.3,1,
	5.2,2.7,3.9,1.4,1,
	5,2,3.5,1,1,
	5.9,3,4.2,1.5,1,
	6,2.2,4,1,1,
	6.1,2.9,4.7,1.4,1,
	5.6,2.9,3.6,1.3,1,
	6.7,3.1,4.4,1.4,1,
	5.6,3,4.5,1.5,1,
	5.8,2.7,4.1,1,1,
	6.2,2.2,4.5,1.5,1,
	5.6,2.5,3.9,1.1,1,
	5.9,3.2,4.8,1.8,1,
	6.1,2.8,4,1.3,1,
	6.3,2.5,4.9,1.5,1,
	6.1,2.8,4.7,1.2,1,
	6.4,2.9,4.3,1.3,1,
	6.6,3,4.4,1.4,1,
	6.8,2.8,4.8,1.4,1,
	6.7,3,5,1.7,1,
	6,2.9,4.5,1.5,1,
	5.7,2.6,3.5,1,1,
	5.5,2.4,3.8,1.1,1,
	5.5,2.4,3.7,1,1,
	5.8,2.7,3.9,1.2,1,
	6,2.7,5.1,1.6,1,
	5.4,3,4.5,1.5,1,
	6,3.4,4.5,1.6,1,
	6.7,3.1,4.7,1.5,1,
	6.3,2.3,4.4,1.3,1,
	5.6,3,4.1,1.3,1,
	5.5,2.5,4,1.3,1,
	5.5,2.6,4.4,1.2,1,
	6.1,3,4.6,1.4,1,
	5.8,2.6,4,1.2,1,
	5,2.3,3.3,1,1,
	5.6,2.7,4.2,1.3,1,
	5.7,3,4.2,1.2,1,
	5.7,2.9,4.2,1.3,1,
	6.2,2.9,4.3,1.3,1,
	5.1,2.5,3,1.1,1,
	5.7,2.8,4.1,1.3,1,
	6.3,3.3,6,2.5,2,
	5.8,2.7,5.1,1.9,2,
	7.1,3,5.9,2.1,2,
	6.3,2.9,5.6,1.8,2,
	6.5,3,5.8,2.2,2,
	7.6,3,6.6,2.1,2,
	4.9,2.5,4.5,1.7,2,
	7.3,2.9,6.3,1.8,2,
	6.7,2.5,5.8,1.8,2,
	7.2,3.6,6.1,2.5,2,
	6.5,3.2,5.1,2,2,
	6.4,2.7,5.3,1.9,2,
	6.8,3,5.5,2.1,2,
	5.7,2.5,5,2,2,
	5.8,2.8,5.1,2.4,2,
	6.4,3.2,5.3,2.3,2,
	6.5,3,5.5,1.8,2,
	7.7,3.8,6.7,2.2,2,
	7.7,2.6,6.9,2.3,2,
	6,2.2,5,1.5,2,
	6.9,3.2,5.7,2.3,2,
	5.6,2.8,4.9,2,2,
	7.7,2.8,6.7,2,2,
	6.3,2.7,4.9,1.8,2,
	6.7,3.3,5.7,2.1,2,
	7.2,3.2,6,1.8,2,
	6.2,2.8,4.8,1.8,2,
	6.1,3,4.9,1.8,2,
	6.4,2.8,5.6,2.1,2,
	7.2,3,5.8,1.6,2,
	7.4,2.8,6.1,1.9,2,
	7.9,3.8,6.4,2,2,
	6.4,2.8,5.6,2.2,2,
	6.3,2.8,5.1,1.5,2,
	6.1,2.6,5.6,1.4,2,
	7.7,3,6.1,2.3,2,
	6.3,3.4,5.6,2.4,2,
	6.4,3.1,5.5,1.8,2,
	6,3,4.8,1.8,2,
	6.9,3.1,5.4,2.1,2,
	6.7,3.1,5.6,2.4,2,
	6.9,3.1,5.1,2.3,2,
	5.8,2.7,5.1,1.9,2,
	6.8,3.2,5.9,2.3,2,
	6.7,3.3,5.7,2.5,2,
	6.7,3,5.2,2.3,2,
	6.3,2.5,5,1.9,2,
	6.5,3,5.2,2,2,
	6.2,3.4,5.4,2.3,2,
	5.9,3,5.1,1.8,2
	};
  	std::vector<XT> features;
    std::vector<YT> labels;	
	
	int n = 0;
    for (size_t i=0; i<iris_data.size(); i += 5) {
    	for (int j=0; j<4; ++j) features.push_back(iris_data[i+j]);
        labels.push_back(static_cast<YT>(iris_data[i+4]));
        n++;
    }	
	X.resize(n, 4);
	X.set(features);
	y.resize(n); 
	y.set(labels);
}


};     // namespace umml

#endif // UMML_IRIS_INCLUDED
