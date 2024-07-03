#ifndef UMML_TINYSETS_INCLUDED
#define UMML_TINYSETS_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 FILE:   tinysets.hpp
 AUTHOR: Anastasios Milikas (amilikas@csd.auth.gr)
 
 Namespace
 ~~~~~~~~~
 mml
 
 Requirements
 ~~~~~~~~~~~~
 mml matrix
 STL string
  
 Description
 ~~~~~~~~~~~
 Tiny datasets appropriate for classification problems:
 
 * logical or dataset
 * logical xor dataset
 * a linear separable tiny dataset (big margin)
 * a linear separable tiny dataset (tiny margin)
 * moons tiny dataset
 * circles tiny dataset
 * binary dataset (bishop +1 and knight -1 moves in a 4x4 board)
*/

#include "../umat.hpp"

#include <string>


namespace umml {
	

template <typename XT, typename YT>
void __tinysets_create_dataset(const std::string& _txt, umat<XT>& X, uvec<YT>& y);


/// or_dataset
/// loads the OR data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void or_dataset(umat<XT>& X, uvec<YT>& y)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);
	
	std::vector<XT> flat = {1.0,1.0 , 0.0,1.0 , 1.0,0.0 , 0.0,0.0};
	std::vector<YT> labels = {1, 1, 1, 0};
	X.resize(4, 2);
	X.set(&flat[0]);
	y.resize(4);
	y.set(&labels[0]);
	
}


/// xor_dataset
/// loads the XOR data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void xor_dataset(umat<XT>& X, uvec<YT>& y)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);
	
	std::vector<XT> flat = {1.0,1.0 , 0.0,1.0 , 1.0,0.0 , 0.0,0.0};
	std::vector<YT> labels = {0, 1, 1, 0};
	X.resize(4, 2);
	X.set(&flat[0]);
	y.resize(4);
	y.set(&labels[0]);
}


/// tiny_linear: big margin separable
/// loads the data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void tiny_linear(umat<XT>& X, uvec<YT>& y)
{
	const std::string _txt = 
	"0                            \n"
	"0  0    0                    \n" 
	" 0   0                       \n"
	"0   0   0                    \n"
	"                             \n"
	"                             \n"
	"                             \n"        
	"                  1   1      \n"
	"               1    1    1  1\n"
	"                  1          \n"
	"                      1    1 \n";
	__tinysets_create_dataset<XT,YT>(_txt, X, y);
}


/// tiny_linear_sm: small margin separable
/// loads the data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void tiny_linear_sm(umat<XT>& X, uvec<YT>& y)
{
	const std::string _txt = 
	"0                            \n"
	"0  0    0                    \n" 
	" 0   0                       \n"
	"0   0   0                    \n"
	"  0   0                      \n"
	"      0 0  0                 \n"
	"         1                   \n"        
	"                  1   1      \n"
	"               1    1    1  1\n"
	"             1 1  1    1     \n"
	"                    1 1    1 \n";
	__tinysets_create_dataset<XT,YT>(_txt, X, y);
}


/// tiny_moons
/// loads the data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void tiny_moons(umat<XT>& X, uvec<YT>& y)
{
	const std::string _txt = 
	"       0                \n"
	"     0   0              \n"
	"   0        0           \n"
	"   0    1              1\n"
	"            0           \n"
	"  0     1             1 \n"
	"               0        \n"
	"          1         1   \n"
	"            1           \n"
	"              1  1      \n";
	__tinysets_create_dataset<XT,YT>(_txt, X, y);
}


/// tiny_circles
/// loads the data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void tiny_circles(umat<XT>& X, uvec<YT>& y)
{
	const std::string _txt = 
	"         0   0        \n"
	"      0         0     \n"
	"   0               0  \n"
	"            1  1      \n"
	"  0      1  1  1     0\n"
	"        1  1  1 1     \n"
	"    0    1  1 1     0 \n"
	"     0     1 1        \n"
	"                  0   \n"
	"        0   0  0      \n";
	__tinysets_create_dataset<XT,YT>(_txt, X, y);
}


/// tiny_xorlike
/// loads the data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void tiny_xorlike(umat<XT>& X, uvec<YT>& y)
{
	const std::string _txt = 
	"1                 0         \n"
	"1  1    1           0   0  0\n"
	" 1   1          0    0      \n"
	"1   1   1          0  0   0 \n"
	"                            \n"
	"                            \n"
	"0   0  0                    \n"
	"   0              1   1     \n"
	" 0 0  0  0     1    1   1  1\n"
	"                  1         \n"
	"    0                 1    1\n";
	__tinysets_create_dataset<XT,YT>(_txt, X, y);
}

template <typename XT, typename YT>
void tiny_binary(umat<XT>& X, uvec<YT>& y)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);
	
	std::vector<XT> in = {
	// 4x4 board, bishop moves
	0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
	0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
	0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
	0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
	0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	// 4x4 board, knight moves
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
	0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
	0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
	0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
	0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
	0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
	0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	};
	
	std::vector<YT> out = {  
	//bishop
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	// knight
	-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 
	};
	
	X.resize(24,32);
	X.set(&in[0]);
	y.resize(24);
	y.set(&out[0]);
}

template <typename XT, typename YT>
inline void __tinysets_create_dataset(const std::string& _txt, umat<XT>& X, uvec<YT>& y)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);
	
	std::vector<XT> flat;
	std::vector<YT> labels;
	int i=0, j=0;
	for (std::string::const_iterator it=_txt.begin(); it!=_txt.end(); ++it) {
		if (*it!=' ' && *it!='\n') {
			flat.push_back(static_cast<XT>(j));
			flat.push_back(static_cast<XT>(i));
			labels.push_back(static_cast<YT>(int(*it-'0')));
		}
		j++;
		if (*it=='\n') {
			i++;
			j = 0; 
		}
	}
	int ndata = (int)labels.size();
	X.resize(ndata, 2);
	X.set(&flat[0]);
	y.resize(ndata);
	y.set(&labels[0]);
}


};     // namespace umml

#endif // UMML_TINYSETS_INCLUDED
