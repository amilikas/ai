#ifndef UMML_CIFARLOADER_INCLUDED
#define UMML_CIFARLOADER_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 CIFAR-10 and CIFAR-100 images binary file format loader functions

 FILE:   cifar10loader.hpp
 AUTHOR: Anastasios Milikas (amilikas@csd.auth.gr)
 
 Namespace
 ~~~~~~~~~
 umml
 
 Requirements
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 STL string
 
 Internal requirements
 STL file streams
  
 Description
 ~~~~~~~~~~~
 FILE FORMAT
 [offset]  [type]          [description]
 0000      8 bit integer   image label (0..9) or (0..99)
 0001-1024 8 bit integers  red channel (32x32)
 1025-2048 8 bit integers  green channel (32x32)
 2049-3072 8 bit integers  blue channel (32x32)
 ........

 LABELS (CIFAR-10)
 0=airplane
 1=automobile
 2=bird
 3=cat
 4=deer
 5=dog
 6=frog
 7=horse
 8=ship
 9=truck
 
 Usage
 ~~~~~ 
 dataframe df;
 CIFARloader<dtype> cifar10;
 for (int i=1; i<=5; ++i) {
	umat<dtype> X_temp;
	uvec<int> y_temp;
	cifar10.load_images(train_file+to_string(i)+".bin", X_temp, y_temp);
	X_train = df.vstack(X_train, X_temp);
	y_train = df.append(y_train, y_temp);
 }
 cifar10.load_images(test_file, X_test, y_test);
 if (!cifar10) std::cout << cifar10.error_description() << "\n";
*/

#include "../umat.hpp"
#include "../dataframe.hpp"

#include <iomanip>
#include <fstream>
#include <iostream>


namespace umml {


class CIFARloader 
{
 public:
	int err_id;
	
	// error codes
	enum {
		OK = 0,
		NotExists,
		BadCount,
		BadAlloc,
		BadDevice,
	};
	
	// constructor
	CIFARloader(): err_id(OK) {}
	
	// images and labels loader
	// returns true if file is loaded ok, otherwise returns false and sets err_id
	template <typename XT=float, typename YT=int>
	bool load_images(const std::string& filename, umat<XT>& X, uvec<YT>& y);

	// checks if no errors occured during loading of CIFAR10 files
	explicit operator bool() const { return err_id==OK; }

	// returns a std::string with the description of the error in err_id
	std::string error_description() const;
};


template <typename XT, typename YT>
bool CIFARloader::load_images(const std::string& filename, umat<XT>& X, uvec<YT>& y)
{
	unsigned char buff[1+32*32*3];
	std::ifstream is;
	bool ok=true;

	// assert CPU memory
	if (X.dev()!=device::CPU || y.dev()!=device::CPU) { ok = false; err_id = BadDevice; goto ret; }
	
	// open file
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { ok = false; err_id = NotExists; goto ret; }

	// allocate matrix X (image data)  and vector y (labels)
	X.resize(10000, 32*32*3);
	y.resize(10000);

	// read labels and pixel data
	int i;
	for (i=0; i<10000; i++) {
		is.read((char*)buff,1+32*32*3);
		if (!is) break;
		y(i) = static_cast<YT>(buff[0]);
		for (int j=0; j<X.xdim(); j++) X(i,j) = static_cast<XT>(buff[1+j]);
	}
	if (i != 10000) { ok = false; err_id = BadCount; goto ret; }

ret:
	is.close();	
	return ok;
}

std::string CIFARloader::error_description() const 
{
	switch (err_id) {
		case NotExists: return "File not exists.";
		case BadCount: return "Unexpected number of samples.";
		case BadAlloc: return "Memory allocation error.";
		case BadDevice: return "Loader works only with CPU memory.";
	};
	return "";
}


};     // namespace umml

#endif // UMML_CIFARLOADER_INCLUDED
