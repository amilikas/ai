#ifndef UMML_MNISTLOADER_INCLUDED
#define UMML_MNISTLOADER_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 MNIST digits binary file format loader functions

 FILE:   mnistloader.hpp
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
 IMAGE FILE
 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000803(2051) magic number (MSB first)
 0004     32 bit integer  60000            number of images   (10000 for test file)
 0008     32 bit integer  28               number of rows
 0012     32 bit integer  28               number of columns
 0016     unsigned byte   ??               pixel
 0017     unsigned byte   ??               pixel
 ........
 xxxx     unsigned byte   ??               pixel

 LABEL FILE
 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 0004     32 bit integer  60000            number of items   (10000 for test file)
 0008     unsigned byte   ??               label
 0009     unsigned byte   ??               label
 ........
 xxxx     unsigned byte   ??               label
 The labels values are 0 to 9.

 Usage
 ~~~~~ 
 MNISTloader<float> mnist;
 mnist.load_images(train_images, X_train);
 mnist.load_labels(train_labels, y_train);
 if (!mnist) std::cout << mnist.error_description() << "\n";
*/

#include "../umat.hpp"

#include <iomanip>
#include <fstream>
#include <iostream>


namespace umml {


class MNISTloader 
{
 public:
	// error codes
	enum {
		OK = 0,
		NotExists,
		BadMagic,
		BadDimensions,
		BadCount,
		BadAlloc,
		BadDevice,
	};
	
	int err_id;
	
	// constructor
	MNISTloader(): err_id(OK) {}

	// checks if no errors occured during loading of MNIST files
	explicit operator bool() const { return err_id==OK; }

	// images loader
	// returns true if file is loaded ok, otherwise returns false and sets err_id
	template <typename XT=float>
	bool load_images(const std::string& filename, umat<XT>& X);

	// labels loader
	// returns true if file is loaded ok, otherwise returns false and sets err_id
	template <typename YT=int>
	bool load_labels(const std::string& filename, uvec<YT>& y);

	// images writer
	// returns true if file is loaded ok, otherwise returns false and sets err_id
	template <typename XT=float>
	bool save_images(const std::string& filename, umat<XT>& X);

	// labels writer
	// returns true if file is loaded ok, otherwise returns false and sets err_id
	template <typename YT=int>
	bool save_labels(const std::string& filename, uvec<YT>& y);
	
	// returns a std::string with the description of the error in err_id
	std::string error_description() const;
 
 private:
	static uint32_t msb2lsb(uint32_t i32);
	static uint32_t lsb2msb(uint32_t i32);
};


uint32_t MNISTloader::msb2lsb(uint32_t i32)
{
	unsigned char inByte0 = (i32 & 0xff);
	unsigned char inByte1 = (i32 & 0xff00) >> 8;
	unsigned char inByte2 = (i32 & 0xff0000) >> 16;
	unsigned char inByte3 = (i32 & 0xff000000) >> 24;
	return (inByte0 << 24) | (inByte1 << 16) | (inByte2 << 8) | (inByte3);
}

uint32_t MNISTloader::lsb2msb(uint32_t i32)
{
	i32 = (((i32 & 0xaaaaaaaa) >> 1) | ((i32 & 0x55555555) << 1));
	i32 = (((i32 & 0xcccccccc) >> 2) | ((i32 & 0x33333333) << 2));
	i32 = (((i32 & 0xf0f0f0f0) >> 4) | ((i32 & 0x0f0f0f0f) << 4));
	i32 = (((i32 & 0xff00ff00) >> 8) | ((i32 & 0x00ff00ff) << 8));
	return (i32 >> 16) | (i32 << 16);
}


template <typename XT>
bool MNISTloader::load_images(const std::string& filename, umat<XT>& X)
{
	unsigned char buff[784];
	uint32_t i32;
	std::ifstream is;
	int nrows, ncols;
	size_t xdim=0, ydim=0;
	bool ok=true;

	// assert CPU memory
	if (X.dev()!=device::CPU) { ok = false; err_id = BadDevice; goto ret; }
	
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { ok = false; err_id = NotExists; goto ret; }
	
	// read header and check for errors
	is.read((char*)&i32, sizeof(i32));
	if (msb2lsb(i32) != 2051) { ok = false; err_id = BadMagic; goto ret; }
	is.read((char*)&i32, sizeof(i32));
	nrows = msb2lsb(i32);
	is.read((char*)&i32, sizeof(i32));
	ydim = (size_t)msb2lsb(i32);
	if (ydim > 28) { ok = false; err_id = BadDimensions; goto ret; }
	is.read((char*)&i32, sizeof(i32));
	xdim = (size_t)msb2lsb(i32);
	if (xdim > 28) { ok = false; err_id = BadDimensions; goto ret; }
	ncols = (int)(ydim*xdim);

	// allocate matrix X 
	X.resize(nrows, ncols);

	// read pixel data
	int i;
	for (i=0; i<nrows; ++i) {
		is.read((char*)buff, ncols);
		if (!is) break;
		for (int j=0; j<ncols; ++j) X(i,j) = static_cast<XT>(buff[j]);
	}
	if (i != nrows) { ok = false; err_id = BadCount; goto ret; }

ret:
	is.close();	
	return ok;
}


template <typename YT>
bool MNISTloader::load_labels(const std::string& filename, uvec<YT>& y)
{
	unsigned char ch;
	uint32_t i32;
	std::ifstream is;
	int nitems;
	bool ok=true;

	// assert CPU memory
	if (y.dev()!=device::CPU) { ok = false; err_id = BadDevice; goto ret; }
	
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { ok = false; err_id = NotExists; goto ret; }
	
	// read header
	is.read((char*)&i32, sizeof(i32));
	if (msb2lsb(i32) != 2049) { ok = false; err_id = BadMagic; goto ret; }
	is.read((char*)&i32, sizeof(i32));
	nitems = (int)msb2lsb(i32);
	
	// allocate vector y 
	y.resize(nitems);
	
	// read labels
	int i;
	for (i=0; i<nitems; ++i) {
		is.read((char*)&ch, 1);
		if (!is) break;
		y(i) = static_cast<YT>(ch);
	}
	if (i != nitems) { ok = false; err_id = BadCount; goto ret; }

ret:
	is.close();	
	return ok;
}


template <typename XT>
bool MNISTloader::save_images(const std::string& filename, umat<XT>& X)
{
	unsigned char buff[784];
	uint32_t i32;
	std::ofstream os;
	bool ok=true;

	// assert CPU memory
	if (X.dev()!=device::CPU) { ok = false; err_id = BadDevice; goto ret; }
	
	os.open(filename, std::ios::out|std::ios::binary);
	if (!os.is_open()) { ok = false; err_id = NotExists; goto ret; }
	
	// write header
	i32 = msb2lsb(2051);
	os.write((char*)&i32, sizeof(i32));
	i32 = msb2lsb(X.ydim());
	os.write((char*)&i32, sizeof(i32));
	i32 = msb2lsb(28);
	os.write((char*)&i32, sizeof(i32));
	os.write((char*)&i32, sizeof(i32));
	// write pixel data
	int i;
	for (i=0; i<X.ydim(); ++i) {
		for (int j=0; j<X.xdim(); ++j) buff[j] = static_cast<unsigned char>(X(i,j));
		os.write((char*)buff, X.xdim());
		if (!os) break;
	}

ret:
	os.close();	
	return ok;
}


template <typename YT>
bool MNISTloader::save_labels(const std::string& filename, uvec<YT>& y)
{
	unsigned char ch;
	uint32_t i32;
	std::ofstream os;
	bool ok=true;

	// assert CPU memory
	if (y.dev()!=device::CPU) { ok = false; err_id = BadDevice; goto ret; }
	
	os.open(filename, std::ios::out|std::ios::binary);
	if (!os.is_open()) { ok = false; err_id = NotExists; goto ret; }
	
	// write header
	i32 = msb2lsb(2049);
	os.write((char*)&i32, sizeof(i32));
	i32 = msb2lsb(y.len());
	os.write((char*)&i32, sizeof(i32));
	
	// write labels
	int i;
	for (i=0; i<y.len(); ++i) {
		ch = static_cast<unsigned char>(y(i));
		os.write((char*)&ch, 1);
		if (!os) break;
	}

ret:
	os.close();	
	return ok;
}


std::string MNISTloader::error_description() const 
{
	switch (err_id) {
		case NotExists: return "File not exists.";
		case BadMagic: return "File has unknown magic number.";
		case BadDimensions: return "Wrong dimensions.";
		case BadCount: return "Unexpected number of samples.";
		case BadAlloc: return "Memory allocation error.";
		case BadDevice: return "Loader works only with CPU memory.";
	};
	return "";
}


};     // namespace umml

#endif // UMML_MNISTLOADER_INCLUDED
