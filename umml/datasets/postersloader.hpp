#ifndef UMML_POSTERSLOADER_INCLUDED
#define UMML_POSTERSLOADER_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 IMDB movie posters dataset loader

 FILE:   postersloader.hpp
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
 IMAGES FILE
 [offset] [type]           [description]
 0000     32 bit integer   number of images (MSB first)
 0004     32 bit integer   height of the images (MSB first)
 0008     32 bit integer   width of the images (MSB first)
 
 0012     32 bit integer   imdb id of the image (MSB first)
 0016     unsigned bytes   red channel, green channel, blue channel
 ....

 LABELS FILE
 [offset] [type]           [description]
 0000     32 bit integer   number of images (MSB first)
 0004     32 bit integer   number of genres (MSB first)
 0008     unsigned bytes   genres (on hot encoding)
 ....

 Usage
 ~~~~~ 
 Postersloader<float> posters;
 posters.load_images(train_images, X_train);
 posters.load_labels(train_labels, y_train);
 if (!posters) std::cout << posters.error_description() << "\n";
*/

#include "../umat.hpp"

#include <iomanip>
#include <fstream>
#include <iostream>


namespace umml {


class Postersloader 
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
	Postersloader(): err_id(OK) {}

	// checks if no errors occured during loading of Posters dataset files
	explicit operator bool() const { return err_id==OK; }

	// images loader
	// returns true if file is loaded ok, otherwise returns false and sets err_id
	template <typename XT=float>
	bool load_images(const std::string& filename, umat<XT>& X);
	
	// labels loader
	// returns true if file is loaded ok, otherwise returns false and sets err_id
	template <typename YT=int>
	bool load_labels(const std::string& filename, umat<YT>& Y);
	 
	// returns a std::string with the description of the error in err_id
	std::string error_description() const;
 
 private:
	static uint32_t msb2lsb(uint32_t i32);
};


uint32_t Postersloader::msb2lsb(uint32_t i32)
{
	unsigned char inByte0 = (i32 & 0xFF);
	unsigned char inByte1 = (i32 & 0xFF00) >> 8;
	unsigned char inByte2 = (i32 & 0xFF0000) >> 16;
	unsigned char inByte3 = (i32 & 0xFF000000) >> 24;
	return (inByte0 << 24) | (inByte1 << 16) | (inByte2 << 8) | (inByte3);
}


template <typename XT>
bool Postersloader::load_images(const std::string& filename, umat<XT>& X)
{
	bool ok=true;
	
	// assert CPU memory
	if (X.dev()!=device::CPU) { ok = false; err_id = BadDevice; goto ret; }
	
	unsigned char* buff = nullptr;
	uint32_t i32;
	std::ifstream is;
	int nrows, ncols;
	size_t xdim=0, ydim=0;
	
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { ok = false; err_id = NotExists; goto ret; }
	
	// read header and check for errors
	is.read((char*)&i32, sizeof(i32));
	nrows = msb2lsb(i32);
	is.read((char*)&i32, sizeof(i32));
	ydim = (size_t)msb2lsb(i32);
	is.read((char*)&i32, sizeof(i32));
	xdim = (size_t)msb2lsb(i32);
	ncols = (int)(ydim*xdim*3);
	
	std::cout << "Images: " << nrows << " images " << ydim << "x" << xdim << "\n";

	// allocate matrix X 
	X.resize(nrows, ncols);
	buff = new unsigned char [ncols];
	if (!buff) { ok = false; err_id = BadAlloc; goto ret; }

	// read pixel data
	int i;
	for (i=0; i<nrows; i++) {
		is.read((char*)&i32, sizeof(i32)); // imdb id
		is.read((char*)buff, ncols);       // red,green and blue channels
		if (!is) break;
		for (int j=0; j<ncols; j++) X(i,j) = static_cast<XT>(buff[j]);
	}
	if (i != nrows) { ok = false; err_id = BadCount; goto ret; }

ret:
	if (buff) delete buff;
	is.close();	
	return ok;
}


template <typename YT>
bool Postersloader::load_labels(const std::string& filename, umat<YT>& Y)
{
	bool ok=true;

	// assert CPU memory
	if (Y.dev()!=device::CPU) { ok = false; err_id = BadDevice; goto ret; }
	
	unsigned char ch[14];
	uint32_t i32;
	std::ifstream is;
	int nitems, ndims;
	
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { ok = false; err_id = NotExists; goto ret; }
	
	// read header
	is.read((char*)&i32, sizeof(i32));
	nitems = (int)msb2lsb(i32);
	is.read((char*)&i32, sizeof(i32));
	ndims = (int)msb2lsb(i32);
	if (ndims > 14) { ok = false; err_id = BadDimensions; goto ret; }
	
	std::cout << "labels: " << nitems << " with " << ndims << " features.\n";

	// allocate Y 
	Y.resize(nitems, ndims);
	
	// read labels
	int i;
	for (i=0; i<nitems; i++) {
		is.read((char*)ch, ndims);
		if (!is) break;
		for (int j=0; j<ndims; ++j) Y(i,j) = static_cast<YT>(ch[j]);
	}
	if (i != nitems) { ok = false; err_id = BadCount; goto ret; }

ret:
	is.close();	
	return ok;
}


std::string Postersloader::error_description() const 
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

#endif // UMML_POSTERSLOADER_INCLUDED
