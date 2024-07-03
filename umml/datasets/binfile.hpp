#ifndef UMML_BINFILE_INCLUDED
#define UMML_BINFILE_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Generic Binary file loader/writer

 FILE:   binfile.hpp
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
 Loads a matrix of H rows and W cols stored using datatype DT

 Usage
 ~~~~~ 
 Binfile<double> bin;
 bin.save(filename, X);                            // save X to file 
 bin.load(filename, X, 1000, 748);                 // file has plain data, no header
 bin.load(filename, X, 1000, 748, sizeof(int)*2);  // file has a header of two 32 bit ints
 if (!bin) std::cout << bin.error_description() << "\n";
*/

#include "../umat.hpp"

#include <iomanip>
#include <fstream>
#include <iostream>


namespace umml {


template <typename DT=double>
class Binfile 
{
 public:
	// error codes
	enum {
		OK = 0,
		NotExists,
		BadDevice,
	};
	
	int err_id;
	
	// constructor
	Binfile(): err_id(OK) {}

	// checks if no errors occured during loading of Posters dataset files
	explicit operator bool() const { return err_id==OK; }

	// returns true if file is saved ok, otherwise returns false and sets err_id
	template <typename XT>
	bool save(const std::string& filename, const umat<XT>& X);

	// returns true if file is loaded ok, otherwise returns false and sets err_id
	template <typename XT>
	bool load(const std::string& filename, umat<XT>& X, int H, int W, size_t hdr_size=0);
	
	// returns a std::string with the description of the error in err_id
	std::string error_description() const;
};


template <typename DT>
template <typename XT>
bool Binfile<DT>::save(const std::string& filename, const umat<XT>& X)
{
	// assert CPU memory
	if (X.dev()!=device::CPU) { err_id = BadDevice; return false; }

	std::ofstream os;
	os.open(filename, std::ios::out|std::ios::binary);
	if (!os.is_open()) { err_id = NotExists; return false; }
	
	for (int i=0; i<X.ydim(); ++i)
	for (int j=0; j<X.xdim(); ++j) {
		DT dtval = static_cast<DT>(X(i,j));
		os.write((const char*)&dtval, sizeof(DT));
	}
	
	os.close();	
	return true;
}


template <typename DT>
template <typename XT>
bool Binfile<DT>::load(const std::string& filename, umat<XT>& X, int H, int W, size_t hdr_size)
{
	// assert CPU memory
	if (X.dev()!=device::CPU) { err_id = BadDevice; return false; }
	
	std::ifstream is;
	
	is.open(filename, std::ios::in|std::ios::binary);
	if (!is.is_open()) { err_id = NotExists; return false; }
	
	is.ignore(hdr_size);
	
	// allocate matrix X 
	X.resize(H, W);

	// read data
	for (int i=0; i<H; ++i)
	for (int j=0; j<W; ++j) {
		DT dtval;
		is.read((char*)&dtval, sizeof(DT));
		X(i,j) = static_cast<XT>(dtval);
	}

	is.close();	
	return true;
}


template <typename DT>
std::string Binfile<DT>::error_description() const 
{
	switch (err_id) {
		case NotExists: return "File not exists.";
		case BadDevice: return "Binfile works only with CPU memory.";
	};
	return "";
}


};     // namespace umml

#endif // UMML_BINFILE_INCLUDED
