/*
 Machine Learning Artificial Intelligence Library

 FILE:     csvloader.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 STL vector
 STL string
 
 Internal dependencies:
 STL algorithm
 STL fstream
 umml algo
 
 Notes
 ~~~~~
 Supports comments in CSV files. Any line that starts with # is considered as a comment. 

 Usage
 ~~~~~
 csvloader csv(filename, separator);
 csv.load(X, nanvalue);
*/


#ifndef UMML_CSVLOADER_INCLUDED
#define UMML_CSVLOADER_INCLUDED

#include "../umat.hpp"
#include "../algo.hpp"

#include <fstream>
#include <iostream>


namespace umml {


// csvloader
class csvloader {
 public:
	// error codes
	enum {
		OK=0,
		NotExists=1,
		BadCount=2,
		BadAlloc=3,
	};
	
	int err_id;
	
	// constructor
	csvloader(const std::string fname, char sep=',', bool has_headers=false) {
		err_id    = OK;
		filename  = fname;
		separator = sep;
		has_hdrs  = has_headers;
	}

	// checks if no errors occured during CSV loading
	explicit operator bool() const { return err_id==OK; }

	// returns CSV's headers (if they present)
	const std::vector<std::string>& headers() const { return hdrs; }
	
	// loads the .csv file's data into the matrix 'X'
	// TODO: non numerical data will be encoded as (integer) labels
	template <typename Type>
	int load(umat<Type>& X, const Type& nanval=static_cast<Type>(std::nan(""))) {
		assert(X.dev()==device::CPU && "csvloader: matrix not in CPU memory");
		std::vector<std::vector<std::string>> rows;
		err_id = load_csv_data(rows);
		if (err_id != OK) return err_id;
		int nrows = (int)rows.size();
		int ncols = (int)rows[0].size();
		X.resize(nrows, ncols);
		typedef std::map<std::string, Type> strmap;
		std::vector<strmap> colmaps(ncols);
		std::vector<Type> colmap_cur(ncols, 0);
		for (int i=0; i<nrows; ++i) 
		for (int j=0; j<ncols; ++j) {
			if (!rows[i][j].empty()) {
				char* pEnd;
				double dval = std::strtod(rows[i][j].c_str(), &pEnd);
				if (pEnd==rows[i][j].c_str()) {
					// non numerical value, encode it to a label
					typename strmap::const_iterator it = colmaps[j].find(rows[i][j]);
					if (it==colmaps[j].end()) {
						dval = colmap_cur[j]++;
						colmaps[j].insert({ rows[i][j], dval });
					} else {
						dval = it->second;
					}
				}
				X(i,j) = static_cast<Type>(dval);
			} else {
				X(i,j) = nanval;
			}
		}
		return OK;
	}

	// returns a std::string with the description of the error in err_id
	std::string error_description() const {
		switch (err_id) {
			case NotExists: return "File not exists.";
			case BadCount: return "Wrong number of columns.";
			case BadAlloc: return "Memory allocation error.";
		};
		return "";
	}

 private:
	int load_csv_data(std::vector<std::vector<std::string>>& rows) {
		std::ifstream is(filename);
		std::string line;
		rows.clear();
		hdrs.clear();
		size_t nlines= 0;
		size_t ncols= 0;
		while (std::getline(is,line)) {
			if (line[0]=='#') continue;
			nlines++;
			std::stringstream ss(line);
			std::string cell;
			if (has_hdrs && nlines==1) {
				while (std::getline(ss,cell,separator)) hdrs.push_back(rtrim(ltrim(cell)));
				ncols = hdrs.size();
			} else {
				std::vector<std::string> row;
				if (ncols > 0) row.reserve(ncols);
				while (std::getline(ss,cell,separator)) row.push_back(cell);
				if (!row.empty()) {
					if (nlines==1) ncols = row.size();
					else if (row.size() != ncols) return BadCount;
					rows.push_back(row);
				}
			}
		}
		return OK;
	}
 	
 private:
	std::string filename;
	char separator;
	bool has_hdrs;
	std::vector<std::string> hdrs;
};


};     // namespace umml

#endif // UMML_CSVLOADER_INCLUDED
