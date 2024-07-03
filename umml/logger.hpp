#ifndef UMML_LOGGER_INCLUDED
#define UMML_LOGGER_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Logging

 FILE:     logger.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Internal dependencies:
 STL string
 STL file streams

 Notes
 ~~~~~
*/

#include <string>
#include <fstream>


namespace umml {


/*
 Logger
*/
class Logger
{
 public:
	Logger(): write_to_fstream(false), ok(true) {}
	Logger(const std::string& file_name): fstream(file_name), write_to_fstream(true), ok(fstream) {}
	
	explicit operator bool() const { return ok; }
      
	template <typename T>
	Logger& operator << (T&& t) {
		if (!ok) return *this;
		// write to cout
		if (!(std::cout << t)) {
			ok = false;
			return *this;
		}
		// write to file stream
		if (write_to_fstream) {
			if (!(fstream << t)) {
				ok = false;
				return *this;
			}
			fstream.flush();
		}
		return *this;
	}
	
 private:
	std::ofstream fstream;
	bool write_to_fstream;
	bool ok;
};


};     // namespace umml

#endif // UMML_LOGGER_INCLUDED
