#ifndef UMML_UTILS_INCLUDED
#define UMML_UTILS_INCLUDED

/*
 Functions
 ~~~~~~~~~
 * ltrim: left-trim whitespace characters from a std::string
 * rtrim: right-trim whitespace characters from a std::string
 * string_to_values: create a std::vector from the comma separated values in a std::string
*/

#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>

#ifdef UMML_POSIX_SUPPORT
#include <sys/stat.h>
#endif


namespace umml {


/*
 ltrim: left-trim whitespace characters from a std::string
*/  
std::string ltrim(const std::string& s) 
{
	const std::string WHITESPACE = " \t\n\r";
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start==std::string::npos) ? "" : s.substr(start);
}

/*
 rtrim: right-trim whitespace characters from a std::string
 note: there is a bug in gcc with null termination character (Uos 8.3.0.3-3+rebuild)
*/  
std::string rtrim(const std::string& s) 
{
	const std::string WHITESPACE = " \t\n\r";
    size_t end = s.find_last_not_of(WHITESPACE);
    while (s[end]=='\0' || s[end]==' ') --end;   // bug work-around
    return (end==std::string::npos) ? "" : s.substr(0, end+1);
}


/*
 create a std::vector from the comma separated values in a std::string 'str'
 it is commonly used for small vector and matrix initialization
*/
template <typename Type>
std::vector<Type> string_to_values(const std::string& str, char sep=',') 
{
	std::stringstream ss(str);
	std::string cell;
	std::vector<Type> vals;
	while (std::getline(ss,cell,sep)) {
		Type val;
		if (!cell.empty()) {
			char* pEnd;
			double dval = std::strtod(cell.c_str(), &pEnd);
			val = static_cast<Type>(dval);
		} else {
			val = static_cast<Type>(std::nan(""));
		}
		vals.push_back(val);
	}
	return vals;
}


void umml_assert(bool cond, const std::string& msg) 
{
	if (!cond) {
		std::cerr << "Fatal: " << msg << "\n";
		std::exit(-1);
	}
}

// max pooling output size
int pooling_size(int n, int k, int stride) {
	return (n-k)/stride + 1;
}

// padding size
int padding_size(int n, int k, int stride) {
	return ((stride-1)*n-stride+k) / 2;
} 

// convolution output size
int conv_size(int n, int k, int stride, int pad=0) { 
	return (n-k+2*pad)/stride + 1; 
}

//forward convolution patches2cols matrix dimensions
void conv2d_p2c_size(int c, int m, int n, int kh, int kw, int stride, int* rows, int* cols) 
{
	int h = conv_size(m, kh, stride);
	int w = conv_size(n, kw, stride);
	*rows = c*kh*kw;
	*cols = h*w;
}

// maybe this one NEEDS PAD????????}
void back2d_p2c_size(int c, int m, int n, int kh, int kw, int stride, int* rows, int* cols) 
{
	int h = conv_size(m, kh, stride);
	int w = conv_size(n, kw, stride);
	*rows = kh*kw;
	*cols = c*h*w;
}

// checks if a value is close to zero (within a tolerance)
template <typename Type>
bool close_to_zero(Type val, Type tolerance=Type(1e-8)) {
	return std::abs(static_cast<double>(val)) <= tolerance;
}

// checks if two values are similar (within a tolerance)
template <typename Type>
bool similar_values(Type val1, Type val2, Type tolerance=Type(1e-8)) {
	return close_to_zero(val1-val2, tolerance);
}

// returns a std::string with the memory usage of the matrix
std::string memory_footprint(size_t bytes)
{
	std::stringstream ss;
	ss << std::fixed;
	ss << std::setprecision(1);
	if (bytes < 1024) ss << bytes << " bytes";
	else if (bytes < 1024*1024) ss << bytes/1024.0 << " KB";
	else if (bytes < 1024*1024*1024) ss << bytes/(1024*1024.0) << " MB";
	else ss << bytes/(1024*1024*1024.0) << " GB";
	return ss.str();
}

// checks if a file exists.
bool check_file_exists(const std::string& fname) 
{
	#ifdef UMML_POSIX_SUPPORT
	// fast, using posix stat() from <sys/stat.h>
	struct stat buffer;
	return (stat(fname.c_str(), &buffer) == 0);
	#else
	// slower, using std::ifstream::open
	std::ifstream is(fname.c_str());
    return is.good();
	#endif
}

// returns the number of milliseconds between two time points t1 and t2
unsigned long duration_milli(std::chrono::steady_clock::time_point t1, std::chrono::steady_clock::time_point t2)
{
    typedef std::chrono::duration<double, std::milli> duration;	
    return (unsigned long)std::chrono::duration_cast<duration>(t2 - t1).count();
}

// format a time duration
std::string format_duration(unsigned long duration, int precision=2)
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(precision);
	double ms = (double)duration;
	if (ms <= 70000.0) {
		ss << std::fixed;
		ss << std::setprecision(2);
		ss << std::setw(5) << (ms/1000.0) << "s";
	} else if (ms <= 7200000.0) {
		int mins = (int)(unsigned long)ms/60000;
		int secs = (int)((unsigned long)ms/1000) % 60;
		ss << mins << "m " << std::setfill('0') << std::setw(2) << secs << "s";
	} else {
		ss << std::fixed;
		ss << std::setprecision(1);
		ss << (ms/3600000.0) << " hours";
	}
	return ss.str();
}

std::string format_duration(std::chrono::steady_clock::time_point t1, std::chrono::steady_clock::time_point t2, int precision=2)
{
	return format_duration(duration_milli(t1, t2), precision);
}


};     // namespace umml

#endif // UMML_UTILS_INCLUDED
