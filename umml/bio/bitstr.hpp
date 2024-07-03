#ifndef UMML_BITSTR_INCLUDED
#define UMML_BITSTR_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Genetic Algorithms: bitstring utility functions.

 FILE:     bitstr.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2022
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 
 Dependencies
 ~~~~~~~~~~~~
 STL vector
  
 Usage example
 ~~~~~~~~~~~~~
*/

#include <vector>


namespace umml {


/*
 bitstr2real
 
 converts the bitstring 'bs' to a floating point number
 ibits: the number of bits for the integer part
 fbits: the number of bits for the fractional (decimal) part
 pos: the position from where the conversion begins (for bitstrings that encode multiple numbers)
*/ 
template <typename Type=float>
void bitstr2real(Type& f, const std::vector<bool>& bs, int ibits, int fbits, int pos=0)
{
	f = Type(0.0);
	bool negative = bs[pos];
	for (int i=1; i<ibits; ++i) if (bs[pos+i]) f += (1 << (i-1));
	for (int i=0; i<fbits; ++i) if (bs[pos+ibits+i]) f += Type(0.5) / (1 << i);
	if (negative) f = -f;
}


/*
 double2bitstr
 
 converts the double 'val' to a bitstring 'bs' (has to be allocated)
 ibits: the number of bits for the integer part
 fbits: the number of bits for the fractional (decimal) part
 pos: the position from where the conversion begins (for bitstrings that encode multiple numbers)
*/ 
void double2bitstr(double val, std::vector<bool>& bs, int ibits, int fbits, int pos=0)
{
	// TODO
	bool negative = val < 0;
	if (negative) bs[pos] = 1;
	//int i_part = int(std::abs(val));
}


}; // namespace umml

#endif // UMML_BITSTR_INCLUDED
