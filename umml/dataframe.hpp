#ifndef UMML_DATAFRAME_INCLUDED
#define UMML_DATAFRAME_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 Dataframe is a utility class that relies on the umat class to perform data handling routines.
 It works only in CPU memory, so every data handling must be done before the data is uploaded 
 to the GPU.

 FILE:     dataframe.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2023-2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 uvec
 umat
 umml rand
 STL algorithm
  
 Usage example
 ~~~~~~~~~~~~~
  
*/


#include <algorithm>
#include "umat.hpp"


namespace umml {


class dataframe {
 public:
	dataframe() {}

	// returns an indeces vector for each i that v1[i]==value
	template <typename T>
	uvec<int> select(const uvec<T>& v, T value);

	// returns an indeces vector for each i that v1[i]==v2[i]
	template <typename T>
	uvec<int> select_equal(const uvec<T>& v1, const uvec<T>& v2);

	// returns an indeces vector for each i that v1[i]!=v2[i]
	template <typename T>
	uvec<int> select_not_equal(const uvec<T>& v1, const uvec<T>& v2);

	// appends vector v1 and v2
	template <typename T>
	uvec<T> append(const uvec<T>& v1, const uvec<T>& v2);

	// returns a vector containing only 'n' elements staring from offset 'start'
	template <typename T>
	uvec<T> copy(const uvec<T>& v, int start, int n);

	// returns a vector containing only 'n' elements staring from offset 'start'
	template <typename T>
	uvec<T> copy(const uvec<T>& v, const uvec<int>& idcs);

	// returns a matrix containing only 'n' rows staring from row 'start'
	template <typename T>
	umat<T> copy_rows(const umat<T>& m, int start, int n);

	// returns a matrix only with rows that are indexed by 'idcs'
	template <typename T>
	umat<T> copy_rows(const umat<T>& m, const uvec<int>& idcs);

	// returns a matrix only with rows that are marked with 1 by 'mask'
	template <typename T>
	umat<T> copy_rows(const umat<T>& m, const std::vector<bool>& mask);
	
	// returns a matrix with every other row than those indexed by 'idcs'
	template <typename T>
	umat<T> exclude_rows(const umat<T>& m, const uvec<int>& idcs);

	// returns a matrix with every other row than those marked with 1 by 'mask'
	template <typename T>
	umat<T> exclude_rows(const umat<T>& m, const std::vector<bool>& mask);

	// stack two matrices horizontally (m2 in the right of m1)
	template <typename T>
	umat<T> hstack(const umat<T>& m1, const umat<T>& m2);
	
	// stack two matrices vertically (m1 then m2)
	template <typename T>
	umat<T> vstack(const umat<T>& m1, const umat<T>& m2);

	// drop a single column
	template <typename T>
	umat<T> drop_column(const umat<T>& m, int col) { return drop_columns(m, col, col); }

	// drop specified column
	// drop_columns(0,0) drops the first column
	// drop_columns(0,2) drops the first three columns (0,1 and 2)
	// drop_columns(-1,-1) drops the last column
	template <typename T>
	umat<T> drop_columns(const umat<T>& m, int first, int last);
	
	// drop the specified columns
	template <typename T>
	umat<T> drop_columns(const umat<T>& m, const std::vector<int>& cols);

	// drop the specified columns
	template <typename T>
	umat<T> drop_columns(const umat<T>& m, const std::vector<bool>& mask);
	
	// swap two rows of the matrix in-place, starting from column indexed by col_start.
	// cidx defaults to 0, meaning swap full rows.
	template <typename T>
	umat<T>& swap_rows(umat<T>& m, int row1, int row2, int col_start=0);

	// shuffle (X, y)
	template <typename XT, typename YT>
	void shuffle(umat<XT>& X, uvec<YT>& y, rng32& rng=global_rng());

	// shuffle (X, Y)
	template <typename XT, typename YT>
	void shuffle(umat<XT>& X, umat<YT>& Y, rng32& rng=global_rng());
};


template <typename T>
uvec<int> dataframe::select(const uvec<T>& v, T value)
{
	assert(v.dev()==device::CPU && "dataframe supports only CPU vectors.");
	std::vector<int> idcs;
	for (int i=0; i<v.len(); ++i) if (v(i)==value) idcs.push_back(i);
	int n = (int)idcs.size();
	uvec<int> out(n);
	out.set(idcs);
	return out;
}

template <typename T>
uvec<int> dataframe::select_equal(const uvec<T>& v1, const uvec<T>& v2)
{
	assert(v1.dev()==device::CPU && v2.dev()==device::CPU && "dataframe supports only CPU vectors.");
	assert(v1.len()==v2.len());
	std::vector<int> idcs;
	for (int i=0; i<v1.len(); ++i) if (v1(i)==v2(i)) idcs.push_back(i);
	int n = (int)idcs.size();
	uvec<int> out(n);
	out.set(idcs);
	return out;
}

template <typename T>
uvec<int> dataframe::select_not_equal(const uvec<T>& v1, const uvec<T>& v2)
{
	assert(v1.dev()==device::CPU && v2.dev()==device::CPU && "dataframe supports only CPU vectors.");
	assert(v1.len()==v2.len());
	std::vector<int> idcs;
	for (int i=0; i<v1.len(); ++i) if (v1(i)!=v2(i)) idcs.push_back(i);
	int n = (int)idcs.size();
	uvec<int> out(n);
	out.set(idcs);
	return out;
}

template <typename T>
uvec<T> dataframe::append(const uvec<T>& v1, const uvec<T>& v2)
{
	assert(v1.dev()==device::CPU && v2.dev()==device::CPU && "dataframe supports only CPU vectors.");
	uvec<T> out(v1.len()+v2.len());
	for (int i=0; i<v1.len(); ++i) out(i) = v1(i);
	for (int i=0; i<v2.len(); ++i) out(i+v1.len()) = v2(i);
	return out;
}

template <typename T>
uvec<T> dataframe::copy(const uvec<T>& v, int start, int n)
{
	assert(v.dev()==device::CPU && "dataframe supports only CPU vectors.");
	if (start < 0) start = v.len() + start;
	assert(start+n <= v.len());
	uvec<T> out(n);
	for (int i=0; i<n; ++i) out(i) = v(start+i);
	return out;
}

template <typename T>
uvec<T> dataframe::copy(const uvec<T>& v, const uvec<int>& idcs)
{
	assert(v.dev()==device::CPU && "dataframe supports only CPU vectors.");
	int n=idcs.len();
	uvec<T> out(n);
	for (int i=0; i<n; ++i) out(i) = v(idcs(i));
	return out;
}

template <typename T>
umat<T> dataframe::copy_rows(const umat<T>& m, int start, int n)
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	if (start < 0) start = m.ydim() + start;
	assert(start+n <= m.ydim());
	umat<T> out(n, m.xdim());
	for (int i=0; i<n; ++i) out.set_row(i, m.row_offset(start+i).get_cmem(), m.xdim());
	return out;
}
	
template <typename T>
umat<T> dataframe::copy_rows(const umat<T>& m, const uvec<int>& idcs)
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	int n=idcs.len();
	umat<T> out(n, m.xdim());
	for (int i=0; i<n; ++i) out.set_row(i, m.row_offset(idcs(i)).get_cmem(), m.xdim());
	return out;
}

template <typename T>
umat<T> dataframe::copy_rows(const umat<T>& m, const std::vector<bool>& mask)
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	assert(m.ydim()==(int)mask.size());
	int n = std::count(mask.begin(), mask.end(), true);
	umat<T> out(n, m.xdim());
	int k=0;
	for (int i=0; i<(int)mask.size(); ++i) {
		if (mask[i]==true) {
			out.set_row(k, m.row_offset(i).get_cmem(), m.xdim());
			k++;
		}
	}
	return out;
}

template <typename T>
umat<T> dataframe::exclude_rows(const umat<T>& m, const uvec<int>& idcs)
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	int n=m.ydim()-idcs.len();
	umat<T> out(n, m.xdim());
	int k=0, j=0;
	for (int i=0; i<m.ydim(); ++i) {
		if (i==idcs(j)) {
			j++;
			continue;
		}
		out.set_row(k++, m.row_offset(i).get_cmem(), m.xdim());
	}
	return out;
}

template <typename T>
umat<T> dataframe::exclude_rows(const umat<T>& m, const std::vector<bool>& mask)
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	assert(m.ydim()==(int)mask.size());
	int n = (int)std::count(mask.begin(), mask.end(), false);
	umat<T> out(n, m.xdim());
	int k=0;
	for (int i=0; i<(int)mask.size(); ++i) {
		if (mask[i]==false) {
			out.set_row(k, m.row_offset(i).get_cmem(), m.xdim());
			k++;
		}
	}
	return out;
}

template <typename T>
umat<T> dataframe::hstack(const umat<T>& m1, const umat<T>& m2) 
{
	assert(m1.dev()==device::CPU && "dataframe supports only CPU matrices.");
	assert(m1.ydim()==0 || m1.ydim()==m2.ydim());
	umat<T> out(m2.ydim(), m1.xdim()+m2.xdim());
	for (int i=0; i<out.ydim(); ++i) {
		T* r = out.row_offset(i).get_mem();
		std::memcpy(r, m1.row_offset(i).get_cmem(), m1.xdim()*sizeof(T));
		std::memcpy(r+m1.xdim(), m2.row_offset(i).get_cmem(), m2.xdim()*sizeof(T));
	}
	return out;
}
	
template <typename T>
umat<T> dataframe::vstack(const umat<T>& m1, const umat<T>& m2)
{
	assert(m1.dev()==device::CPU && "dataframe supports only CPU matrices.");
	assert(m1.xdim()==0 || m1.xdim()==m2.xdim());
	umat<T> out(m1.ydim()+m2.ydim(), m2.xdim());
	for (int i=0; i<m1.ydim(); ++i)
		std::memcpy(out.row_offset(i).get_mem(), m1.row_offset(i).get_cmem(), m1.xdim()*sizeof(T));
	for (int i=0; i<m2.ydim(); ++i)
		std::memcpy(out.row_offset(m1.ydim()+i).get_mem(), m2.row_offset(i).get_cmem(), m2.xdim()*sizeof(T));
	return out;
}

template <typename T>
umat<T> dataframe::drop_columns(const umat<T>& m, int first, int last) 
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	if (first < 0) first = m.xdim()+first;
	if (last < 0) last = m.xdim()+last;
	int n = last-first+1;
	assert(last >= first && n < m.xdim() && first >= 0 && first < m.xdim());
	umat<T> out(m.ydim(), m.xdim()-n);
	for (int i=0; i<m.ydim(); ++i) {
		T* pdst = out.row_offset(i).get_mem();
		const T* psrc = m.row_offset(i).get_cmem();
		for (int j=0; j<first; ++j) *pdst++ = *psrc++;
		psrc += n;
		for (int j=last+1; j<m.xdim(); ++j) *pdst++ = *psrc++;
	}
	return out;
}
	
template <typename T>
umat<T> dataframe::drop_columns(const umat<T>& m, const std::vector<int>& cols) 
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	int ndrop = (int)cols.size();
	assert(ndrop < m.xdim());
	for (int c : cols) if (c < 0 || c > m.xdim()) return m;
	umat<T> out(m.ydim(), m.xdim()-ndrop);
	int k=0;
	for (int j=0; j<m.xdim(); ++j) {
		if (std::find(cols.begin(), cols.end(), j) == cols.end()) {
			for (int i=0; i<m.ydim(); ++i) out(i,k) = m(i,j);
			++k;
		}
	}
	return out;
}

template <typename T>
umat<T> dataframe::drop_columns(const umat<T>& m, const std::vector<bool>& mask) 
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	int ndrop = std::count(mask.begin(), mask.end(), false);
	assert(ndrop < m.xdim());
	umat<T> out(m.ydim(), m.xdim()-ndrop);
	for (int i=0; i<m.ydim(); ++i) {
		int k=0;
		for (int j=0; j<m.xdim(); ++j) {
			if (!mask[j]) {
				out(i,k) = m(i,j);
				++k;
			}
		}
	}
	return out;
}

template <typename T>
umat<T>& dataframe::swap_rows(umat<T>& m, int r1, int r2, int cidx) 
{
	assert(m.dev()==device::CPU && "dataframe supports only CPU matrices.");
	for (int j=cidx; j<m.xdim(); ++j) std::swap(m(r1,j) , m(r2,j));
	return m;
}

template <typename XT, typename YT>
void dataframe::shuffle(umat<XT>& X, uvec<YT>& y, rng32& rng)
{
	assert(X.ydim()==y.len());
	std::vector<int> idcs;
	build_shuffled_indeces(idcs, X.ydim(), rng);
	umat<XT> tmpX = X;
	uvec<YT> tmpy = y;
	X.copy_rows(tmpX, idcs);
	y.copy(tmpy, idcs);
}

template <typename XT, typename YT>
void dataframe::shuffle(umat<XT>& X, umat<YT>& Y, rng32& rng)
{
	assert(X.ydim()==Y.ydim());
	std::vector<int> idcs;
	build_shuffled_indeces(idcs, X.ydim(), rng);
	umat<XT> tmpX = X;
	umat<YT> tmpY = Y;
	X.copy_rows(tmpX, idcs);
	Y.copy_rows(tmpY, idcs);
}


};     // namespace umml

#endif // UMML_DATAFRAME_INCLUDED
