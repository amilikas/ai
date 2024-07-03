#ifndef UMML_BITVEC_INCLUDED
#define UMML_BITVEC_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Space efficient vector of bits

 FILE:     bitvec.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2024
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 
 Dependencies
 ~~~~~~~~~~~~
 STL vector
 STL string
  
 Usage example
 ~~~~~~~~~~~~~
*/

#include <vector>
#include <string>
#include <fstream>
#include <cassert>


namespace umml {


class bitvec {
 private:
	std::vector<uint64_t> _data;
	size_t _size;

 public:
	bitvec(): _size(0) {}
	bitvec(size_t n) { resize(n); }
	bitvec(const std::vector<bool> v) { resize(v.size()); set(v); }

	void resize(size_t n) {
		_size = n;
		_data.resize((n+63)/64, 0);
	}

	size_t size() const { return _size; }

	void zero() {
		std::fill(_data.begin(), _data.end(), 0ULL);
	}

	void set(const std::vector<bool>& v) {
		assert(v.size()==size());
		for (size_t i=0; i<size(); ++i) set(i, v[i]);
	}

	void set(size_t pos, bool value) {
		assert(pos < _size && "Index out of range");
		if (value) {
			_data[pos/64] |= (1ULL << (pos%64));
		} else {
			_data[pos/64] &= ~(1ULL << (pos%64));
		}
	}

	bool get(size_t pos) const {
		assert(pos < _size && "Index out of range");
		return (_data[pos/64] >> (pos%64)) & 1;
	}

	void flip(size_t pos) {
		assert(pos < _size && "Index out of range");
		_data[pos/64] ^= (1ULL << (pos%64));
	}

	int popcount() const {
		int count = 0;
		for (size_t i=0; i < _data.size(); ++i) {
			count += __builtin_popcountll(_data[i]);
		}
		return count;
	}

	void copy(const bitvec& src, size_t n, size_t srcpos=0, size_t dstpos=0) {
		assert(srcpos+n <= src.size());
		assert(dstpos+n <= size());
		for (size_t i=0; i<n; ++i) set(dstpos+i, src.get(srcpos+i));
	}

	std::vector<bool> to_vector() const {
		std::vector<bool> v(size());
		for (size_t i=0; i<size(); ++i) v[i] = get(i);
		return v;
	}

	static bitvec zeros(size_t bps) {
		bitvec b(bps);
		b.zero();
		return b;
	}

	static bitvec ones(size_t bps) {
		bitvec b(bps);
		for (size_t i=0; i<bps; ++i) b.set(i, true);
		return b;
	}

	std::string format() const {
		if (_size==0) return "";
		std::string s(_size, '0');
		for (size_t pos=0; pos < _size; ++pos) {
			if (get(pos)) s[pos] = '1';
		}
		return s;
	}

	void save(std::ofstream& os) const {
		for (size_t i=0; i < _data.size(); ++i) {
			os.write((const char*)&_data[i], sizeof(uint64_t));
		}
	}

	void load(std::ifstream& is) {
		for (size_t i=0; i < _data.size(); ++i) {
			is.read((char*)&_data[i], sizeof(uint64_t));
		}		
	}
};


};     // namespace umml

#endif // UMML_BITVEC_INCLUDED