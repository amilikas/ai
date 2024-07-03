#ifndef UMML_STATS_INCLUDED
#define UMML_STATS_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Statistics

 FILE:     stats.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2024
 KEYWORDS: statistics, variance, deviation
 
 Namespace
 ~~~~~~~~~
 umml
 
 Notes
 ~~~~~
 PCA<double> works better than PCA<float>, since the later introduces many rounding errors
 due to lower precission. This is especially noticeable after an inverse_transform() 
  
 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 
 Internal dependencies:
 STL vector
 STL strings
 STL stringstream and iomanip

 Functions
 ~~~~~~~~~
 * variance:
 * deviation:
 * histogram:
*/


#include "umat.hpp"


namespace umml {


// Population variance
template <typename Type, template <typename> class Data>
Type variance(const Data<Type>& data, bool population=true) 
{
	if (data.empty()) return Type(-1.0);
	Type mean = data.sum() / data.len();
	return data.sum_squared(-mean) / (population ? data.len() : data.len()-1);
}

// Standard deviation directly from data
template <typename Type, template <typename> class Data>
Type deviation(const Data<Type>& data, bool population=true) 
{
	if (data.empty()) return Type(-1.0);
	return std::sqrt(variance(data, population));
}

// Histogram
// Works only for CPU memory (no padding)
template <typename Type>
class histogram {
 public:
	Type minval() const { return _minval; }
	Type maxval() const { return _maxval; }
	Type binwidth() const { return _binwidth; }
	std::vector<int> hist() const { return _hist; }
	
	template <typename Data>
	void fit(const Data& data, int nbins) {
		_hist.clear();
		_hist.resize(nbins);
		if (data.empty()) return;
		_minval = data.minimum();
		_maxval = data.maximum();
		_binwidth = (_maxval-_minval) / nbins;
		for (int i=0; i<data.len(); ++i) {
			Type val = data.sequential(i);
			int bin = static_cast<int>((val-_minval) / _binwidth);
			if (bin >= nbins) bin = nbins - 1;
			_hist[bin]++;
		}
	}

	std::string format(size_t decimals=2) const {
		Type from = _minval;
		std::stringstream ss;
		ss << std::fixed;
		ss << std::setprecision(decimals);
		for (size_t b=0; b<_hist.size(); ++b) {
			int count = _hist[b];
			Type to = from + _binwidth;
			ss << "w:" << from << "," << count << "\n";
			from = to;
		}
		return ss.str();
	}

	// block can be the "█" unicode character (U+2588)
	std::string graphic(size_t width=80, size_t decimals=1, size_t padding=0, const std::string& block="█") const {
		Type from = _minval;
		int maxcount = *std::max_element(_hist.begin(), _hist.end());
		std::stringstream ss;
		ss << std::fixed;
		ss << std::setprecision(decimals);
		ss << "Scale: 0.." << maxcount << "\n";
		for (size_t b=0; b<_hist.size(); ++b) {
			int count = _hist[b];
			Type to = from + _binwidth;
			ss << std::setw(2) << b+1 << " (" << std::setw(padding) 
			   << from << "," << std::setw(padding) << to << "): ";
			for (int i=0; i<(int)(width*(float)count/maxcount+0.5); ++i) ss << block;
			ss << "\n";
			from = to;
		}
		return ss.str();
	}
	
 private:
	std::vector<int> _hist;
	Type _binwidth;
	Type _minval;
	Type _maxval;
};


};     // namespace umml

#endif // UMML_STATS_INCLUDED
