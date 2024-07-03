#ifndef UMML_ALGO_INCLUDED
#define UMML_ALGO_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 Algorithms

 FILE:     algo.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-23
 KEYWORDS: priority queue, shuffle, resize
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 STL string
 STL vector
 
 Internal dependencies:
 STL algorithm
 STL map

 Functions
 ~~~~~~~~~
 * integer power
 * resize_rgb: resize an RGB image using bilinear filtering (interpolation)

 Classes
 ~~~~~~~
 * pqueue: Priority queue.
*/

#include <map>
#include <algorithm>
#include "compiler.hpp"


namespace umml {


/*
 base^exp with exp an integer > 0 
*/
template <typename T>
T powi(T base, int exp) 
{
	T result = 1;
	for (;;) {
		if (exp & 1) result *= base;
		exp >>= 1;
		if (!exp) break;
		base *= base;
	}
	return result;
}


/*
 resize a grayscale (one channel) image with bilinear interpolation
*/
template <typename Type>
void resize_channel(const Type* src, int h, int w, int src_pitch, Type* dst, int dh, int dw, int dst_pitch)
{
	float yratio = ((float)(h-1)) / dh;
	float xratio = ((float)(w-1)) / dw;
	for (int i=0; i<dh; ++i) {
		for (int j=0; j<dw; ++j) {
			int x = (int)(xratio * j);
			int y = (int)(yratio * i);
			float xdiff = xratio*j - x;
			float ydiff = yratio*i - y;
			int src_index = y*src_pitch + x;
			Type a = src[src_index];
			Type b = src[src_index + 1];
			Type c = src[src_index + src_pitch];
			Type d = src[src_index + src_pitch + 1];

			float val = a*(1-xdiff)*(1-ydiff) + b*xdiff*(1-ydiff) + c*ydiff*(1-xdiff) + d*xdiff*ydiff;

			dst[i*dst_pitch+j] = (Type)val;
		}
	}
}


/*
 Priority Queue
 It is commonly used in k-d tree, KNN
*/  
template <typename T>
class pqueue {
 public:
	pqueue(int cap) { capacity = cap; }

	int size() { return capacity; }
	int len() { return (int)elems.size(); }
	
    // Enqueues a new element into the queue with the specified priority. 
    void enqueue(double priority, const T& value) {
		elems.insert(std::make_pair(priority, value));
		if (len() > capacity) {
			typename std::map<double,T>::iterator last = elems.end();
			--last;
			elems.erase(last);
		}
	}

    // Returns the element with the smallest priority value.
    T dequeue() {
		T ret = elems.begin()->second;
		elems.erase(elems.begin());
		return ret;
	}

	// Returns the best priority 
	double best() const {
		return elems.size() ? elems.begin()->first : inf;
	}
	
	// Returns the worst priority
	double worst() const {
		return elems.size() ? elems.rbegin()->first : inf;
	}

 private:
	const double inf = std::numeric_limits<double>::infinity();
	std::map<double,T> elems;
	int capacity;
};


};     // namespace umml

#endif // UMML_ALGO_INCLUDED
