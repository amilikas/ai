#ifndef UMML_TYPES_INCLUDE
#define UMML_TYPES_INCLUDE

/*
 Machine Learning Artificial Intelligence Library

 FILE:     types.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2022
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
  
 Classes
 ~~~~~~~
 dims2 - 2D shape
 dims3 - 3D shape
 dims4 - 4D shape
*/


namespace umml {


// 2d shape
struct dims2 { 
	int x, y; 
	size_t size() const { return x*y; }
};

// 3d shape
struct dims3 { 
	int x, y, z; 
	size_t size() const { return x*y*z; }
};

// 4d shape
struct dims4 { 
	int x, y, z, t; 
	size_t size() const { return x*y*z*t; }
};


};     // namespace umml

#endif // UMML_TYPES_INCLUDE
