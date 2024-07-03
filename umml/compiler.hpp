#ifndef UMML_COMPILER_INCLUDED
#define UMML_COMPILER_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 C++ compiler setup

 FILE:     compiler.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2022-23
 
 Namespace
 ~~~~~~~~~
 umml
*/


#include <iostream>

#ifdef NO_ASSERT
#define assert(ignore) ((void)0)
#else
#include <cassert>
#endif

#if defined(__INTEL_COMPILER)
	#define UMML_ICC
#elif defined(__GNUC__)
	#define UMML_GCC
	#define UMML_POSIX_SUPPORT
#elif defined(_MSC_VER)
	#define UMML_MSVC
	#define NOMINMAX
#endif


namespace umml {


#define UMML_MAX_WARNINGS 4
static bool warning_displayed[UMML_MAX_WARNINGS] = {false, false, false, false };

void umml_warn_once(int warning_num, const char* warning) {
	if (warning_num < UMML_MAX_WARNINGS) {
		if (!warning_displayed[warning_num]) std::cout << warning << "\n";
		warning_displayed[warning_num] = true;
	}
}


};     // namespace umml

#endif // UMML_COMPILER_INCLUDED
