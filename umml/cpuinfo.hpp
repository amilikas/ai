#ifndef UMML_CPUINFO_INCLUDED
#define UMML_CPUINFO_INCLUDED

/*
 Get CPU id using CPUID x86 instruction.

 [1] https://linustechtips.com/topic/635593-how-do-i-get-cpu-mobo-ram-gpu-psu-info-with-c/
 
 Usage
 ~~~~~
 
 x86cpuinfo info;
 get_cpu_info(info);
 cout << get_cpu_model();
 cout << umml_compute_info(use_gpu:true/false);
*/

#include <stdint.h>
#include <cstring>
#include <string>
#include "dev.hpp"
#include "utils.hpp"

#ifdef __USE_OPENCL__
#include "ocl.hpp"
#endif

namespace umml {


// Information returned by CPUID subfunctions 0 and 1
struct	x86cpuinfo {
	std::string vendor;
	uint32_t    stepping;
	uint32_t    model;
	uint32_t    family;
	uint32_t    type;
	uint32_t    ext_model;
	uint32_t    ext_family;
};


// X86 CPU registers relevant to CPUID
struct _x86regs {
	uint32_t eax;
	uint32_t ebx;
	uint32_t ecx;
	uint32_t edx;
};

static inline void _cpuid(uint32_t eax_in, _x86regs* regs_out)
{
	asm volatile ("cpuid"	: 	"=a"(regs_out->eax), 
					"=b"(regs_out->ebx),
					"=c"(regs_out->ecx),
					"=d"(regs_out->edx)
			 	:	"a"(eax_in));
}

// Copies the given cpu register to given char buffer.
void _copy_reg_to_buff(uint32_t reg, char* buff)
{
	for (int i=0; i<4; ++i) {
		buff[i] = reg & 0xff;
		reg >>= 8;
	}
}

// Calls CPUID subfunction 0 to retrieve highest supported subfunction in eax.
uint32_t GetHighestFunction()
{
	_x86regs regs;
	_cpuid(0, &regs);
	return regs.eax;
}


// Calls CPUID subfunction 1 to retrieve CPUinfo, bitmasks are explained here: 
// https://en.wikipedia.org/wiki/CPUID#EAX.3D1:_Processor_Info_and_Feature_Bits
void get_cpu_info(x86cpuinfo& info)
{
	_x86regs regs;

	// vendor
	char vendor_id[13];
	_cpuid(0, &regs);
	std::memset(vendor_id, 0, sizeof(vendor_id));
	_copy_reg_to_buff(regs.ebx, vendor_id);
	_copy_reg_to_buff(regs.edx, vendor_id + 4);
	_copy_reg_to_buff(regs.ecx, vendor_id + 8);
	info.vendor = std::string(vendor_id);
	
	// features
	_cpuid(1, &regs);
	info.stepping = regs.eax & 0x0f;
	info.model = (regs.eax & 0xf0) >> 4;
	info.family = (regs.eax & 0xf00) >> 8;
	info.type = (regs.eax & 0x3000) >> 12;
	info.ext_model = (regs.eax & 0xf0000) >> 16;
	info.ext_family = (regs.eax & 0xff00000) >> 20;
}

std::string get_cpu_model()
{
	_x86regs regs;
	std::string model;
    for (unsigned int i=0x80000002; i<0x80000005; ++i) {
        _cpuid(i, &regs);
        model += std::string((const char*)&regs.eax, 4);
        model += std::string((const char*)&regs.ebx, 4);
        model += std::string((const char*)&regs.ecx, 4);
        model += std::string((const char*)&regs.edx, 4);
    }
    return ltrim(rtrim(model));
}


std::string umml_compute_info(bool use_gpu=false)
{
	std::stringstream ss;
	ss << "CPU: [" << get_cpu_model() << "]\n";

#ifdef __USE_OPENCL__
	ss << "GPU: [" << __ocl__.gpu_description() << "]\n";
#endif

#ifdef __USE_OPENMP__
	if (openmp<>::threads <= 1) ss << "single thread (but OpenMP is enabled)";
	else ss << openmp<>::threads << " threads with OpenMP";
#else
	ss << "single thread";
#endif

#ifdef __USE_BLAS__
	ss << ", OpenBLAS gemm.";
#endif

	return ss.str();
}


};     // namespace umml

#endif // UMML_CPUINFO_INCLUDED
