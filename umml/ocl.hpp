#ifndef UMML_OCL_INCLUDED
#define UMML_OCL_INCLUDED

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "dev.hpp"
#include "func.hpp"
#include "blas_ocl.hpp"
#include "kernels_ocl.hpp"
#include "utils.hpp"


namespace umml {


//template <typename T=void>
class __OCL {
 private:
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	std::vector<cl::Kernel> kernels;
	int comp_units;
	std::string gpu_descr;

 private:
 	enum {
		dcopy2d, fcopy2d,
		dcopy3d, fcopy3d,
		dreciprocal, freciprocal,
		
		dv_seta, fv_seta,
		dv_setax, fv_setax,
		dv_yx2, fv_yx2,
		dv_ypa, fv_ypa,
		dv_axpy, fv_axpy,
		dv_zaxpby, fv_zaxpby,
		dv_hadamard, fv_hadamard,
		dv_fypax, fv_fypax,
		dv_svep, fv_svep,
		dv_svep2, fv_svep2,
		dv_sve, fv_sve, iv_sve,
		dv_cnteq, fv_cnteq,
		dv_eucl2, fv_eucl2,
		dv_manh, fv_manh,
		dv_argmaxp, fv_argmaxp,
		dv_argmax, fv_argmax,
		dv_argminp, fv_argminp,
		dv_argmin, fv_argmin,
		dv_dot, fv_dot,

		dm_seta, fm_seta,
		dm_setax, fm_setax,
		dm_mpa, fm_mpa,
		dm_mpax, fm_mpax,
		dm_mma, fm_mma,
		dm_axpy, fm_axpy,
		dm_zaxpby, fm_zaxpby,
		dm_hadamard, fm_hadamard,
		dm_mmulv, fm_mmulv,
		dm_mdivv, fm_mdivv,
		dm_fmpax, fm_fmpax,
		dm_smrp, fm_smrp,
		dm_smrp2, fm_smrp2,
		dm_smr, fm_smr, im_smr,
		dm_smc, fm_smc, im_smc,
		
		dm_cnteq, fm_cnteq,
		dm_eucl2, fm_eucl2,
		dm_manh, fm_manh,
		dm_argmax, fm_argmax,
		dm_matmax, fm_matmax,
		dm_argmin, fm_argmin,
		dm_matmin, fm_matmin,

		dm_outer, fm_outer,
		dv_gemv1, fv_gemv1,
		dv_gemv2, fv_gemv2,
		dm_gemm, fm_gemm,
		dm_gemt, fm_gemt,
		dm_gram, fm_gram,
		dm_bmm, fm_bmm,

		dc_setax, fc_setax,
		dc_func3d, fc_func3d,
		dc_sum3d, fc_sum3d,
		dc_outer3d, fc_outer3d,

		kernels_length
 	};


 public:
	__OCL() {
		get_devices();
		context = cl::Context({device});
		queue = cl::CommandQueue(context, device);
		build_programs();
	}

	std::string gpu_description() const {
		return gpu_descr;
	}
	
	template <typename Type>
	cl::Buffer alloc(int n) {
		return cl::Buffer(context, CL_MEM_READ_WRITE, n*sizeof(Type));
	}

	template <typename Type>
	void to_gpu(cl::Buffer& buffer, const Type* mem, int n) {
		queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, n*sizeof(Type), mem);
	}

	template <typename Type>
	void to_cpu(cl::Buffer& buffer, Type* mem, int n) {
		queue.enqueueReadBuffer(buffer, CL_TRUE, 0, n*sizeof(Type), mem);
	}

	template <typename Type>
	void set_buffer_element(cl::Buffer& buffer, int offset, Type value) {
		queue.enqueueWriteBuffer(buffer, CL_TRUE, offset*sizeof(Type), sizeof(Type), &value);
	}

	template <typename Type>
	Type get_buffer_element(const cl::Buffer& buffer, int offset) {
		Type value;
		queue.enqueueReadBuffer(buffer, CL_TRUE, offset*sizeof(Type), sizeof(Type), &value);
		return value;
	}

	template <typename Type>
	void fill(cl::Buffer& dst, Type val, int offset, int n) {
		queue.enqueueFillBuffer(dst, val, offset*sizeof(Type), n*sizeof(Type));
	}

	template <typename Type>
	void copy(const cl::Buffer& src, cl::Buffer& dst, int src_offset, int dst_offset, int n) {
		queue.enqueueCopyBuffer(src, dst, src_offset*sizeof(Type), dst_offset*sizeof(Type), n*sizeof(Type));
	}

	template <typename Type>
	void copy2d(const cl::Buffer& src, int spitch, cl::Buffer& dst, int dpitch, int sy, int sx, int dy, int dx, int ylen, int xlen) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dcopy2d] : kernels[fcopy2d]);
		kernel.setArg(0, src);
		kernel.setArg(1, spitch);
		kernel.setArg(2, dst);
		kernel.setArg(3, dpitch);
		kernel.setArg(4, sy);
		kernel.setArg(5, sx);
		kernel.setArg(6, dy);
		kernel.setArg(7, dx);
		kernel.setArg(8, ylen);
		kernel.setArg(9, xlen);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(xlen),TILEPAD(ylen)), cl::NullRange);
	}

	template <typename Type>
	void copy3d(int zdim, const cl::Buffer& src, int spitch, int szstride, 
				cl::Buffer& dst, int dpitch, int dzstride, int sy, int sx, int dy, int dx, int ylen, int xlen) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dcopy3d] : kernels[fcopy3d]);
		kernel.setArg( 0, zdim);
		kernel.setArg( 1, src);
		kernel.setArg( 2, spitch);
		kernel.setArg( 3, szstride);
		kernel.setArg( 4, dst);
		kernel.setArg( 5, dpitch);
		kernel.setArg( 6, dzstride);
		kernel.setArg( 7, sy);
		kernel.setArg( 8, sx);
		kernel.setArg( 9, dy);
		kernel.setArg(10, dx);
		kernel.setArg(11, ylen);
		kernel.setArg(12, xlen);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(xlen),TILEPAD(ylen),zdim), cl::NullRange);
	}

	template <typename Type>
	void reciprocal(Type alpha, const cl::Buffer& x, int n, cl::Buffer& y) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dreciprocal] : kernels[freciprocal]);
		kernel.setArg(0, alpha);
		kernel.setArg(1, x);
		kernel.setArg(2, n);
		kernel.setArg(3, y);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
	}

 private:
	void get_devices() {
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size()==0) {
			std::cout << "No platforms found. Check OpenCL installation!\n";
			exit(1);
		}
		platform = platforms[0];
		std::cout << "Using platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << "\n";
		gpu_descr = std::string(platforms[0].getInfo<CL_PLATFORM_NAME>()) + " ";

		std::vector<cl::Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if (devices.size()==0){
			std::cout << "No devices found. Check OpenCL installation!\n";
			exit(1);
		}
		device = devices[0];
		for (size_t i=0; i<devices.size(); ++i) {
			std::string info = devices[i].getInfo<CL_DEVICE_NAME>();
			comp_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			if (i==0) {
				std::cout << "[*] ";
				gpu_descr += info + ", " + std::to_string(comp_units) + " CUs";
			} else {
				std::cout << "[ ] ";
			}
			std::cout << info << " " << comp_units << " compute units\n";
		}
	}

	// replaces kernel name and parameter types
	std::string parse_code(const std::string& kname, const std::string& tname, const std::string& code) {
		std::string kernel_code = code;
		size_t pos;
		pos = kernel_code.find("__name__");
		if (pos != std::string::npos) kernel_code.replace(pos, 8, kname);
		while ((pos=kernel_code.find("__type__")) != std::string::npos) {
			kernel_code.replace(pos, 8, tname);
		}
		//std::cout << "-------------\n" << kernel << "-------------\n";
		return kernel_code;
	}

 	void create_kernels(cl::Program& program, int dk, const std::string& dname, int fk, const std::string& fname) {
		cl_int kerr = CL_SUCCESS;
 		kernels[dk] = cl::Kernel(program, dname.c_str(), &kerr);
 		if (kerr != CL_SUCCESS) std::cerr << "Error creating kernel " << dk << " (enum). Error code=" << kerr << "\n";
 		kernels[fk] = cl::Kernel(program, fname.c_str(), &kerr);
 		if (kerr != CL_SUCCESS) std::cerr << "Error creating kernel " << dk << " (enum). Error code=" << kerr << "\n";
 	}

 	void push_sources(cl::Program::Sources& sources, const std::string& source, const std::string& dname, const std::string& fname) {
 		push_source_code(sources, source, dname, "double");
 		push_source_code(sources, source, fname, "float");
 	}

	// load kernel codes and build kernels
	void build_programs() {
		cl::Program::Sources sources;

		push_source_code(sources, ocl_defines_code, "", "");
		push_source_code(sources, ocl_fenums_code,	"", "");
		push_source_code(sources, ocl_sve_code,     "iv_sve", "int");
		push_source_code(sources, ocl_smr_code,     "im_smr", "int");
		push_source_code(sources, ocl_smc_code,     "im_smc", "int");

		push_sources(sources, ocl_copy2d_code, 		"dcopy2d","fcopy2d");
		push_sources(sources, ocl_copy3d_code, 		"dcopy3d","fcopy3d");
		push_sources(sources, ocl_reciprocal_code,	"dreciprocal","freciprocal");
		push_sources(sources, ocl_funcs_code, 		"", "");
		
		push_sources(sources, ocl_yseta_code, 		"dv_seta", 		"fv_seta");
		push_sources(sources, ocl_ysetax_code, 		"dv_setax", 	"fv_setax");
		push_sources(sources, ocl_yx2_code, 		"dv_yx2", 		"fv_yx2");
		push_sources(sources, ocl_ypa_code, 		"dv_ypa", 		"fv_ypa");
		push_sources(sources, ocl_axpy_code, 		"dv_axpy", 		"fv_axpy");
		push_sources(sources, ocl_zaxpby_code, 		"dv_zaxpby", 	"fv_zaxpby");
		push_sources(sources, ocl_vhadamard_code, 	"dv_hadamard", 	"fv_hadamard");
		push_sources(sources, ocl_fypax_code, 		"dv_fypax", 	"fv_fypax");
		push_sources(sources, ocl_svep_code, 		"dv_svep", 		"fv_svep");
		push_sources(sources, ocl_svep2_code, 		"dv_svep2", 	"fv_svep2");
		push_sources(sources, ocl_sve_code, 		"dv_sve", 		"fv_sve");
		push_sources(sources, ocl_vcnteq_code,		"dv_cnteq", 	"fv_cnteq");
		push_sources(sources, ocl_veucl2_code,		"dv_eucl2", 	"fv_eucl2");
		push_sources(sources, ocl_vmanh_code,		"dv_manh", 		"fv_manh");
		push_sources(sources, ocl_vargmaxp_code, 	"dv_argmaxp", 	"fv_argmaxp");
		push_sources(sources, ocl_vargmax_code, 	"dv_argmax", 	"fv_argmax");
		push_sources(sources, ocl_vargminp_code, 	"dv_argminp", 	"fv_argminp");
		push_sources(sources, ocl_vargmin_code, 	"dv_argmin", 	"fv_argmin");
		push_sources(sources, ocl_dotp_code, 		"dv_dot", 		"fv_dot");
		
		push_sources(sources, ocl_ma_code, 			"dm_seta", 		"fm_seta");
		push_sources(sources, ocl_man_code, 		"dm_setax", 	"fm_setax");
		push_sources(sources, ocl_mpa_code, 		"dm_mpa", 		"fm_mpa");
		push_sources(sources, ocl_mpax_code, 		"dm_mpax", 		"fm_mpax");
		push_sources(sources, ocl_mma_code, 		"dm_mma", 		"fm_mma");
		push_sources(sources, ocl_axpym_code, 		"dm_axpy", 		"fm_axpy");
		push_sources(sources, ocl_zaxpbym_code, 	"dm_zaxpby", 	"fm_zaxpby");
		push_sources(sources, ocl_hadamard_code, 	"dm_hadamard", 	"fm_hadamard");
		push_sources(sources, ocl_mmulv_code, 		"dm_mmulv", 	"fm_mmulv");
		push_sources(sources, ocl_mdivv_code, 		"dm_mdivv", 	"fm_mdivv");
		push_sources(sources, ocl_fmpax_code, 		"dm_fmpax", 	"fm_fmpax");
		push_sources(sources, ocl_smrp_code, 		"dm_smrp", 		"fm_smrp");
		push_sources(sources, ocl_smrp2_code, 		"dm_smrp2", 	"fm_smrp2");
		push_sources(sources, ocl_smr_code, 		"dm_smr", 		"fm_smr");
		push_sources(sources, ocl_smc_code, 		"dm_smc", 		"fm_smc");
		push_sources(sources, ocl_mcnteq_code,		"dm_cnteq", 	"fm_cnteq");
		push_sources(sources, ocl_meucl2_code,		"dm_eucl2", 	"fm_eucl2");
		push_sources(sources, ocl_mmanh_code,		"dm_manh", 		"fm_manh");
		push_sources(sources, ocl_margmax_code, 	"dm_argmax", 	"fm_argmax");
		push_sources(sources, ocl_matmax_code, 		"dm_matmax", 	"fm_matmax");
		push_sources(sources, ocl_margmin_code, 	"dm_argmin", 	"fm_argmin");
		push_sources(sources, ocl_matmin_code, 		"dm_matmin", 	"fm_matmin");
		
		push_sources(sources, ocl_outer_code, 		"dm_outer", 	"fm_outer");
		push_sources(sources, ocl_gemv1_code, 		"dv_gemv1", 	"fv_gemv1");
		push_sources(sources, ocl_gemv2_code, 		"dv_gemv2", 	"fv_gemv2");
		push_sources(sources, ocl_gemm2_code, 		"dm_gemm", 		"fm_gemm");
		push_sources(sources, ocl_gemt_code, 		"dm_gemt", 		"fm_gemt");
		push_sources(sources, ocl_gram_code, 		"dm_gram", 		"fm_gram");
		push_sources(sources, ocl_bmm_code, 		"dm_bmm", 		"fm_bmm");

		push_sources(sources, ocl_can_code, 		"dc_setax", 	"fc_setax");
		push_sources(sources, ocl_func3d_code, 		"dc_func3d", 	"fc_func3d");
		push_sources(sources, ocl_sum3d_code, 		"dc_sum3d", 	"fc_sum3d");
		push_sources(sources, ocl_outer3d_code, 	"dc_outer3d", 	"fc_outer3d");

		// compile program
		cl::Program program = compile_sources(sources);

		// create kernels
		kernels.resize(kernels_length);	
		kernels[iv_sve] = cl::Kernel(program,   "iv_sve");
		kernels[im_smr] = cl::Kernel(program,   "im_smr");
		kernels[im_smc] = cl::Kernel(program,   "im_smc");
		create_kernels(program, dcopy2d, 		"dcopy2d", 		fcopy2d, 		"fcopy2d");		 // copy2d
		create_kernels(program, dcopy3d, 		"dcopy3d", 		fcopy3d, 		"fcopy3d");		 // copy3d
		create_kernels(program, dreciprocal,	"dreciprocal", 	freciprocal,	"freciprocal");  // reciprocal (α*1/x)
		create_kernels(program, dv_seta, 		"dv_seta", 		fv_seta, 		"fv_seta");		 // y = α
		create_kernels(program, dv_setax, 		"dv_setax", 	fv_setax, 		"fv_setax");	 // y = αx
		create_kernels(program, dv_yx2, 		"dv_yx2", 		fv_yx2, 		"fv_yx2");	 	 // y = x^2
		create_kernels(program, dv_ypa, 		"dv_ypa", 		fv_ypa, 		"fv_ypa");		 // y += α
		create_kernels(program, dv_axpy, 		"dv_axpy", 		fv_axpy, 		"fv_axpy");		 // y += αx
		create_kernels(program, dv_zaxpby, 		"dv_zaxpby", 	fv_zaxpby, 		"fv_zaxpby");	 // z = αx + βy
		create_kernels(program, dv_hadamard, 	"dv_hadamard", 	fv_hadamard,	"fv_hadamard");	 // z = x*y
		create_kernels(program, dv_fypax, 		"dv_fypax", 	fv_fypax, 		"fv_fypax");	 // y = f(y+αx)
		create_kernels(program, dv_svep, 		"dv_svep", 		fv_svep, 		"fv_svep");		 // p = sum(v) partial
		create_kernels(program, dv_svep2, 		"dv_svep2", 	fv_svep2, 		"fv_svep2");	 // p = sum(v^2) partial
		create_kernels(program, dv_sve, 		"dv_sve", 		fv_sve, 		"fv_sve");		 // s = sum(v)
		create_kernels(program, dv_cnteq, 		"dv_cnteq",		fv_cnteq, 		"fv_cnteq");	 // count equal elements
		create_kernels(program, dv_eucl2, 		"dv_eucl2",		fv_eucl2, 		"fv_eucl2");	 // euclidean squared
		create_kernels(program, dv_manh, 		"dv_manh",		fv_manh, 		"fv_manh");	 	 // manhattan
		create_kernels(program, dv_argmaxp, 	"dv_argmaxp", 	fv_argmaxp, 	"fv_argmaxp");	 // i = argmax(v) partial
		create_kernels(program, dv_argmax, 		"dv_argmax", 	fv_argmax, 		"fv_argmax");	 // i = argmax(v)
		create_kernels(program, dv_argminp, 	"dv_argminp", 	fv_argminp, 	"fv_argminp");	 // i = argmin(v) partial
		create_kernels(program, dv_argmin, 		"dv_argmin", 	fv_argmin, 		"fv_argmin");	 // i = argmin(v)
		create_kernels(program, dv_dot, 		"dv_dot", 		fv_dot, 		"fv_dot");		 // δ = αx.βy
		create_kernels(program, dm_seta, 		"dm_seta", 		fm_seta, 		"fm_seta");		 // M = α
		create_kernels(program, dm_setax, 		"dm_setax", 	fm_setax, 		"fm_setax");	 // M = αN
		create_kernels(program, dm_mpa, 		"dm_mpa", 		fm_mpa, 		"fm_mpa");		 // M += α
		create_kernels(program, dm_mpax, 		"dm_mpax", 		fm_mpax, 		"fm_mpax");		 // M += αx
		create_kernels(program, dm_mma, 		"dm_mma", 		fm_mma, 		"fm_mma");		 // M *= α
		create_kernels(program, dm_axpy, 		"dm_axpy", 		fm_axpy, 		"fm_axpy");		 // Y += αX
		create_kernels(program, dm_zaxpby, 		"dm_zaxpby", 	fm_zaxpby,		"fm_zaxpby");	 // Z = αX + βY
		create_kernels(program, dm_hadamard, 	"dm_hadamard", 	fm_hadamard,	"fm_hadamard");	 // C = A*B
		create_kernels(program, dm_mmulv, 		"dm_mmulv", 	fm_mmulv,		"fm_mmulv");	 // C *= v
		create_kernels(program, dm_mdivv, 		"dm_mdivv", 	fm_mdivv,		"fm_mdivv");	 // C /= v
		create_kernels(program, dm_fmpax, 		"dm_fmpax", 	fm_fmpax, 		"fm_fmpax");	 // M = f(M + αx)
		create_kernels(program, dm_smrp, 		"dm_smrp", 		fm_smrp, 		"fm_smrp");	 	 // si = ΣΜij partial
		create_kernels(program, dm_smrp2, 		"dm_smrp2", 	fm_smrp2, 		"fm_smrp2");	 // si = ΣΜij^2 partial
		create_kernels(program, dm_smr, 		"dm_smr", 		fm_smr, 		"fm_smr");	 	 // si = ΣΜij
		create_kernels(program, dm_smc, 		"dm_smc", 		fm_smc, 		"fm_smc");	 	 // sj = ΣΜij
		create_kernels(program, dm_cnteq, 		"dm_cnteq",		fm_cnteq, 		"fm_cnteq");	 // count equal elements
		create_kernels(program, dm_eucl2, 		"dm_eucl2",		fm_eucl2, 		"fm_eucl2");	 // euclidean squared
		create_kernels(program, dm_manh, 		"dm_manh",		fm_manh, 		"fm_manh");	 	 // manhattan
		create_kernels(program, dm_argmax, 		"dm_argmax", 	fm_argmax, 		"fm_argmax");	 // i = argmax(M_rows)
		create_kernels(program, dm_matmax, 		"dm_matmax", 	fm_matmax, 		"fm_matmax");	 // max cols/rows
		create_kernels(program, dm_argmin, 		"dm_argmin", 	fm_argmin, 		"fm_argmin");	 // i = argmin(M_rows)
		create_kernels(program, dm_matmin, 		"dm_matmin", 	fm_matmin, 		"fm_matmin");	 // min cols/rows
		create_kernels(program, dm_outer, 		"dm_outer", 	fm_outer, 		"fm_outer");	 // M = v x u
		create_kernels(program, dv_gemv1, 		"dv_gemv1", 	fv_gemv1, 		"fv_gemv1");	 // y = Ax (naive)
		create_kernels(program, dv_gemv2, 		"dv_gemv2", 	fv_gemv2, 		"fv_gemv2"); 	 // y = Ax (tiled)
		create_kernels(program, dm_gemm, 		"dm_gemm", 		fm_gemm, 		"fm_gemm");		 // C = A.B
		create_kernels(program, dm_gemt, 		"dm_gemt", 		fm_gemt, 		"fm_gemt");		 // C = A^T
		create_kernels(program, dm_gram, 		"dm_gram", 		fm_gram, 		"fm_gram");		 // C = A^T*A (gramm matrix)
		create_kernels(program, dm_bmm, 		"dm_bmm", 		fm_bmm, 		"fm_bmm");		 // C = A.B batched

		create_kernels(program, dc_setax, 		"dc_setax", 	fc_setax, 		"fc_setax");	 // C = αN
		create_kernels(program, dc_func3d, 		"dc_func3d", 	fc_func3d, 		"fc_func3d");	 // C = f(C)
		create_kernels(program, dc_sum3d, 		"dc_sum3d", 	fc_sum3d, 		"fc_sum3d");	 // M = Σslices
		create_kernels(program, dc_outer3d, 	"dc_outer3d", 	fc_outer3d, 	"fc_outer3d");	 // outer product 3d
	}

 public:
	void push_source_code(cl::Program::Sources& sources, const std::string& source, 
						  const std::string& name, const std::string& type) {
		std::string code = parse_code(name, type, source);
		sources.push_back({code.c_str(),code.length()});
	}

	cl::Program compile_sources(const cl::Program::Sources& sources) {
		cl::Program program(context, sources);
		if (program.build({device}) != CL_SUCCESS) {
			std::cerr << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
			std::exit(1);
		}
		return program;
	}

	/*
	cl::Kernel create_kernel(const cl::Program& program, const std::string& kname) {
		return cl::Kernel(program, kname.c_str());
	}
	*/

 	void execute(cl::Kernel& kernel, const cl::NDRange& offset, const cl::NDRange& global, 
				 const cl::NDRange& local=cl::NullRange) {
 		queue.enqueueNDRangeKernel(kernel, offset, global, local);
 		queue.flush();
 	}

 	void queue_flush() {
 		queue.flush();
 	}

 	void queue_finish() {
 		queue.finish();
 	}



 	//
 	// Vectors
 	//

	template <typename Type>
	void vec_set(cl::Buffer& y, int n, Type alpha) {
		fill(y, alpha, 0, n);
	}
	
	template <typename Type>
	void vec_set(cl::Buffer& y, const cl::Buffer& x, int n, Type alpha) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_setax] : kernels[fv_setax]);
		kernel.setArg(0, y);
		kernel.setArg(1, x);
		kernel.setArg(2, n);
		kernel.setArg(3, alpha);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
	}

	template <typename Type>
	void vec_squared(cl::Buffer& y, const cl::Buffer& x, int n) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_yx2] : kernels[fv_yx2]);
		kernel.setArg(0, y);
		kernel.setArg(1, x);
		kernel.setArg(2, n);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
	}
	
	template <typename Type>
	void vec_plus(cl::Buffer& y, int n, Type alpha) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_ypa] : kernels[fv_ypa]);
		kernel.setArg(0, y);
		kernel.setArg(1, n);
		kernel.setArg(2, alpha);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
	}
	
	template <typename Type>
	void axpy(Type alpha, const cl::Buffer& x, int n, cl::Buffer& y) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_axpy] : kernels[fv_axpy]);
		kernel.setArg(0, alpha);
		kernel.setArg(1, x);
		kernel.setArg(2, n);
		kernel.setArg(3, y);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
	}
	
	template <typename Type>
	void zaxpby(Type alpha, const cl::Buffer& x, int n, Type beta, const cl::Buffer& y, cl::Buffer& z) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_zaxpby] : kernels[fv_zaxpby]);
		kernel.setArg(0, alpha);
		kernel.setArg(1, x);
		kernel.setArg(2, n);
		kernel.setArg(3, beta);
		kernel.setArg(4, y);
		kernel.setArg(5, z);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
	}

	template <typename Type>
	void hadamard(const cl::Buffer& x, const cl::Buffer& y, cl::Buffer& z, int n) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_hadamard] : kernels[fv_hadamard]);
		kernel.setArg(0, x);
		kernel.setArg(1, y);
		kernel.setArg(2, z);
		kernel.setArg(3, n);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
	}
	
	template <typename Type>
	void vec_func(cl::Buffer& y, int n, int f, Type alpha=0, const cl::Buffer& x=NullBuffer) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_fypax] : kernels[fv_fypax]);
		kernel.setArg(0, y);
		kernel.setArg(1, n);
		kernel.setArg(2, f);
		kernel.setArg(3, alpha);
		kernel.setArg(4, x);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
	}
	
	template <typename Type>
	void sve(const cl::Buffer& a, int n, cl::Buffer& sum) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer partial = alloc<Type>(workgroups);
		cl::Kernel& ksvep = (sizeof(Type)==sizeof(double) ? kernels[dv_svep] : kernels[fv_svep]);
		ksvep.setArg(0, a);
		ksvep.setArg(1, n);
		ksvep.setArg(2, partial);
		execute(ksvep, cl::NullRange, cl::NDRange(workgroups*WS), cl::NDRange(WS));

		cl::Kernel& ksve = (sizeof(Type)==sizeof(double) ? kernels[dv_sve] : kernels[fv_sve]);
		ksve.setArg(0, partial);
		ksve.setArg(1, workgroups);
		ksve.setArg(2, sum);
		execute(ksve, cl::NullRange, cl::NDRange(1), cl::NullRange);
	}

	template <typename Type>
	void sve2(const cl::Buffer& a, int n, Type alpha, cl::Buffer& sum) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer partial = alloc<Type>(workgroups);
		cl::Kernel& ksvep2 = (sizeof(Type)==sizeof(double) ? kernels[dv_svep2] : kernels[fv_svep2]);
		ksvep2.setArg(0, a);
		ksvep2.setArg(1, n);
		ksvep2.setArg(2, alpha);
		ksvep2.setArg(3, partial);
		execute(ksvep2, cl::NullRange, cl::NDRange(workgroups*WS), cl::NDRange(WS));

		cl::Kernel& ksve = (sizeof(Type)==sizeof(double) ? kernels[dv_sve] : kernels[fv_sve]);
		ksve.setArg(0, partial);
		ksve.setArg(1, workgroups);
		ksve.setArg(2, sum);
		execute(ksve, cl::NullRange, cl::NDRange(1), cl::NullRange);
	}

	template <typename Type>
	void count_equal(const cl::Buffer& a, int n, const cl::Buffer& b, cl::Buffer& cnt) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;

		cl::Buffer partial = alloc<int>(workgroups);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_cnteq] : kernels[fv_cnteq]);
		kernel.setArg(0, a);
		kernel.setArg(1, n);
		kernel.setArg(2, b);
		kernel.setArg(3, partial);
		execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS), cl::NDRange(WS));

		kernels[iv_sve].setArg(0, partial);
		kernels[iv_sve].setArg(1, workgroups);
		kernels[iv_sve].setArg(2, cnt);
		execute(kernels[iv_sve], cl::NullRange, cl::NDRange(1), cl::NullRange);
	}

	template <typename Type>
	void dist_squared(const cl::Buffer& a, int n, const cl::Buffer& b, cl::Buffer& dist) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;

		cl::Buffer partial = alloc<Type>(workgroups);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_eucl2] : kernels[fv_eucl2]);
		kernel.setArg(0, a);
		kernel.setArg(1, n);
		kernel.setArg(2, b);
		kernel.setArg(3, partial);
		execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS), cl::NDRange(WS));

		cl::Kernel& ksve = (sizeof(Type)==sizeof(double) ? kernels[dv_sve] : kernels[fv_sve]);
		ksve.setArg(0, partial);
		ksve.setArg(1, workgroups);
		ksve.setArg(2, dist);
		execute(ksve, cl::NullRange, cl::NDRange(1), cl::NullRange);
	}

	template <typename Type>
	void manhattan(const cl::Buffer& a, int n, const cl::Buffer& b, cl::Buffer& dist) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;

		cl::Buffer partial = alloc<Type>(workgroups);
		kernels[fv_manh].setArg(0, a);
		kernels[fv_manh].setArg(1, n);
		kernels[fv_manh].setArg(2, b);
		kernels[fv_manh].setArg(3, partial);
		execute(kernels[fv_manh], cl::NullRange, cl::NDRange(workgroups*WS), cl::NDRange(WS));

		cl::Kernel& ksve = (sizeof(Type)==sizeof(double) ? kernels[dv_sve] : kernels[fv_sve]);
		ksve.setArg(0, partial);
		ksve.setArg(1, workgroups);
		ksve.setArg(2, dist);
		execute(ksve, cl::NullRange, cl::NDRange(1), cl::NullRange);
	}
	
	template <typename Type>
	void argmax(const cl::Buffer& a, int n, cl::Buffer& pos) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;

		cl::Buffer partial = alloc<Type>(workgroups);
		cl::Buffer partialIdx = alloc<int>(workgroups);
		cl::Kernel& kargmaxp = (sizeof(Type)==sizeof(double) ? kernels[dv_argmaxp] : kernels[fv_argmaxp]);
		kargmaxp.setArg(0, a);
		kargmaxp.setArg(1, n);
		kargmaxp.setArg(2, partial);
		kargmaxp.setArg(3, partialIdx);
		execute(kargmaxp, cl::NullRange, cl::NDRange(workgroups*WS), cl::NDRange(WS));

		cl::Kernel& kargmax = (sizeof(Type)==sizeof(double) ? kernels[dv_argmax] : kernels[fv_argmax]);
		kargmax.setArg(0, partial);
		kargmax.setArg(1, partialIdx);
		kargmax.setArg(2, workgroups);
		kargmax.setArg(3, pos);
		execute(kargmax, cl::NullRange, cl::NDRange(1), cl::NullRange);
	}

	template <typename Type>
	void argmin(const cl::Buffer& a, int n, cl::Buffer& pos) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;

		cl::Buffer partial = alloc<Type>(workgroups);
		cl::Buffer partialIdx = alloc<int>(workgroups);
		cl::Kernel& kargminp = (sizeof(Type)==sizeof(double) ? kernels[dv_argminp] : kernels[fv_argminp]);
		kargminp.setArg(0, a);
		kargminp.setArg(1, n);
		kargminp.setArg(2, partial);
		kargminp.setArg(3, partialIdx);
		execute(kargminp, cl::NullRange, cl::NDRange(workgroups*WS), cl::NDRange(WS));

		cl::Kernel& kargmin = (sizeof(Type)==sizeof(double) ? kernels[dv_argmin] : kernels[fv_argmin]);
		kargmin.setArg(0, partial);
		kargmin.setArg(1, partialIdx);
		kargmin.setArg(2, workgroups);
		kargmin.setArg(3, pos);
		execute(kargmin, cl::NullRange, cl::NDRange(1), cl::NullRange);
	}

	template <typename Type>
	void dot(Type alpha, const cl::Buffer& a, int n, Type beta, const cl::Buffer& b, cl::Buffer& y) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer partial = alloc<Type>(workgroups);

		cl::Kernel& kdot = (sizeof(Type)==sizeof(double) ? kernels[dv_dot] : kernels[fv_dot]);
		kdot.setArg(0, alpha);
		kdot.setArg(1, a);
		kdot.setArg(2, n);
		kdot.setArg(3, beta);
		kdot.setArg(4, b);
		kdot.setArg(5, partial);
		execute(kdot, cl::NullRange, cl::NDRange(workgroups*WS), cl::NDRange(WS));

		cl::Kernel& ksve = (sizeof(Type)==sizeof(double) ? kernels[dv_sve] : kernels[fv_sve]);
		ksve.setArg(0, partial);
		ksve.setArg(1, workgroups);
		ksve.setArg(2, y);
		execute(ksve, cl::NullRange, cl::NDRange(1), cl::NullRange);		
	}



	//
	// Matrices
	//

	template <typename Type>
	void mat_set(cl::Buffer& y, int m, int n, int pitch, Type alpha) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_seta] : kernels[fm_seta]);
		kernel.setArg(0, y);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, alpha);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}
	
	template <typename Type>
	void mat_set(cl::Buffer& y, int m, int n, int apitch, Type alpha, const cl::Buffer& x, int xpitch) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_setax] : kernels[fm_setax]);
		kernel.setArg(0, y);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, apitch);
		kernel.setArg(4, alpha);
		kernel.setArg(5, x);
		kernel.setArg(6, xpitch);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}
	
	template <typename Type>
	void mat_plus(cl::Buffer& y, int m, int n, int pitch, Type alpha) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_mpa] : kernels[fm_mpa]);
		kernel.setArg(0, y);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, alpha);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}
	
	template <typename Type>
	void mplusv(cl::Buffer& y, int m, int n, int pitch, Type alpha, const cl::Buffer& x, int axis) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_mpax] : kernels[fm_mpax]);
		kernel.setArg(0, y);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, alpha);
		kernel.setArg(5, x);
		kernel.setArg(6, axis);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}
	
	template <typename Type>
	void mat_mul(cl::Buffer& y, int m, int n, int pitch, Type alpha) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_mma] : kernels[fm_mma]);
		kernel.setArg(0, y);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, alpha);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}
	
	template <typename Type>
	void axpy(Type alpha, const cl::Buffer& x, int m, int n, int xpitch, cl::Buffer& y, int ypitch) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_axpy] : kernels[fm_axpy]);
		kernel.setArg(0, alpha);
		kernel.setArg(1, x);
		kernel.setArg(2, m);
		kernel.setArg(3, n);
		kernel.setArg(4, xpitch);
		kernel.setArg(5, y);
		kernel.setArg(6, ypitch);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}
	
	template <typename Type>
	void zaxpby(Type alpha, const cl::Buffer& x, int m, int n, int xpitch, 
				Type beta, const cl::Buffer& y, int ypitch, cl::Buffer& z, int zpitch) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_zaxpby] : kernels[fm_zaxpby]);
		kernel.setArg(0, alpha);
		kernel.setArg(1, x);
		kernel.setArg(2, m);
		kernel.setArg(3, n);
		kernel.setArg(4, xpitch);
		kernel.setArg(5, beta);
		kernel.setArg(6, y);
		kernel.setArg(7, ypitch);
		kernel.setArg(8, z);
		kernel.setArg(9, zpitch);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}
	
	template <typename Type>
	void hadamard(const cl::Buffer& a, const cl::Buffer& b, cl::Buffer& c, int m, int n, int pitch) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_hadamard] : kernels[fm_hadamard]);
		kernel.setArg(0, a);
		kernel.setArg(1, b);
		kernel.setArg(2, c);
		kernel.setArg(3, m);
		kernel.setArg(4, n);
		kernel.setArg(5, pitch);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}

	template <typename Type>
	void prod_vec(cl::Buffer& a, int m, int n, int pitch, const cl::Buffer& v, int axis) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_mmulv] : kernels[fm_mmulv]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, v);
		kernel.setArg(5, axis);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}

	template <typename Type>
	void div_vec(cl::Buffer& a, int m, int n, int pitch, const cl::Buffer& v, int axis) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_mdivv] : kernels[fm_mdivv]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, v);
		kernel.setArg(5, axis);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}

	template <typename Type>
	void mat_func(cl::Buffer& y, int m, int n, int pitch, int f, Type beta=1, Type alpha=0, const cl::Buffer& x=NullBuffer, int axis=0) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_fmpax] : kernels[fm_fmpax]);
		kernel.setArg(0, y);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, f);
		kernel.setArg(5, beta);
		kernel.setArg(6, alpha);
		kernel.setArg(7, x);
		kernel.setArg(8, axis);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}

	template <typename Type>
	void argmax(const cl::Buffer& a, int m, int n, int pitch, cl::Buffer& idcs) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_argmax] : kernels[fm_argmax]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, idcs);
		execute(kernel, cl::NullRange, cl::NDRange(m), cl::NullRange);
	}

	template <typename Type>
	void reduce_max2d(const cl::Buffer& a, int m, int n, int pitch, cl::Buffer& out, int axis) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_matmax] : kernels[fm_matmax]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, out);
		kernel.setArg(5, axis);
		execute(kernel, cl::NullRange, cl::NDRange(axis==0 ? n:m), cl::NullRange);
	}

	template <typename Type>
	void argmin(const cl::Buffer& a, int m, int n, int pitch, cl::Buffer& idcs) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_argmin] : kernels[fm_argmin]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, idcs);
		execute(kernel, cl::NullRange, cl::NDRange(m), cl::NullRange);
	}

	template <typename Type>
	void reduce_min2d(const cl::Buffer& a, int m, int n, int pitch, cl::Buffer& out, int axis) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_matmin] : kernels[fm_matmin]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, out);
		kernel.setArg(5, axis);
		execute(kernel, cl::NullRange, cl::NDRange(axis==0 ? n:m), cl::NullRange);
	}

/*
	template <typename Type>
	void reduce_sum2d(const cl::Buffer& a, int m, int n, int pitch, cl::Buffer& rows, int axis) {
		cl::Buffer aa = a;
		if (axis==0) {
			// transpose matrix 'a'
			aa = alloc<Type>(DIMPAD(n)*DIMPAD(m));
			//queue.enqueueFillBuffer(aa, Type(0), 0, DIMPAD(n)*DIMPAD(m)*sizeof(Type));
			gemt<Type>(a, m, n, pitch, aa, DIMPAD(m));
			std::swap(m, n);
			pitch = DIMPAD(n);
		}
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer rows_partial = alloc<Type>(DIMPAD(m)*workgroups);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_smrp] : kernels[fm_smrp]);
		kernel.setArg(0, aa);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, rows_partial);
		execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS,DIMPAD(m)), cl::NDRange(WS));

		cl::Buffer partial = alloc<Type>(DIMPAD(m));
		cl::Kernel& ksmr = (sizeof(Type)==sizeof(double) ? kernels[dm_smr] : kernels[fm_smr]);
		ksmr.setArg(0, rows_partial);
		ksmr.setArg(1, m);
		ksmr.setArg(2, workgroups);
		ksmr.setArg(3, workgroups);
		ksmr.setArg(4, rows);
		execute(ksmr, cl::NullRange, cl::NDRange(DIMPAD(m)), cl::NullRange);
	}
*/
	template <typename Type>
	void reduce_sum2d(const cl::Buffer& a, int m, int n, int pitch, cl::Buffer& out, int axis) {
		if (axis==0) {
			cl::Kernel& ksmc = (sizeof(Type)==sizeof(double) ? kernels[dm_smc] : kernels[fm_smc]);
			ksmc.setArg(0, a);
			ksmc.setArg(1, m);
			ksmc.setArg(2, n);
			ksmc.setArg(3, pitch);
			ksmc.setArg(4, out);
			execute(ksmc, cl::NullRange, cl::NDRange(TILEPAD(n)), cl::NullRange);
		} else {
			const int WS = 256;
			const int workgroups = (n+WS-1) / WS;
			cl::Buffer rows_partial = alloc<Type>(TILEPAD(m)*workgroups);
			cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_smrp] : kernels[fm_smrp]);
			kernel.setArg(0, a);
			kernel.setArg(1, m);
			kernel.setArg(2, n);
			kernel.setArg(3, pitch);
			kernel.setArg(4, rows_partial);
			execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS,TILEPAD(m)), cl::NDRange(WS));

			cl::Buffer partial = alloc<Type>(TILEPAD(m));
			cl::Kernel& ksmr = (sizeof(Type)==sizeof(double) ? kernels[dm_smr] : kernels[fm_smr]);
			ksmr.setArg(0, rows_partial);
			ksmr.setArg(1, m);
			ksmr.setArg(2, workgroups);
			ksmr.setArg(3, workgroups);
			ksmr.setArg(4, out);
			execute(ksmr, cl::NullRange, cl::NDRange(TILEPAD(m)), cl::NullRange);
		}
	}

	template <typename Type>
	void sme_full(const cl::Buffer& a, int m, int n, int pitch, cl::Buffer& sum) {
		cl::Buffer rows_partial = alloc<Type>(TILEPAD(m));
		cl::Kernel& ksmr = (sizeof(Type)==sizeof(double) ? kernels[dm_smr] : kernels[fm_smr]);
		ksmr.setArg(0, a);
		ksmr.setArg(1, m);
		ksmr.setArg(2, n);
		ksmr.setArg(3, pitch);
		ksmr.setArg(4, rows_partial);
		execute(ksmr, cl::NullRange, cl::NDRange(TILEPAD(m)), cl::NullRange);
		sve<Type>(rows_partial, m, sum);
	}

	template <typename Type>
	void sme(const cl::Buffer& a, int m, int n, int pitch, cl::Buffer& sum) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer rows_partial = alloc<Type>(TILEPAD(m)*workgroups);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_smrp] : kernels[fm_smrp]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, rows_partial);
		execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS,TILEPAD(m)), cl::NDRange(WS));

		cl::Buffer partial = alloc<Type>(TILEPAD(m));
		cl::Kernel& ksmr = (sizeof(Type)==sizeof(double) ? kernels[dm_smr] : kernels[fm_smr]);
		ksmr.setArg(0, rows_partial);
		ksmr.setArg(1, m);
		ksmr.setArg(2, workgroups);
		ksmr.setArg(3, workgroups);
		ksmr.setArg(4, partial);
		execute(ksmr, cl::NullRange, cl::NDRange(TILEPAD(m)), cl::NullRange);

		sve<Type>(partial, m, sum);
	}

	template <typename Type>
	void sme2(const cl::Buffer& a, int m, int n, int pitch, Type alpha, cl::Buffer& sum) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer rows_partial = alloc<Type>(TILEPAD(m)*workgroups);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_smrp2] : kernels[fm_smrp2]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, pitch);
		kernel.setArg(4, alpha);
		kernel.setArg(5, rows_partial);
		execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS,TILEPAD(m)), cl::NDRange(WS));

		cl::Buffer partial = alloc<Type>(TILEPAD(m));
		cl::Kernel& ksmr = (sizeof(Type)==sizeof(double) ? kernels[dm_smr] : kernels[fm_smr]);
		ksmr.setArg(0, rows_partial);
		ksmr.setArg(1, m);
		ksmr.setArg(2, workgroups);
		ksmr.setArg(3, workgroups);
		ksmr.setArg(4, partial);
		execute(ksmr, cl::NullRange, cl::NDRange(TILEPAD(m)), cl::NullRange);

		sve<Type>(partial, m, sum);
	}

	template <typename Type>
	void count_equal(const cl::Buffer& a, int m, int n, int apitch, 
					 const cl::Buffer& b, int bpitch, Type novalue, cl::Buffer& cnt) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer rows_partial = alloc<int>(TILEPAD(m)*workgroups);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_cnteq] : kernels[fm_cnteq]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, apitch);
		kernel.setArg(4, b);
		kernel.setArg(5, bpitch);
		kernel.setArg(6, novalue);
		kernel.setArg(7, rows_partial);
		execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS,TILEPAD(m)), cl::NDRange(WS));

		cl::Buffer partial = alloc<int>(TILEPAD(m));
		kernels[im_smr].setArg(0, rows_partial);
		kernels[im_smr].setArg(1, m);
		kernels[im_smr].setArg(2, workgroups);
		kernels[im_smr].setArg(3, workgroups);
		kernels[im_smr].setArg(4, partial);
		execute(kernels[im_smr], cl::NullRange, cl::NDRange(TILEPAD(m)), cl::NullRange);

		kernels[iv_sve].setArg(0, partial);
		kernels[iv_sve].setArg(1, m);
		kernels[iv_sve].setArg(2, cnt);
		execute(kernels[iv_sve], cl::NullRange, cl::NDRange(1), cl::NullRange);
	}

	template <typename Type>
	void dist_squared(const cl::Buffer& a, int m, int n, int apitch, 
					  const cl::Buffer& b, int bpitch, cl::Buffer& dist) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer rows_partial = alloc<Type>(TILEPAD(m)*workgroups);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_eucl2] : kernels[fm_eucl2]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, apitch);
		kernel.setArg(4, b);
		kernel.setArg(5, bpitch);
		kernel.setArg(6, rows_partial);
		execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS,TILEPAD(m)), cl::NDRange(WS));
		sme_full<Type>(rows_partial, m, workgroups, workgroups, dist);
	}

	template <typename Type>
	void manhattan(const cl::Buffer& a, int m, int n, int apitch, 
				   const cl::Buffer& b, int bpitch, cl::Buffer& dist) {
		const int WS = 256;
		const int workgroups = (n+WS-1) / WS;
		cl::Buffer rows_partial = alloc<Type>(TILEPAD(m)*workgroups);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_manh] : kernels[fm_manh]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, apitch);
		kernel.setArg(4, b);
		kernel.setArg(5, bpitch);
		kernel.setArg(6, rows_partial);
		execute(kernel, cl::NullRange, cl::NDRange(workgroups*WS,TILEPAD(m)), cl::NDRange(WS));
		sme_full<Type>(rows_partial, m, workgroups, workgroups, dist);
	}
	
	template <typename Type>
	void outer(cl::Buffer& z, int pitch, const cl::Buffer& x, int m, const cl::Buffer& y, int n) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_outer] : kernels[fm_outer]);		
		kernel.setArg(0, z);
		kernel.setArg(1, pitch);
		kernel.setArg(2, x);
		kernel.setArg(3, m);
		kernel.setArg(4, y);
		kernel.setArg(5, n);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}
	
	template <typename Type>
	void gemv1(Type alpha, const cl::Buffer& a, int m, int n, int pitch, 
			   const cl::Buffer& x, Type beta, cl::Buffer& y) {
		const int TS = TILES;
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_gemv1] : kernels[fv_gemv1]);
		kernel.setArg(0, alpha);
		kernel.setArg(1, a);
		kernel.setArg(2, m);
		kernel.setArg(3, n);
		kernel.setArg(4, pitch);
		kernel.setArg(5, x);
		kernel.setArg(6, beta);
		kernel.setArg(7, y);
		int GlobalWorkSize = TS;
		execute(kernel, cl::NullRange, cl::NDRange(GlobalWorkSize), cl::NDRange(TS));
	}
	
	template <typename Type>
	void gemv2(Type alpha, const cl::Buffer& a, int m, int n, int pitch, 
			   const cl::Buffer& x, Type beta, cl::Buffer& y) {
		const int TS = 16;
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dv_gemv2] : kernels[fv_gemv2]);
		kernel.setArg(0, alpha);
		kernel.setArg(1, a);
		kernel.setArg(2, m);
		kernel.setArg(3, n);
		kernel.setArg(4, pitch);
		kernel.setArg(5, x);
		kernel.setArg(6, beta);
		kernel.setArg(7, y);
		execute(kernel, cl::NullRange, cl::NDRange(TS,TILEPAD(m)), cl::NDRange(TS));
	}
	
	template <typename Type>
	void gemm(const cl::Buffer& a, const cl::Buffer& b, cl::Buffer& c, 
			  int m, int n, int p, int apitch, int bpitch, int cpitch) {
		const int TS = TILES;
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_gemm] : kernels[fm_gemm]);
		kernel.setArg(0, a);
		kernel.setArg(1, b);
		kernel.setArg(2, c);
		kernel.setArg(3, m);
		kernel.setArg(4, n);
		kernel.setArg(5, p);
		kernel.setArg(6, apitch);
		kernel.setArg(7, bpitch);
		kernel.setArg(8, cpitch);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(p),TILEPAD(m)), cl::NDRange(TS,TS));
		//execute(kernel, cl::NullRange, cl::NDRange(p,m), cl::NullRange);
	}
	
	template <typename Type>
	void gemt(const cl::Buffer& a, int m, int n, int apitch, cl::Buffer& t, int tpitch) {
		const int TS = TILES;
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_gemt] : kernels[fm_gemt]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, apitch);
		kernel.setArg(4, t);
		kernel.setArg(5, tpitch);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NDRange(TS,TS)); 
		//execute(kernels[fm_gemt], cl::NullRange, cl::NDRange(DIMPAD(n),DIMPAD(m)), cl::NullRange); 
	}
	
	template <typename Type>
	void gram(const cl::Buffer& a, int m, int n, int apitch, cl::Buffer& c, int cpitch) {
		const int TS = TILES;
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_gram] : kernels[fm_gram]);
		kernel.setArg(0, a);
		kernel.setArg(1, m);
		kernel.setArg(2, n);
		kernel.setArg(3, apitch);
		kernel.setArg(4, c);
		kernel.setArg(5, cpitch);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NDRange(TS,TS));
	}

	template <typename Type>
	void bmm(const cl::Buffer& a, const cl::Buffer& b, cl::Buffer& c, int bs, int m, int n, int p, 
			 int apitch, int bpitch, int cpitch, int astride, int bstride, int cstride) {
		const int TS = 6;
		const int shmem_size = 2*TS*n*sizeof(Type);
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dm_bmm] : kernels[fm_bmm]);
		kernel.setArg( 0, a);
		kernel.setArg( 1, b);
		kernel.setArg( 2, c);
		kernel.setArg( 3, bs);
		kernel.setArg( 4, m);
		kernel.setArg( 5, n);
		kernel.setArg( 6, p);
		kernel.setArg( 7, apitch);
		kernel.setArg( 8, bpitch);
		kernel.setArg( 9, cpitch);
		kernel.setArg(10, astride);
		kernel.setArg(11, bstride);
		kernel.setArg(12, cstride);
		kernel.setArg(13, shmem_size);
		//execute(kernel, cl::NullRange, cl::NDRange(PAD<TS>(p),PAD<TS>(m),bs), cl::NDRange(TS,TS));
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(p),TILEPAD(m),bs), cl::NDRange(TS,TS));
		//execute(kernel, cl::NullRange, cl::NDRange(p,m,bs), cl::NullRange);
	}


	//
	// Cubes
	//

	template <typename Type>
	void cub_set(cl::Buffer& y, int c, int m, int n, int xpitch, int ypitch, 
				 Type alpha, const cl::Buffer& x, int xxpitch, int xypitch) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dc_setax] : kernels[fc_setax]);
		kernel.setArg(0, y);
		kernel.setArg(1, c);
		kernel.setArg(2, m);
		kernel.setArg(3, n);
		kernel.setArg(4, xpitch);
		kernel.setArg(5, ypitch);
		kernel.setArg(6, alpha);
		kernel.setArg(7, x);
		kernel.setArg(8, xxpitch);
		kernel.setArg(9, xypitch);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m),c), cl::NullRange);
	}

	template <typename Type>
	void cub_func(cl::Buffer& y, int h, int m, int n, int pitch, int zstride, int f) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dc_func3d] : kernels[fc_func3d]);
		kernel.setArg(0, y);
		kernel.setArg(1, h);
		kernel.setArg(2, m);
		kernel.setArg(3, n);
		kernel.setArg(4, pitch);
		kernel.setArg(5, zstride);
		kernel.setArg(6, f);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m),h), cl::NullRange);
	}

	template <typename Type>
	void reduce_sum3d(const cl::Buffer& in, int h, int m, int n, int xpitch, int ypitch, cl::Buffer& out) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dc_sum3d] : kernels[fc_sum3d]);
		kernel.setArg(0, in);
		kernel.setArg(1, h);
		kernel.setArg(2, m);
		kernel.setArg(3, n);
		kernel.setArg(4, xpitch);
		kernel.setArg(5, ypitch);
		kernel.setArg(6, out);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m)), cl::NullRange);
	}

	template <typename Type>
	void outer3d(cl::Buffer& out, int h, int zstride, int pitch, 
				 const cl::Buffer& v, int m, int vzstride, const cl::Buffer& u, int n, int uzstride) {
		cl::Kernel& kernel = (sizeof(Type)==sizeof(double) ? kernels[dc_outer3d] : kernels[fc_outer3d]);
		kernel.setArg(0, out);
		kernel.setArg(1, h);
		kernel.setArg(2, zstride);
		kernel.setArg(3, pitch);
		kernel.setArg(4, v);
		kernel.setArg(5, m);
		kernel.setArg(6, vzstride);
		kernel.setArg(7, u);
		kernel.setArg(8, n);
		kernel.setArg(9, uzstride);
		execute(kernel, cl::NullRange, cl::NDRange(TILEPAD(n),TILEPAD(m),h), cl::NullRange);
	}
};


// all functions should be called via __ocl__. 
static __OCL __ocl__;

};     // namespace umml

#endif // UMML_OCL_INCLUDED
