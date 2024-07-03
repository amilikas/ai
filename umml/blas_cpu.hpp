namespace umml {

// reciprocal
// y = α(1/x)
template <typename Type>
void cpu_reciprocal(Type alpha, const Type* x, int n, Type* y) 
{
	for (int i=0; i<n; ++i) y[i] = (x[i] != 0 ? alpha/x[i] : 0);
}


// axpy : vector(n) += scalar*vector(n)
// y = αx + y
template <typename Type>
void cpu_axpy(Type alpha, const Type* x, int n, Type* y) 
{
	for (int i=0; i<n; ++i) y[i] += alpha*x[i];
}


// zaxpby : vector(n) = scalar*vector(n) + scalar*vector(n)
// z = αx + βy
template <typename Type>
void cpu_zaxpby(Type alpha, const Type* x, int n, Type beta, const Type* y, Type* z) 
{
	for (int i=0; i<n; ++i) z[i] = alpha*x[i] + beta*y[i];
}

// axpy : matrix(m,n) += scalar*matrix(m,n)
// Y = αX + Y
template <typename Type>
void cpu_axpy(Type alpha, const Type* x, int m, int n, int xpitch, Type* y, int ypitch) 
{
	for (int i=0; i<m; ++i)
	for (int j=0; j<n; ++j) y[i*ypitch+j] += alpha*x[i*xpitch+j];
}


// zaxpby : matrix(m,n) = scalar*matrix(m,n) + scalar*matrix(m,n)
// Z = αX + βY
template <typename Type>
void cpu_zaxpby(Type alpha, const Type* x, int m, int n, int xpitch, Type beta, const Type* y, int ypitch, Type* z, int zpitch) 
{
	for (int i=0; i<m; ++i)
	for (int j=0; j<n; ++j) z[i*zpitch+j] = alpha*x[i*xpitch+j] + beta*y[i*ypitch+j];
}

// dot : scalar = vector(n) * vector(n)
// dot = x.y
template <typename Type>
Type cpu_dot(const Type* x, int n, const Type* y) 
{
	Type sum = 0;
	for (int i=0; i<n; ++i) sum += x[i] * y[i];
	return sum;
}


// gemv : vector(m) = scalar*matrix(m,n)*vector(n) + scalar*vector(m)
// y = αA*x + βy
template <typename Type>
void cpu_gemv(Type alpha, const Type* a, int m, int n, int pitch, const Type* x, Type beta, Type* y) 
{
	for (int i=0; i<m; ++i) {
		Type sum = 0;
		for (int j=0; j<n; ++j) sum += alpha*a[i*pitch+j] * x[j];
		y[i] = sum + (beta==0 ? 0 : beta*y[i]);
	}
}


// gemm : matrix(m,p) = matrix(m,n) * matrix(n,p)
// C = A*B
template <typename Type>
void cpu_gemm(const Type* a, const Type* b, Type* c, int m, int n, int p, int apitch, int bpitch, int cpitch) 
{
	#pragma omp parallel for 
	for (int i=0; i<m; ++i) 
	for (int k=0; k<n; ++k) {
		Type preload = a[i*apitch+k]; 
		for (int j=0; j<p; ++j) c[i*cpitch+j] += preload * b[k*bpitch+j];
	}
}


// M = v x u (vector outer product)
template <typename Type>
void cpu_outer(Type* out, int pitch, const Type* v, int m, const Type* u, int n)
{
	for (int i=0; i<m; ++i)
	for (int j=0; j<n; ++j) out[i*pitch+j] = v[i]*u[j];
}

// M_z = [v x u]_z (z vector outer product)
template <typename Type>
void cpu_outer3d(Type* out, int h, int zstride, int pitch, 
				 const Type* v, int m, int vzstride, const Type* u, int n, int uzstride)
{
	#pragma omp parallel for 
	for (int z=0; z<h; ++z)
	for (int i=0; i<m; ++i)
	for (int j=0; j<n; ++j) out[z*zstride+i*pitch+j] = v[z*vzstride+i]*u[z*uzstride+j];
}


// gramm : A(n,n) = A(m,n).T * A(m,n) = A'(n,m) * A(m,n)
// https://en.wikipedia.org/wiki/In-place_matrix_transposition
template <typename Type>
void cpu_gram(const Type* a, int m, int n, int apitch, Type* c, int cpitch) 
{
	#pragma omp parallel for
	for (int k=0; k<m; ++k) 
	for (int i=0; i<n; ++i) 
	for (int j=0; j<n; ++j) c[i*cpitch+j] += a[k*apitch+i] * a[k*apitch+j];
}


template <typename Type>
void cpu_gemt(const Type* a, int m, int n, int apitch, Type* t, int tpitch) 
{
	for (int j=0; j<n; ++j)
	for (int i=0; i<m; ++i) t[j*tpitch+i] = a[i*apitch+j];
}


}; // namespace umml
