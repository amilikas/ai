#ifndef UMML_MAXPOOL_CPU_INCLUDED
#define UMML_MAXPOOL_CPU_INCLUDED


namespace umml {


template <typename Type>
void cpu_maxpool2d(const Type* a, int Z, int M, int N, int xpitch, int zstride, 
				   int k, int stride, Type* c, int cxpitch, int czstride)
{
	int h = pooling_size(M,k,stride);
	int w = pooling_size(N,k,stride);
	for (int z=0; z<Z; ++z) {
		int px=0, py=0;
		for (int i=0; i<h; ++i) {
			for (int j=0; j<w; ++j) {
				Type max = a[z*zstride + py*xpitch + px];
				for (int ki=0; ki<k; ++ki)
				for (int kj=0; kj<k; ++kj) {
					Type val = a[z*zstride + (py+ki)*xpitch + px+kj];
					if (val > max) max = val;
				}
				c[z*czstride + i*cxpitch + j] = max;
				px += stride;
			}
			py += stride; 
			px = 0;
		}
	}
}

template <typename Type>
void cpu_maxpool2d(const Type* a, int Z, int M, int N, int xpitch, int zstride, 
				   int k, int stride, Type* c, int cxpitch, int czstride, int* idcs)
{
	int h = pooling_size(M,k,stride);
	int w = pooling_size(N,k,stride);
	for (int z=0; z<Z; ++z) {
		int px=0, py=0;
		for (int i=0; i<h; ++i) {
			for (int j=0; j<w; ++j) {
				int mloc = z*zstride + py*xpitch + px;
				Type max = a[mloc];
				for (int ki=0; ki<k; ++ki)
				for (int kj=0; kj<k; ++kj) {
					int loc = z*zstride + (py+ki)*xpitch + px+kj;
					Type val = a[loc];
					if (val > max) {
						max  = val;
						mloc = loc;
					}
				}
				c[z*czstride + i*cxpitch+j] = max;
				idcs[z*h*w + i*w + j] = mloc;
				px += stride;
			}
			py += stride; 
			px = 0;
		}
	}
}


};     // namespace umml

#endif // UMML_MAXPOOL_CPU_INCLUDED
