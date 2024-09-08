#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "kmeans.hpp"
#include "pca.hpp"
#include "glplot.hpp"
#include "datasets/mnistloader.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }

#define CLUSTERS_FNAME "mnist_kmeans_clusters.dat"

// Datatype for data and neural network
typedef float dtype;

// Seed for RNG
constexpr int SEED = 48;

// Number of clusters for KMeans
constexpr int NClusters = 10;


// -------------- Main -------------------

int main() 
{
	umat<dtype> X, X_valid, X_test;
	umat<dtype> Y, Y_valid, Y_test;
	uvec<int> y, y_valid, y_test;
	uvec<int> clustering;
	steady_clock::time_point t1, t2;

	// set openmp threads to maximum
	umml_set_openmp_threads();

	// seed RNG
	umml_seed_rng(SEED);

	// load mnist
	string path = "../data/MNIST/";
	string train_images = path + "train-images-idx3-ubyte";
	string train_labels = path + "train-labels-idx1-ubyte";
	string test_images  = path + "t10k-images-idx3-ubyte";
	string test_labels  = path + "t10k-labels-idx1-ubyte";
	MNISTloader mnist;
	mnist.load_images(train_images, X);
	mnist.load_labels(train_labels, y);
	mnist.load_images(test_images, X_test);
	mnist.load_labels(test_labels, y_test);
	if (!mnist) {
		cout << "Error loading MNIST: " << mnist.error_description() << "\n";
		return -1;
	} else {
		cout << "Loaded " << X.ydim() << " training images, " << y.len() << " labels.\n";
		cout << "Loaded " << X_test.ydim() << " test images, " << y_test.len() << " labels.\n";
	}

	// split training/validation sets
	dataframe df;
	X_valid = df.copy_rows(X, 50000, 10000);
	y_valid = df.copy(y, 50000, 10000);
	X.reshape(50000, X.xdim());
	y.reshape(50000);

	// scale data
	X.mul(1.0/255);
	X.plus(-0.5);
	X_valid.mul(1.0/255);
	X_valid.plus(-0.5);
	X_test.mul(1.0/255);
	X_test.plus(-0.5);


	// Kmeans clustering
	KMeans<dtype> km;
	{
	KMeans<dtype>::params opt;
	opt.info_iters = 10;
	km.set_params(opt);
	}
	bool random_init = true;
	bool recluster = true;
	if (km.load("mnist_kmeans_clusters.dat") && km.n_clusters()==NClusters) {
		cout << km.n_clusters() << " clusters found in the disk file.\n";
		recluster = false;
	} else {
		cout << km.n_clusters() << "Clusters not found. Do you want to do a reclustering (y/n)? ";
		string answer;
		cin >> answer;
		if (answer[0]=='n' || answer[0]=='N') {
			cout << "Nothing more to do.\n";
			return 0;
		}
	}
	if (recluster) {
		if (!random_init) {
			umat<dtype> X_seed(NClusters, X.xdim());
			for (int i=0; i<NClusters; ++i) {
				int digit = i % 10;
				int pos = y.find_random(digit);
				X_seed.set_row(i, X.row_offset(pos), X_seed.xdim());
			}
			km.seed(NClusters, X_seed);
		}
		t1 = chrono::steady_clock::now();
		clustering = km.fit(NClusters, X);
		t2 = chrono::steady_clock::now();
		cout << NClusters << " clusters created in " << format_duration(t1, t2) << ".\n";
		km.save("mnist_kmeans_clusters.dat");
	} else {
		clustering = km.cluster(X);
	}
	umat<dtype> centr;
	uvec<int> spc;
	km.clustering_info(centr, spc);
	cout << "Samples per cluster: " << spc.format() << "\n";


	// PCA dimentionality reduction
	umat<dtype> Xr;
	PCA<dtype> pca(3);
	pca.fit_transform(X, Xr);


	// plot
	glplot plt;
	plt.set_window_geometry(1024,768);
	plt.set_window_title("MNIST clustering");
	vector<glplot::point2d> graph[NClusters];
	std::vector<int> colors = glplot::make_palette(NClusters, 10, false);
	std::vector<int> palette16 = glplot::make_palette16();
	for (int sample=0; sample<Xr.ydim(); ++sample) {
		glplot::point2d pt;
		pt.x = (float)Xr(sample,0);
		pt.y = -(float)Xr(sample,1);
		//pt.z = (float)Xr(sample,2);
		graph[clustering(sample)].push_back(pt);
	}
	plt.set_bkgnd_color(glplot::White);
	for (int i=0; i<NClusters; ++i) {
		plt.add_2d(graph[i], colors[i], glplot::Scatter, 5);
		//plt.add_2d(graph[i], palette16[i%16], glplot::Scatter, 5);
	}
	plt.show();

	return 0;
}
