// MNIST in-class clustering

#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "kmeans.hpp"
#include "pca.hpp"
#include "glplot.hpp"
//#include "nn/backprop.hpp"
//#include "bio/ga.hpp"
#include "datasets/mnistloader.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }

#define SAVES_FOLDER         "../../saves/mnist/clustering/"
#define LOG_NAME             "mnistcluster.log"
#define CLUSTERING_FNAME     "mnist_inclass_clusters_"


// Data type for datasets
typedef float dtype;

// Seed for local RNG
constexpr int SEED = 48;



// -------------- Main -------------------

int main() 
{
	umat<dtype> X, X_valid, X_test;
	uvec<int> y, y_valid, y_test;
	steady_clock::time_point t1, t2;

	Logger log(LOG_NAME);

	// set openmp threads to maximum
	umml_set_openmp_threads();

	// seed RNG
	umml_seed_rng(SEED);

	// load mnist
	string path = "../../../auth/data/MNIST/";
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

	log << umml_compute_info(false) << "\n";
	log << "Train data: X " << X.shape() << " " << X.bytes() << ", y " << y.shape() << "\n";
	log << "Validation data: X " << X_valid.shape() << " " << X_valid.bytes() << ", y " << y_valid.shape() << "\n";
	log << "Testing data: X " << X_test.shape() << " " << X_test.bytes() << ", y " << y_test.shape() << "\n";



	DO_TEST(true)

	// ========================
	// Clustering (KMeans)
	// ========================

	vector<int> NClusters = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
	string answer = " ";
	for (int digit=0; digit <=9; ++digit) {
		// select digit and make new datasets
		uvec<int> clustering;
		umat<dtype> X_digit;
		uvec<int> y_digit;
		uvec<int> idcs;
		idcs = df.select(y, digit);
		X_digit = df.vstack(X_digit, df.copy_rows(X, idcs));
		y_digit = df.append(y_digit, df.copy(y, idcs));

		// in-class clustering with kmeans
		KMeans<dtype> km;
		{
		KMeans<dtype>::params opt;
		opt.info_iters = 10;
		km.set_params(opt);
		}
		bool recluster = true;
		string fname = string(SAVES_FOLDER)+CLUSTERING_FNAME+to_string(digit)+".dat";
		cout << "Digit " << digit << ": ";
		if (km.load(fname) && km.n_clusters()==NClusters[digit]) {
			cout << km.n_clusters() << " clusters found in the disk file.\n";
			recluster = false;
		} else {
			cout << "clusters not found (expecting " << NClusters[digit] << ", found "
			 	 << km.n_clusters() << ").\n";
			if (!(answer[0]=='n' || answer[0]=='N' || answer[0]=='y' || answer[0]=='Y')) {
			 	cout << "Do you want to do a reclustering (y/n)? ";
				cin >> answer;
				if (answer[0]=='n' || answer[0]=='N') {
					cout << "Nothing more to do.\n";
					return 0;
				}
			}
		}
		if (recluster) {
			km.unitialize();
			log << "Clustering...\n";
			t1 = chrono::steady_clock::now();
			clustering = km.fit(NClusters[digit], X_digit);
			log << "clustering.len=" << clustering.len() << "\n";
			t2 = chrono::steady_clock::now();
			log << NClusters[digit] << " clusters created in " << format_duration(t1, t2) << ".\n";
			km.save(fname);
		} else {
			clustering = km.cluster(X_digit);
		}

		{
		umat<dtype> centr;
		uvec<int> spc;
		km.clustering_info(centr, spc);
		log << "Samples per cluster: " << spc.format() << 
			//" (silhouette coefficient: " << silhouette_coefficient(X_digit, centr) << 
			"\n\n";
		}

		// PCA dimentionality reduction
		umat<dtype> Xr;
		PCA<dtype> pca(2);
		pca.fit_transform(X_digit, Xr);

		// plot
		glplot plt;
		std::string title = "MNIST in-class clustering for " + std::to_string(digit);
		plt.set_window_geometry(1024,768);
		plt.set_window_title(title);
		vector<glplot::point2d> graph[NClusters[digit]];
		std::vector<int> palette16 = glplot::make_palette16();
		for (int sample=0; sample<Xr.ydim(); ++sample) {
			glplot::point2d pt;
			pt.x = (float)Xr(sample,0);
			pt.y = -(float)Xr(sample,1);
			//pt.z = (float)Xr(sample,2);
			graph[clustering(sample)].push_back(pt);
		}
		plt.set_bkgnd_color(glplot::White);
		for (int i=0; i<NClusters[digit]; ++i) {
			//plt.add_2d(graph[i], colors[i], glplot::Scatter, 5);
			plt.add_2d(graph[i], palette16[i%16], glplot::Scatter, 5);
		}
		plt.show();
		
	}

	END_TEST



	return 0;
}
