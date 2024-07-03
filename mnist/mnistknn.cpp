#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "knn.hpp"
#include "kmeans.hpp"
#include "clustermap.hpp"
#include "datasets/mnistloader.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }

// Datatype for data and neural network
typedef float dtype;

int main() 
{
	umat<dtype> X, X_train, X_valid, X_test;
	umat<dtype> Y_train_1hot, Y_valid_1hot, Y_test_1hot;
	uvec<int> y, y_train, y_valid, y_test, y_unmod, y_pred, y_train_pred;
	steady_clock::time_point t1, t2;
	confmat<> cm;
	int cm_mode = CM_Macro;

	Logger log("mnist-knn.log");

	// set openmp threads to maximum
	umml_set_openmp_threads();

	// load mnist
	bool load_validation = true;
	string path = "../../../auth/data/MNIST/";
	string original = "train";
	string modified = "modified/ng2k";
	string unmodified = "modified/ng2k-unmod";
	string train_images = path + original + "-images-idx3-ubyte";
	string train_labels = path + original + "-labels-idx1-ubyte";
	string unmod_labels = path + unmodified + "-labels-idx1-ubyte";
	string valid_images = path + "modified/v5k-images-idx3-ubyte";
	string valid_labels = path + "modified/v5k-labels-idx1-ubyte";
	string test_images  = path + "t10k-images-idx3-ubyte";
	string test_labels  = path + "t10k-labels-idx1-ubyte";
	MNISTloader mnist;
	mnist.load_images(train_images, X);
	mnist.load_labels(train_labels, y);
	mnist.load_images(test_images, X_test);
	mnist.load_labels(test_labels, y_test);
	mnist.load_labels(unmod_labels, y_unmod);
	if (load_validation) {
		mnist.load_images(valid_images, X_valid);
		mnist.load_labels(valid_labels, y_valid);
	}
	if (!mnist) {
		cout << "Error loading MNIST: " << mnist.error_description() << "\n";
		return -1;
	} else {
		cout << "Loaded " << X.ydim() << " training images, " << y.len() << " labels.\n";
		cout << "Loaded " << X_test.ydim() << " test images, " << y_test.len() << " labels.\n";
		if (load_validation)		
			cout << "Loaded " << X_valid.ydim() << " validation images, " << y_valid.len() << " labels.\n";
	}

	// Dataframe
	if (false) {
		dataframe df;
		uvec<int> idcs = df.select_equal(y, y_unmod);
		X_train = df.copy_rows(X, idcs);
		y_train = df.copy(y, idcs);	
	} else {
		X_train.resize_like(X); 
		X_train.set(X);
		y_train.resize_like(y); 
		y_train.set(y);
	}


	// scale data
	X_train.mul(1.0/255);
	X_train.plus(-0.5);
	X_valid.mul(1.0/255);
	X_valid.plus(-0.5);
	X_test.mul(1.0/255);
	X_test.plus(-0.5);



	DO_TEST(false)
	log << "=========== Training KNN ===========\n";
	KNN<dtype> knn(11, KNN<dtype>::Auto); // KNN<>::KDTree
	t1 = chrono::steady_clock::now();
	knn.train(X_train, y_train);
	t2 = chrono::steady_clock::now();
	log << "Trained in " << format_duration(t1, t2) << ".\n";
	t1 = chrono::steady_clock::now();
	y_pred = knn.classify(X_test);
	t2 = chrono::steady_clock::now();
	log << "Classified in " << format_duration(t1, t2) << ".\n";
	cm = confusion_matrix<>(y_test, y_pred, cm_mode);
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";
	END_TEST



	DO_TEST(true)
	log << "=========== k-Means clustering ===========\n";
	bool random_init = true;
	int n_clusters = 100;
	KMeans<dtype> km;
	{
	KMeans<dtype>::params opt;
	opt.info_iters = 10;
	km.set_params(opt);
	}
	bool recluster = true;
	if (km.load("mnist_kmeans_clusters.dat")) {
		cout << km.n_clusters() << " clusters found in the disk file. Do you want to load and skip clustering (y/n)? ";
		string answer;
		cin >> answer;
		recluster = (answer[0]=='n' || answer[0]=='N');
	}
	if (recluster) {
		if (!random_init) {
			n_clusters = 10;
			umat<dtype> X_seed(n_clusters, X_train.xdim());
			for (int i=0; i<n_clusters; ++i) {
				int pos = y_train.find_random(i);
				X_seed.set_row(i, X_train.row_offset(pos), X_seed.xdim());
			}
			km.seed(n_clusters, X_seed);
		}
		t1 = chrono::steady_clock::now();
		km.fit(n_clusters, X_train);
		t2 = chrono::steady_clock::now();
		log << n_clusters << " clusters created in " << format_duration(t1, t2) << ".\n";
		km.save("mnist_kmeans_clusters.dat");
	}
	umat<dtype> centroids;
	uvec<int> spc;
	km.clustering_info(centroids, spc);
	//log << "Samples per cluster: " << spc.format() << "\n";

	ClusterMap<dtype> cmap;
	cmap.set_openmp_threads(12);
	t1 = chrono::steady_clock::now();
	cmap.map(centroids, X_train, y_train);
	t2 = chrono::steady_clock::now();
	log << "Mapped in " << format_duration(t1, t2) << ".\n";

	log << "Training set classification.. ";
	t1 = chrono::steady_clock::now();
	y_pred = cmap.classify(X_train);
	t2 = chrono::steady_clock::now();
	log << "done in " << format_duration(t1, t2) << ".\n";
	cm = confusion_matrix<>(y_train, y_pred, cm_mode);
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";

	log << "Test set classification.. ";
	t1 = chrono::steady_clock::now();
	y_pred = cmap.classify(X_test);
	t2 = chrono::steady_clock::now();
	log << "done in " << format_duration(t1, t2) << ".\n";
	cm = confusion_matrix<>(y_test, y_pred, cm_mode);
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";
	
	log << "Test set classification with KNN.. ";
	KNN<dtype> knn(5, KNN<dtype>::Auto);
	knn.train(centroids, cmap.get_mapping());
	t1 = chrono::steady_clock::now();
	y_pred = knn.classify(X_test);
	t2 = chrono::steady_clock::now();
	log << "done in " << format_duration(t1, t2) << ".\n";
	cm = confusion_matrix<>(y_test, y_pred, cm_mode);
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";
	END_TEST
}
