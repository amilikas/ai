#include "dataset.hpp"
#include "scaler.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "timer.hpp"
#include "multiclass.hpp"
#include "lsvm.hpp"
#include "knn.hpp"
#include "nn/backprop.hpp"
#include "nn/art1.hpp"
#include "kmeans.hpp"
#include "clustermap.hpp"
#include "debug.hpp"
#include "datasets/mnistloader.hpp"


using namespace std;
using namespace mml;

#define DO_TEST(x) if(x){
#define END_TEST   }

constexpr int OMP_THREADS = 10;

typedef float dtype;

int main() 
{
	matr<dtype> X, X_mod, X_unmod, X_valid;
	ivec y, y_mod, y_unmod, y_valid;
	unsigned long t1, t2;
	string fname = "modmnist";
	Logger log(fname+".log");

	mml_set_openmp_threads(OMP_THREADS);
	log << mml_cpu_info() << "\n";

	// modify method
	// non-gaussian: all modified digits are set to a specific, random digit.
	// gaussian: each modified digit is set to a random digit.
	enum { GaussianNoise, NonGaussianNoise };
	int method = NonGaussianNoise;
	int mod_size = 2000;
	int valid_size =  5000;
	string modified = "modified/";
	string validation = "modified/v";
	if (method==GaussianNoise) modified += "g";
	else modified += "ng";
	modified += std::to_string(mod_size/1000);
	validation += std::to_string(valid_size/1000);

	// load mnist
	string path = "../data/MNIST/";
	string train_images = path + "train-images-idx3-ubyte";
	string train_labels = path + "train-labels-idx1-ubyte";
	MNISTloader mnist;
	mnist.load_images(train_images, X);
	mnist.load_labels(train_labels, y);
	if (!mnist) {
		log << "Error loading MNIST: " << mnist.error_description() << "\n";
		return -1;
	} else {
		log << "Loaded " << X.nrows() << " training images, " << y.len() << " labels.\n";
	}

	// keep the first 'train_size' samples as training set and
	// the last 'valid_size' as validation set (leave those unmodified)
	X_valid = X.copy_rows(-valid_size, valid_size);
	y_valid = y.patch_copy(-valid_size, valid_size);
	X = X.copy_rows(0, mod_size);
	y = y.patch_copy(0, mod_size);
	log << "Validation set: " << X_valid.nrows() << " images, " << y_valid.len() << " labels.\n";
	log << "Modified training set: " << X.nrows() << " images, " << y.len() << " labels.\n";
	
	// prepare tempered and validation file names
	string modified_images = path + modified + "k-images-idx3-ubyte";
	string modified_labels = path + modified + "k-labels-idx1-ubyte";
	string valid_images = path + validation + "k-images-idx3-ubyte";
	string valid_labels = path + validation + "k-labels-idx1-ubyte";
	string unmod_images = path + modified + "k-unmod-images-idx3-ubyte";
	string unmod_labels = path + modified + "k-unmod-labels-idx1-ubyte";

	

	DO_TEST(true)
	float ratio = 0.5;
	log << "=========== Modifying MNIST " << ratio*100 << "% for each digit ===========\n";
	t1 = time_ms();
	for (int digit=0; digit<=9; ++digit) {
		ivec idcs = y.select(digit);
		matr<dtype> X_digit = X.copy_rows(idcs);
		ivec y_digit = y.copy_selected(idcs);
		X_unmod.vstack(X_digit);
		y_unmod.append(y_digit);
		int wrong_digit = digit;
		while (wrong_digit==digit) random_ints<int>(&wrong_digit, 1, 0, 9);
		int n = idcs.len()*ratio;
		for (int i=0; i<n; ++i) {
			if (method==GaussianNoise) {
				wrong_digit = digit;
				while (wrong_digit==digit) random_ints<int>(&wrong_digit, 1, 0, 9);
			}
			y_digit(i) = wrong_digit;
		}
		X_mod.vstack(X_digit);
		y_mod.append(y_digit);
		//X_part.vstack(X_digit.rows_copy(n, X_digit.nrows()-n));
		//y_part.append(y_digit.patch_copy(n, X_digit.nrows()-n));
	}
	// only labels are actually modified
	mnist.save_images(modified_images, X_mod);
	mnist.save_labels(modified_labels, y_mod);
	mnist.save_images(unmod_images, X_unmod);
	mnist.save_labels(unmod_labels, y_unmod);
	if (valid_size > 0) {
		mnist.save_images(valid_images, X_valid);
		mnist.save_labels(valid_labels, y_valid);
	}
	t2 = time_ms();
	log << "Finished in " << format_timing(t2, t1) << ".\n";

	log << "\nUnmodified dataset contains:\n";
	for (int c=0; c<=9; ++c) {
		log << "digit " << c << ": " << y_unmod.count(c) << " samples.\n";
	}
	log << "\nModified dataset contains:\n";
	for (int c=0; c<=9; ++c) {
		log << "digit " << c << ": " << y_mod.count(c) << " samples.\n";
	}

	END_TEST

	return 0;
}
