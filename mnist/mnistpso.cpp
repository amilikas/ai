// Outliers detection
//

#include "cpuinfo.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "dataframe.hpp"
#include "datasets/mnistloader.hpp"
#include "bio/pso.hpp"
#include "nn/ffnn.hpp"
#include "nn/backprop.hpp"
#include "glplot.hpp"
#include <iostream>

using namespace std;
using namespace std::chrono;
using namespace umml;


#define DO_TEST(x) if(x){
#define END_TEST   }


typedef float dtype;

#define   LOG_FILENAME "mnistpso-1to10-1k-explore.log"
#define   NET_NAME     "mnistpso"
#define   SAVES_FOLDER "../../saves/mnist/"


constexpr int     OMP_THREADS  = 12;

constexpr int     BATCH_SIZE   = 30;
constexpr double  LRN_RATE     = 0.03;

constexpr int     GROUP_SIZE   = 10;
constexpr int     SWARM_SIZE   = 1000;
constexpr int     PSO_MAXITERS = 50;


string tostring(const vector<bool>& chain)
{
	stringstream ss;
	for (auto a : chain) ss << (a ? "1" : "0");
	return ss.str();
}

template <typename Type>
vector<bool> tobitstring(const vector<Type>& x, Type threshold=0.5)
{
	vector<bool> bstr(x.size());
	for (size_t i=0; i<x.size(); ++i) bstr[i] = (x[i] > threshold ? 1 : 0);
		return bstr;
}


int fconv = fReLU;
int cnn_ffc = fReLU;
int ff_ffc = fLogistic;

FFNN<dtype> create_cnn()
{
	FFNN<dtype> net(string(NET_NAME)+"-cnn");
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(6,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(16,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(120,4,1,Valid,fconv));
	net.add(new DenseLayer<dtype>(84, cnn_ffc));
	net.add(new SoftmaxLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_ffnn()
{
	FFNN<dtype> net(string(NET_NAME)+"-ff");
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	//net.add(new DenseLayer<dtype>(32, ff_ffc));
	//net.add(new DenseLayer<dtype>(32, ff_ffc));
	net.add(new SoftmaxLayer<dtype>(10));
	//net.add(new DenseLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_network()
{
	//return create_cnn();
	return create_ffnn();
}

// converts a chain to a solution
// selects the training samples
template <typename dtype=float>
void solution(const std::vector<bool>& chain, 
			  const umat<dtype>& X, const umat<dtype>& Y, 
			  umat<dtype>& X_train, umat<dtype>& Y_train)
{
	assert(Y.ydim() % GROUP_SIZE == 0);
	int len = Y.ydim() / GROUP_SIZE;
	assert(chain.size()==(size_t)len);
	int n = std::count(chain.begin(), chain.end(), true);
	X_train.resize(n*GROUP_SIZE, X.xdim());
	Y_train.resize(n*GROUP_SIZE, Y.xdim());
	int k=0;
	for (int i=0; i<len; ++i) {
		if (chain[i]) {
			for (int j=0; j<GROUP_SIZE; ++j) {
				X_train.set_row(k+j, X.row_offset(i*GROUP_SIZE+j), X.xdim());
				Y_train.set_row(k+j, Y.row_offset(i*GROUP_SIZE+j), Y.xdim());
			}
			k += GROUP_SIZE;
		}
	}
}


// trains a neural network using backpropagation
template <typename dtype=float>
void train(const std::vector<bool>& chain, FFNN<dtype>& net, 
		   const umat<dtype>& X, const umat<dtype>& Y, int iters)
{
	umat<dtype> X_train, Y_train;

	solution(chain, X, Y, X_train, Y_train);

	Backprop<dtype> bp;
	{
	typename Backprop<dtype>::params opt;
	opt.batch_size = BATCH_SIZE;
	opt.max_iters = iters;
	opt.info_iters = 1;
	opt.verbose = false;
	bp.set_params(opt);
	}
	lossfunc::softmaxce<dtype> loss;
	gdstep::learnrate<dtype> st(LRN_RATE);
	shuffle::deterministic sh;
	
	bp.train(net, loss, st, sh, X_train, Y_train);
}


// fitness function
// determines how good a solution is
struct Fitness {
	umat<dtype> X, Y, Xv, Yv;
	std::vector<uvec<dtype>> weights;
	std::vector<uvec<dtype>> biases;
	double operator ()(const std::vector<dtype>& chain) {
		FFNN<dtype> net = create_network();
		for (int i=1; i<net.nlayers(); ++i)
			net.get_layer(i)->set_trainable_parameters(weights[i-1], biases[i-1]);
		train(tobitstring(chain), net, X, Y, 10);
		umat<dtype> Y_pred;
		umat<dtype> Ypred1hot;
		net.predict(Xv, Y_pred);
		Ypred1hot.resize_like(Y_pred);
		Ypred1hot.argmaxto1hot(Y_pred);
		double acc = accuracy(Yv, Ypred1hot);
		double f = acc;
		//double f = std::pow(1+acc, 5) / std::pow(2,5);
		//double f = std::exp(6*acc-6.0);
		//double f = (acc < 0.95 ? std::tanh(2.5*acc) : acc);

		return 1.0 - f;
	}
};


// create a mask from a vector of indeces
std::vector<bool> create_mask(const uvec<int>& idcs, int n)
{
	std::vector<bool> mask(n, false);
	for (int i=0; i<idcs.len(); ++i) {
		int j=idcs(i);
		assert(j < n);
		mask[j] = true;
	}
	return mask;
}



int main()
{
	umat<dtype> X, X_train, X_valid, X_test;
	uvec<int> y, y_train, y_valid, y_test, y_unmod, y_pred, y_train_pred;
	umat<dtype> Y_pred, Ypred1hot;
	steady_clock::time_point t1, t2;

	umml_seed_rng(48);
	umml_set_openmp_threads(OMP_THREADS);

	Logger log(LOG_FILENAME);

	// load mnist
	bool load_validation = true;
	string path = "../../../auth/data/MNIST/";
	string original = "train";
	string modified = "modified/ng2k";
	string unmodified = "modified/ng2k-unmod";
	string train_images = path + modified + "-images-idx3-ubyte";
	string train_labels = path + modified + "-labels-idx1-ubyte";
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


	dataframe df;
	uvec<int> idcs = df.select_equal(y, y_unmod);
	std::vector<bool> mask = create_mask(idcs, y.len());
	std::vector<bool> clean = mask;
	std::vector<bool> dirty(y.len(), true);


	// show bit string
	if (false) {
	glplot plt;
	plt.set_window_geometry(1100,256);
	plt.set_window_title("Image view");
	uvec<float> img(2000);
	int i=0;
	for (int x=0; x<200; ++x) 
	for (int y=0; y<10; ++y) {
		img(y*200+x) = mask[i] ? 1.0f : 0.3f;
		++i;
	}
	plt.add_grayscale_image(img.mem(), 200, 10);
	plt.show();
	std::exit(0);
	}

	
	bool shuffle=false;
	if (shuffle) {
		/*
		dataset<dtype> ds(X,y);
		ds.shuffle();
		ds.split_train_test_sets(1.0);
		matr<dtype> X_tmp;
		vect<int> y_tmp;
		ds.get_splits(X_train, X_tmp, y_train, y_tmp);
		*/ 
	} else {
		// WARNING: comment these to see how training with modified set performs
		X_train.resize_like(X); X_train.set(X);
		y_train.resize_like(y); y_train.set(y);
	}
	
	// scale data
	X_train.mul(1.0/255);
	X_train.plus(-0.5);
	X_valid.mul(1.0/255);
	X_valid.plus(-0.5);
	X_test.mul(1.0/255);
	X_test.plus(-0.5);

	// convert labels to 1hot encoding
	// encode digits with one-hot encoding
	onehot_enc<> enc;
	umat<dtype> Y_train_1hot, Y_valid_1hot, Y_test_1hot;
	enc.fit_encode(y_train, Y_train_1hot);
	enc.encode(y_valid, Y_valid_1hot);
	enc.encode(y_test, Y_test_1hot);

	// create the FFNN
	FFNN<dtype> net = create_network();
	if (!net) {
		cout << "Error creating neural network: " << net.error_description() << "\n";
		return -1;
	}

	log << umml_compute_info() << "\n";
	log << "Training data: " << X_train.shape() << " " << X_train.bytes() << "\n";
	log << "Validation data: " << X_valid.shape() << " " << X_valid.bytes() << "\n";
	log << "Testing data: " << X_test.shape() << " " << X_test.bytes() << "\n";
	log << net.info() << "\n";

	// save initial weights, so all neural networks start with the same weights
	std::vector<uvec<dtype>> weights;
	std::vector<uvec<dtype>> biases;
	for (int i=1; i<net.nlayers(); ++i) {
		uvec<dtype> ws, bs;
		net.get_layer(i)->get_trainable_parameters(ws, bs);
		weights.push_back(ws);
		biases.push_back(bs);
	}


	DO_TEST(false)
	// ========================
	// Normal training
	// ========================
	vector<vector<bool>> chains = { dirty, dirty, clean, clean };
	FFNN<dtype> cnn = create_cnn();
	for (vector<bool> chain : chains) {
	//{ vector<bool> chain = clean;
		umml_seed_rng(48);
		FFNN<dtype> net;
		cnn.clone_to(net);
		t1 = chrono::steady_clock::now();
		train(chain, net, X_train, Y_train_1hot, 25);
		t2 = chrono::steady_clock::now();
		log << "CNN trained in " << format_duration(t1, t2) << ".\n";

		net.predict(X_valid, Y_pred);
		Ypred1hot.resize_like(Y_pred);
		Ypred1hot.argmaxto1hot(Y_pred);
		log << "Validation set predicted with " << accuracy(Y_valid_1hot, Ypred1hot) << " accuracy.\n";

		net.predict(X_test, Y_pred);
		Ypred1hot.resize_like(Y_pred);
		Ypred1hot.argmaxto1hot(Y_pred);
		log << "Test set predicted with " << accuracy(Y_test_1hot, Ypred1hot) << " accuracy.\n";
	}
	END_TEST


	DO_TEST(true)

	// ========================
	// PSO parameters
	// ========================

	int swarmsize = SWARM_SIZE;
	int tot_bits = Y_train_1hot.ydim() / GROUP_SIZE;
	log << "Number of values in each chain: " << tot_bits << "\n";
	
	PSO<dtype> pso;
	// no openmp parallelization, results are inconsistent due to SGD
	PSO<dtype>::params opt;
	opt.info_iters = 1;
	opt.c1 = 1.5;
	opt.c2 = 2.0;
	opt.max_iters = PSO_MAXITERS;
	opt.filename = string(SAVES_FOLDER)+net.get_name()+".pso";
	opt.autosave = 1;
	opt.parallel = PSO<dtype>::SingleThread;
	pso.set_params(opt);
	pso.set_logging_stream(&log);

	Fitness ff;
	ff.X.resize_like(X_train); 
	ff.X.set(X_train);
	ff.Y.resize_like(Y_train_1hot); 
	ff.Y.set(Y_train_1hot);
	ff.Xv.resize_like(X_valid);
	ff.Xv.set(X_valid);
	ff.Yv.resize_like(Y_valid_1hot);
	ff.Yv.set(Y_valid_1hot);
	ff.weights = weights;
	ff.biases = biases;

	initializer::values::random<dtype> rndinit(0.0, 1.0);
	pso.init(swarmsize, tot_bits, 0.0, 1.0, rndinit);
	
	string answer = "n";
	if (1) if (check_file_exists(opt.filename)) {
		cout << "A swarm is found in " << opt.filename << ". Please choose:\n";
		cout << "[n]. Do NOT load it and procced with evolution.\n";
		cout << "[t]. Load it and continue evolution.\n";
		cout << "[y]. Load it and skip evolution.\n";
		for (;;) {
			cin >> answer;
			if (tolower(answer[0])=='n' || tolower(answer[0])=='t' || tolower(answer[0])=='y') break;
		}
		if (tolower(answer[0])=='y' || tolower(answer[0])=='t') {
			if (pso.load(opt.filename)) {
				cout << "Swarm loaded ok, " << pso.iterations() << " iterations.\n";
			} else {
				cout << "Error loading swarm: " << pso.error_description() << "\n";
				answer = "n";
			}
		}
	}
	if (tolower(answer[0]) != 'y') {
		log << "Searching with " << swarmsize << " particles for " << opt.max_iters << " iterations...\n";
		if (tolower(answer[0]) != 'n') {
			pso.step();
		}
		for (;;) {
			pso.evaluate(ff);
			pso.update();
			if (pso.done()) break;
			pso.step();
		}
		pso.finish();
	}
	
	
	log << "PSO finished, top-10 particles:\n";
	for (int i=0; i<std::min(10,swarmsize); ++i) {
		PSO<dtype>::particle m = pso.get_particle(i);
		log << "particle " << std::setw(3) << i+1 << ", minima=" << m.best_minimum << "\n";
	}

	// training and test accuracy with the best member
	t1 = chrono::steady_clock::now();
	train(tobitstring(pso.get_particle(0).x), net, X_train, Y_train_1hot, 25);
	t2 = chrono::steady_clock::now();
	log << "Trained in " << format_duration(t1, t2) << ".\n";
	
	net.predict(X_valid, Y_pred);
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);
	if (!enc.decode(Ypred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
	log << "Validation set predicted with " << accuracy<>(y_valid, y_pred) << " accuracy.\n";
	
	confmat<> cm;
	const int cm_mode = CM_Macro;
	net.predict(X_test, Y_pred);
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);
	if (!enc.decode(Ypred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
	cm = confusion_matrix<>(y_test, y_pred, cm_mode);
	log << "Test set\n";
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";


	umml_seed_rng(48);
	FFNN<dtype> cnn = create_cnn();
	t1 = chrono::steady_clock::now();
	train(tobitstring(pso.get_particle(0).x), cnn, X_train, Y_train_1hot, 25);
	t2 = chrono::steady_clock::now();
	log << "CNN trained in " << format_duration(t1, t2) << ".\n";
	cnn.predict(X_test, Y_pred);
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);
	if (!enc.decode(Ypred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
	cm = confusion_matrix<>(y_test, y_pred, cm_mode);
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";


	// show bitstring
	if (true) {
	glplot plt;
	plt.set_window_geometry(1100,256);
	plt.set_window_title("Best members");
	plt.set_bkgnd_color(0.16f, 0.48f, 0.58f);
	plt.add_image_grid(6,1,15);
	{
		uvec<float> img(2000);
		for (int x=0; x<200; ++x) 
		for (int y=0; y<10; ++y) {
			int g = (x*10+y);
			img(y*200+x) = mask[g] ? 1.0f : 0.3f;
		}
		plt.add_grayscale_image(img.mem(), 200, 10);
	}
	for (int b=0; b<5; ++b) {
		uvec<float> img(2000);
		std::vector<bool> m = tobitstring(pso.get_particle(b).x);
		//cout << tostring(m) << "\n";
		for (int x=0; x<200; ++x) 
		for (int y=0; y<10; ++y) {
			int g = (x*10+y) / GROUP_SIZE;
			img(y*200+x) = m[g] ? 1.0f : 0.3f;
		}
		plt.add_grayscale_image(img.mem(), 200, 10);
	}
	plt.show();
	}

	END_TEST


	return 0;
}
