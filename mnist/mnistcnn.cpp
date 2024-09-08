#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "nn/backprop.hpp"
#include "datasets/mnistloader.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }

#define SAVES_FOLDER         "../saves/"

// Datatype for data and neural network
typedef float dtype;

// Activation functions
//int fconv = fLinear;
int fconv = fReLU;
int cnn_ffc = fReLU;
int ff_ffc = fLogistic;


FFNN<dtype> create_lenet()
{
	FFNN<dtype> net("lenet");
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(6,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DropoutLayer<dtype>(0.25));
	net.add(new Conv2DLayer<dtype>(16,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DropoutLayer<dtype>(0.25));
	net.add(new Conv2DLayer<dtype>(120,4,1,Valid,fconv));
	net.add(new DenseLayer<dtype>(84, cnn_ffc));
	net.add(new SoftmaxLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_cnn()
{
	return create_lenet();
}

FFNN<dtype> create_ffnn()
{
	FFNN<dtype> net("mnistffnn");
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new DenseLayer<dtype>(32, ff_ffc));
	net.add(new DenseLayer<dtype>(32, ff_ffc));
	net.add(new SoftmaxLayer<dtype>(10));
	return net;
}


int main() 
{
	umat<dtype> X, X_train, X_valid, X_test;
	umat<dtype> Y_train_1hot, Y_valid_1hot, Y_test_1hot;
	uvec<int> y, y_train, y_valid, y_test, y_unmod, y_pred, y_train_pred;
	steady_clock::time_point t1, t2;

	// set openmp threads to maximum
	umml_set_openmp_threads();

	// load mnist
	bool load_validation = true;
	string path = "../data/MNIST/";
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

	// encode digits with one-hot encoding
	onehot_enc<> enc;
	enc.fit_encode(y_train, Y_train_1hot);
	enc.encode(y_valid, Y_valid_1hot);
	enc.encode(y_test, Y_test_1hot);



	DO_TEST(true)

	FFNN<dtype> net = create_cnn();
	if (!net) {
		cout << "Error creating neural network: " << net.error_description() << "\n";
		return -1;
	}

	Logger log(net.get_name()+".log");

	bool use_gpu = false;
	if (use_gpu) { 
		X_train.to_gpu(); Y_train_1hot.to_gpu();
		X_valid.to_gpu(); Y_valid_1hot.to_gpu();
		X_test.to_gpu(); Y_test_1hot.to_gpu(); 
	}

	log << umml_compute_info(use_gpu) << "\n";
	log << "Training data: " << X_train.shape() << " " << X_train.bytes() << "\n";
	log << "Validation data: " << X_valid.shape() << " " << X_valid.bytes() << "\n";
	log << "Testing data: " << X_test.shape() << " " << X_test.bytes() << "\n";
	log << net.info() << "\n";

	string answer="n";
	if (0) if (check_file_exists(string(SAVES_FOLDER)+net.get_name()+".sav")) {
		cout << "Trainined network found in " << net.get_name() << ".sav. Do you want to load it? (y/n) ";
		cin >> answer;
		if (answer[0]=='y' || answer[0]=='Y') {
			answer = "y";
			if (net.load(string(SAVES_FOLDER)+net.get_name()+".sav")) {
				cout << "Trainined parameters loaded ok.\n";
			} else {
				cout << "Error loading parameters: " << net.error_description() << "\n";
				answer = "n";
			}
		}
	}

	// training
	if (answer[0] != 'y') {
		Backprop<dtype> bp;
		{
		Backprop<dtype>::params opt;
		opt.batch_size = 30;
		opt.max_iters = 5;  // 5
		opt.info_iters = 1;
		opt.verbose = true;
		opt.autosave = 5;
		opt.multisave = true;
		opt.filename = string(SAVES_FOLDER)+net.get_name()+".sav";
		bp.set_params(opt);
		}
		bp.set_logging_stream(&log);
		lossfunc::softmaxce<dtype> loss;
		gdstep::learnrate<dtype> st(0.01);
		//gdstep::momentum<dtype> st(0.001, 0.9);
		//gdstep::adam<dtype> st(0.0005, 0.9);
		//shuffle::deterministic sh;
		shuffle::stochastic sh;

		log << "Backprop: " << bp.info() << ", " << st.info() << ", " << loss.info() << "\n\n";
		log << "Training on " << device_name(X_train.dev()) << "...\n";

		t1 = chrono::steady_clock::now();
		if (load_validation) bp.train(net, loss, st, sh, X_train, Y_train_1hot, X_valid, Y_valid_1hot);
		else bp.train(net, loss, st, sh, X_train, Y_train_1hot);
		t2 = chrono::steady_clock::now();
		log << "Trained in " << format_duration(t1, t2) << ".\n";
	}

	// evaluate
	double acc;
	umat<dtype> Y_pred(X_train.dev());
	umat<dtype> Ypred1hot(X_train.dev());
	
	t1 = chrono::steady_clock::now();
	net.predict(X_train, Y_pred);
	t2 = chrono::steady_clock::now();
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);
	acc = accuracy(Y_train_1hot, Ypred1hot);
	log << "Training set predicted in " << format_duration(t1, t2) << " with " << acc << " accuracy.\n";

	t1 = chrono::steady_clock::now();
	net.predict(X_test, Y_pred);
	t2 = chrono::steady_clock::now();
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);
	acc = accuracy(Y_test_1hot, Ypred1hot);
	log << "Test set predicted in " << format_duration(t1, t2) << " with " << acc << " accuracy.\n";

	confmat<> cm;
	const int cm_mode = CM_Macro;
	Ypred1hot.to_cpu();
	if (!enc.decode(Ypred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
	cm = confusion_matrix<>(y_test, y_pred, cm_mode);
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";

	END_TEST


	return 0;
}
