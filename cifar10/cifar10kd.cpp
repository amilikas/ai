// Knowledge distillation

#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "splitter.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "nn/backprop.hpp"
#include "datasets/cifarloader.hpp"
#include "datasets/binfile.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }

// Datatype for data and neural network
typedef float dtype;

#define SAVES_FOLDER         "../saves/"

// Activation functions
//int fconv = fLinear;
int fconv = fReLU;
int cnn_ffc = fReLU;
int ff_ffc = fLogistic;


template <typename Type=float>
void apply_softmaxT(umat<Type>& Y, Type T)
{
	assert(Y.dev()==device::CPU);
	for (int i=0; i<Y.ydim(); ++i) {
		Type max = Y(i,0);
		for (int j=1; j<Y.xdim(); ++j) if (Y(i,j) > max) max = Y(i,j);
		Type sum = 0;
		for (int j=0; j<Y.xdim(); ++j) {
			Y(i,j) = std::exp((Y(i,j) - max)/T);
			sum += Y(i,j);
		}
		for (int j=0; j<Y.xdim(); ++j) Y(i,j) /= sum;
	}
}


FFNN<dtype> create_teacher()
{
	FFNN<dtype> net("cifar10-kd-teacher");
	net.add(new InputLayer<dtype>(dims3{32,32,3}));
	net.add(new Conv2DLayer<dtype>(32, 5,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DropoutLayer<dtype>(0.25));
	net.add(new Conv2DLayer<dtype>(64, 3,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DropoutLayer<dtype>(0.25));
	net.add(new Conv2DLayer<dtype>(128, 3,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DenseLayer<dtype>(128, cnn_ffc));
	//net.add(new SoftmaxLayer<dtype>(10));
	net.add(new DenseLayer<dtype>(10));
	return net;	
}

FFNN<dtype> create_student()
{
	FFNN<dtype> net("cifar10-kd-student");
	net.add(new InputLayer<dtype>(dims3{32,32,3}));
	net.add(new Conv2DLayer<dtype>(8, 5,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	//net.add(new DropoutLayer<dtype>(0.25));
	net.add(new Conv2DLayer<dtype>(16, 3,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	//net.add(new DropoutLayer<dtype>(0.25));
	net.add(new Conv2DLayer<dtype>(32, 3,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DenseLayer<dtype>(64, cnn_ffc));
	//net.add(new SoftmaxLayer<dtype>(10));
	net.add(new DenseLayer<dtype>(10));
	return net;	
}


int main() 
{
	/*
	// check KLdivergence calculate (remove softmax code) - CORRECT
	// https://machinelearningmastery.com/divergence-between-probability-distributions/
	umat<dtype> P(1,6); P.set("0,0,1, 0.10, 0.40, 0.50");
	umat<dtype> Q(1,3); Q.set("0.80, 0.15, 0.05");
	umat<dtype> g;
	lossfunc::kdloss<dtype> loss(0.0, 1.0);
	std::cout << "KLloss=" << loss.calculate(g, P, Q) << " (should be 1.33568)\n";
	return 0;
	*/


	umat<dtype> X, X_train, X_valid, X_test;
	umat<dtype> Y_train, Y_logits, Y_train_1hot, Y_valid_1hot, Y_test_1hot;
	uvec<int> y, y_train, y_valid, y_test, y_pred, y_train_pred;
	steady_clock::time_point t1, t2;

	// set openmp threads to maximum
	umml_set_openmp_threads();

	// load cifar10
	string path = "../data/CIFAR10/";
	string train_file  = path + "data_batch_";
	string test_file   = path + "test_batch.bin";
	string logits_file = path + "cifar10-teacher-logits.bin";


	CIFARloader cifar10;
	dataframe df;
	for (int i=1; i<=5; ++i) {
		umat<dtype> X_temp;
		uvec<int> y_temp;
		if (cifar10.load_images(train_file+to_string(i)+".bin", X_temp, y_temp)) {
			X = df.vstack(X, X_temp);
			y = df.append(y, y_temp);
		}
	}
	cifar10.load_images(test_file, X_test, y_test);
	if (!cifar10) {
		cout << "Error loading CIFAR10: " << cifar10.error_description() << "\n";
		return -1;
	} else {
		cout << "Loaded " << X.ydim() << " training images, " << y.len() << " labels.\n";
		cout << "Loaded " << X_test.ydim() << " test images, " << y_test.len() << " labels.\n";
	}

	Binfile<double> bin;
	Y_logits.resize(40000,10);
	bin.load(logits_file, Y_logits, 40000, 10);
	if (!bin) {
		cout << "Error loading teacher logits: " << bin.error_description() << "\n";
		return -1;
	} else {
		cout << "Loaded " << Y_logits.shape() << " logits.\n";
	}


	bool load_validation = true;
	if (load_validation) {
		splitter<dtype,int> ds(X, y);
		ds.split_train_test_sets(40000,10000);
		ds.get_splits(X_train, X_valid, y_train, y_valid);
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

	// convert labels to 1hot encoding
	onehot_enc<> enc;
	enc.fit_encode(y_train, Y_train_1hot);
	enc.encode(y_valid, Y_valid_1hot);
	enc.encode(y_test, Y_test_1hot);

	// concatenate Y_train_1hot and Y_logits
	Y_train.resize(Y_logits.ydim(), 2*Y_logits.xdim());
	Y_train.copy_cols(Y_train_1hot, 0, 10, 0);
	Y_train.copy_cols(Y_logits, 0, 10, 10);
	cout << "Y_train: " << Y_train.shape() << "\n";



	DO_TEST(true)

	FFNN<dtype> net = create_student();
	if (!net) {
		cout << "Error creating neural network: " << net.error_description() << "\n";
		return -1;
	}

	Logger log(net.get_name()+".log");

	// upload to GPU
	bool use_gpu = false;
	if (use_gpu) { 
		X_train.to_gpu(); X_valid.to_gpu(); X_test.to_gpu(); 
		Y_valid_1hot.to_gpu(); Y_train_1hot.to_gpu(); 
	}

	log << umml_compute_info(use_gpu) << "\n";
	log << "Training data: " << X_train.shape() << " " << X_train.bytes() << "\n";
	log << "Validation data: " << X_valid.shape() << " " << X_valid.bytes() << "\n";
	log << "Testing data: " << X_test.shape() << " " << X_test.bytes() << "\n\n";
	log << net.info() << "\n";

	string answer="n";
	if (0) if (check_file_exists(string(SAVES_FOLDER)+net.get_name()+".sav")) {
		cout << "Trainined network found in " << net.get_name() << ".sav. Do you want to load it? (y/n) ";
		cin >> answer;
		if (answer[0]=='y' || answer[0]=='Y') {
			answer = "y";
			if (net.load(net.get_name()+".sav")) {
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
		opt.batch_size = 16;
		opt.max_iters = 200;  // 5
		opt.info_iters = 1;
		opt.verbose = true;
		opt.autosave = 10;
		opt.multisave = true;
		opt.filename = string(SAVES_FOLDER)+net.get_name()+".sav";
		bp.set_params(opt);
		}
		bp.set_logging_stream(&log);
		//gdstep::learnrate<dtype> st(0.01);
		//gdstep::momentum<dtype> st(0.001, 0.9);
		gdstep::adam<dtype> st(0.0005, 0.9);
		//lossfunc::softmaxce<dtype> loss;
		lossfunc::kdloss<dtype> loss(1.0, 2.0);
		shuffle::deterministic sh;


		log << "Backprop: " << bp.info() << ", " << st.info() << ", " << loss.info() << "\n\n";
		log << "Training on " << device_name(X_train.dev()) << "...\n";

		t1 = chrono::steady_clock::now();
		if (load_validation) bp.train(net, loss, st, sh, X_train, Y_train, X_valid, Y_valid_1hot);
		else bp.train(net, loss, st, sh, X_train, Y_train);
		t2 = chrono::steady_clock::now();
		log << "Trained in " << format_duration(t1, t2) << ".\n";
		//net.save(net.get_name()+".sav");
	}


	// test
	double acc;
	umat<dtype> Y_pred(X_train.dev());
	umat<dtype> Ypred1hot;
	
	t1 = chrono::steady_clock::now();
	net.predict(X_train, Y_pred);
	t2 = chrono::steady_clock::now();
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);
	acc = accuracy(Y_train_1hot, Ypred1hot);
	log << "Training set predicted in " << format_duration(t1, t2) << " with " << acc << " accuracy.\n";

	Ypred1hot.to_gpu();
	Y_train_1hot.to_cpu();
	Ypred1hot.resize_like(Y_logits);
	Ypred1hot.argmaxto1hot(Y_logits);
	log << "Accuracy from logits: " << accuracy(Ypred1hot, Y_train_1hot) << "\n";
	
	Y_pred.to_device(X_train.dev());
	t1 = chrono::steady_clock::now();
	net.predict(X_test, Y_pred);
	t2 = chrono::steady_clock::now();
	Y_pred.to_cpu();
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);

	confmat<> cm;
	const int cm_mode = CM_Macro;
	if (!enc.decode(Ypred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
	cm = confusion_matrix<>(y_test, y_pred, cm_mode);
	log << "Test set predicted in " << format_duration(t1, t2) << " with " << accuracy<>(y_test,y_pred) << " accuracy.\n";
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n\n";

	log << "Examples:\n";
	Y_pred.to_device(X_train.dev());
	net.predict(X_train, Y_pred);
	Y_train_1hot.to_cpu();
	Y_logits.to_cpu();
	Y_pred.to_cpu();

	umat<dtype> T10, L10, P10;
	T10.resize(10,10); T10.copy_rows(Y_train_1hot, 0, 10);
	L10.resize(10,10); L10.copy_rows(Y_logits, 0, 10);
	P10.resize(10,10); P10.copy_rows(Y_pred, 0, 10);
	log << "== LOGITS ==\n";
	for (int i=0; i<10; ++i) {
		log << T10.row(i).format(0,6) << "\n" << L10.row(i).format(2,6) << "\n" << P10.row(i).format(2,6) << "\n\n";
	}


	END_TEST


	return 0;
}
