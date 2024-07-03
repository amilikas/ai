#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "splitter.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "nn/backprop.hpp"
#include "datasets/cifarloader.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }

// folder to save weights
#define SAVES_FOLDER         "../../saves/cifar10/"

// Datatype for data and neural network
typedef float dtype;

// Activation functions
//int fconv = fLinear;
int fconv = fReLU;
int cnn_ffc = fReLU;
int ff_ffc = fLogistic;


FFNN<dtype> create_teacher()
{
	FFNN<dtype> net("cifar10cnn-teacher");
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
	net.add(new SoftmaxLayer<dtype>(10));
	return net;	
}

FFNN<dtype> create_student()
{
	FFNN<dtype> net("cifar10cnn-student");
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
	net.add(new SoftmaxLayer<dtype>(10));
	return net;	
}

FFNN<dtype> create_cnn()
{
	return create_teacher();
}



int main() 
{
	umat<dtype> X, X_train, X_valid, X_test;
	umat<dtype> Y_train, Y_valid, Y_test;
	uvec<int> y, y_train, y_valid, y_test, y_pred, y_train_pred;
	steady_clock::time_point t1, t2;

	// set openmp threads to maximum
	umml_set_openmp_threads();

	// load cifar10
	string path = "../../../auth/data/CIFAR10/";
	string train_file = path + "data_batch_";
	string test_file  = path + "test_batch.bin";
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
		std::cout << cifar10.error_description() << "\n";
		return -1;
	} else {
		cout << "Loaded " << X.ydim() << " training images, " << y.len() << " labels.\n";
		cout << "Loaded " << X_test.ydim() << " test images, " << y_test.len() << " labels.\n";
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
	enc.fit_encode(y_train, Y_train);
	enc.encode(y_valid, Y_valid);
	enc.encode(y_test, Y_test);



	DO_TEST(true)

	FFNN<dtype> net = create_cnn();
	if (!net) {
		cout << "Error creating neural network: " << net.error_description() << "\n";
		return -1;
	}

	Logger log(net.get_name()+".log");

	// upload to GPU
	bool use_gpu = true;
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
	if (0) if (check_file_exists(SAVES+net.get_name()+".sav")) {
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
		log << "Training...\n";
		Backprop<dtype> bp;
		{
		Backprop<dtype>::params opt;
		opt.batch_size = 16;
		opt.max_iters = 5;  // 5
		opt.info_iters = 1;
		opt.verbose = true;
		opt.autosave = 5;
		opt.multisave = false;
		opt.filename = string(SAVES_FOLDER)+net.get_name()+".sav";
		bp.set_params(opt);
		}
		bp.set_logging_stream(&log);
		lossfunc::softmaxce<dtype> loss;
		//gdstep::learnrate<dtype> st(0.01);
		//gdstep::momentum<dtype> st(0.001, 0.9);
		gdstep::adam<dtype> st(0.0005, 0.9);
		shuffle::deterministic sh;

		log << "Backprop: " << bp.info() << ", " << st.info() << ", " << loss.info() << "\n\n";
		log << "Training on " << device_name(X_train.dev()) << "...\n";

		t1 = chrono::steady_clock::now();
		if (load_validation) bp.train(net, loss, st, sh, X_train, Y_train, X_valid, Y_valid);
		else bp.train(net, loss, st, sh, X_train, Y_train);
		t2 = chrono::steady_clock::now();
		log << "Trained in " << format_duration(t1, t2) << ".\n";
		//net.save(net.get_name()+".sav");
	}
	Y_train_1hot.to_cpu();

	// test
	double acc;
	umat<dtype> Y_pred;
	umat<dtype> Ypred1hot;
	Y_pred.to_device(X_train.dev());
	
	t1 = chrono::steady_clock::now();
	net.predict(X_train, Y_pred);
	t2 = chrono::steady_clock::now();
	Y_pred.to_cpu();
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);
	acc = accuracy(Y_train, Ypred1hot);
	log << "Training set predicted in " << format_duration(t1, t2) << " with " << acc << " accuracy.\n";
	
	Y_pred.to_device(X_train.dev());
	t1 = chrono::steady_clock::now();
	net.predict(X_test, Y_pred);
	t2 = chrono::steady_clock::now();
	Y_pred.to_cpu();
	Ypred1hot.resize_like(Y_pred);
	Ypred1hot.argmaxto1hot(Y_pred);
	acc = accuracy(Y_test, Ypred1hot);
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
