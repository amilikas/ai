#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "stats.hpp"
#include "nn/backprop.hpp"
#include "datasets/mnistloader.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }

#define DROPOUT true

#if DROPOUT==true
#define TEACHER_DESCR "(dropout)"
#else
#define TEACHER_DESCR "(nodropout)"
#endif

#define SAVES_FOLDER         "../../saves/mnist/"
#define TRAINED_TEACHER_NAME "mnist-trained-teacher" TEACHER_DESCR
#define STUDENT_NAME         "mnist-student"
#define LOG_NAME             "mnist-kd.log"

// Datatype for data and neural network
typedef float dtype;

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


FFNN<dtype> create_teacher(const std::string& name, bool softmax)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(6,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(16,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(120,4,1,Valid,fconv));
	#if DROPOUT==true
	net.add(new DropoutLayer<dtype>(0.25));
	#endif
	net.add(new DenseLayer<dtype>(84, cnn_ffc));
	if (softmax) net.add(new SoftmaxLayer<dtype>(10));
	else net.add(new DenseLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_student(const std::string& name, bool softmax)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	/*
	net.add(new Conv2DLayer<dtype>(4,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(10,5,1,Same,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(40,4,1,Same,fconv));
	*/
	net.add(new DenseLayer<dtype>(40, cnn_ffc));
	if (softmax) net.add(new SoftmaxLayer<dtype>(10));
	else net.add(new DenseLayer<dtype>(10));
	return net;
}


template <typename Type, class Loss, class Stepper, class Shuffle>
void train_network(FFNN<Type>& net, Logger& log, Loss& loss, Stepper& st, Shuffle& sh, int epochs,
				   const umat<Type>& X_train, const umat<Type>& Y_train, 
				   const umat<Type>& X_valid, const umat<Type>& Y_valid)
{
	steady_clock::time_point t1, t2;
	Backprop<dtype> bp;
	{
	Backprop<dtype>::params opt;
	opt.batch_size = 30;
	opt.max_iters = epochs;
	opt.info_iters = 10;
	opt.verbose = true;
	opt.autosave = 5;
	opt.multisave = false;
	opt.filename = string(SAVES_FOLDER)+net.get_name()+".sav";
	bp.set_params(opt);
	}
	bp.set_logging_stream(&log);
	log << "Backprop: " << bp.info() << ", " << st.info() << ", " << loss.info() << "\n\n";
	log << "Training on " << device_name(X_train.dev()) << "...\n";

	t1 = chrono::steady_clock::now();
	bp.train(net, loss, st, sh, X_train, Y_train, X_valid, Y_valid);
	t2 = chrono::steady_clock::now();
	log << "Trained in " << format_duration(t1, t2) << ".\n";
}

template <typename Type>
umat<Type> evaluate_network(FFNN<Type>& net, Logger& log, const umat<Type>& X, const umat<Type>& Y, const string& setname)
{
	steady_clock::time_point t1, t2;
	umat<Type> Y_pred(X.dev()), Y_pred1hot(X.dev());
	t1 = chrono::steady_clock::now();
	net.predict(X, Y_pred);
	t2 = chrono::steady_clock::now();
	Y_pred1hot.resize_like(Y_pred);
	Y_pred1hot.argmaxto1hot(Y_pred);
	double acc = accuracy(Y, Y_pred1hot);
	log << setname << " set predicted in " << format_duration(t1, t2) << " with " << acc << " accuracy.\n";
	return Y_pred1hot;
}


int main() 
{
	umat<dtype> X, X_teacher, X_student, X_valid, X_test;
	umat<dtype> Y, Y_teacher, Y_student, Y_valid, Y_test, Y_logits;
	uvec<int> y, y_train, y_valid, y_test, y_pred, y_train_pred;
	steady_clock::time_point t1, t2;

	// set openmp threads to maximum
	umml_set_openmp_threads();

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

	// encode digits with one-hot encoding
	onehot_enc<> enc;
	enc.fit_encode(y, Y);
	enc.encode(y_valid, Y_valid);
	enc.encode(y_test, Y_test);

	// teacher's training set
	X_teacher = X; 
	Y_teacher = Y;




	DO_TEST(true)

	Logger log(LOG_NAME);

	bool use_gpu = false;
	if (use_gpu) { 
		X_teacher.to_gpu(); X_valid.to_gpu(); X_test.to_gpu();
		Y_teacher.to_gpu(); Y_valid.to_gpu(); Y_test.to_gpu();
		Y_logits.to_gpu();
	}

	log << umml_compute_info(use_gpu) << "\n";
	log << "Teacher data: X " << X_teacher.shape() << " " << X_teacher.bytes() << ", Y " << Y_teacher.shape() << "\n";
	log << "Student data: X " << X_student.shape() << " " << X_student.bytes() << ", Y " << Y_student.shape() << "\n";
	log << "Validation data: X " << X_valid.shape() << " " << X_valid.bytes() << ", Y " << Y_valid.shape() << "\n";
	log << "Testing data: X " << X_test.shape() << " " << X_test.bytes() << ", Y " << Y_test.shape() << "\n";

	FFNN<dtype> teacher = create_teacher(TRAINED_TEACHER_NAME, false);
	if (!teacher) {
		cout << "Error creating teacher neural network: " << teacher.error_description() << "\n";
		return -1;
	}
	log << teacher.info() << "\n";

	if (!check_file_exists(string(SAVES_FOLDER)+teacher.get_name()+".sav")) {
		cout << "Trained teacher network not found. Do you want to start training it? (y/n) ";
		string answer="n";
		cin >> answer;
		if (answer[0]=='y' || answer[0]=='Y') {
			FFNN<dtype> trained_teacher = create_teacher(TRAINED_TEACHER_NAME, true);
			if (!trained_teacher) {
				cout << "Error creating trained teacher neural network: " << trained_teacher.error_description() << "\n";
				return -1;
			}
			log << trained_teacher.info() << "\n";
			lossfunc::softmaxce<dtype> loss;
			gdstep::adam<dtype> st(0.0001);
			shuffle::deterministic sh;
			train_network(trained_teacher, log, loss, st, sh, 25, X_teacher, Y_teacher, X_valid, Y_valid);
			evaluate_network(trained_teacher, log, X_teacher, Y_teacher, "Training");
		} else {
			cout << "Nothing to do.\n";
			return 0;
		}
	}
	if (teacher.load(string(SAVES_FOLDER)+teacher.get_name()+".sav")) {
		cout << "Trainined teacher parameters loaded ok.\n";
		// Calculate logits for X_train
		cout << "Calculating training set logits..\n";
		teacher.predict(X, Y_logits);
		evaluate_network(teacher, log, X_test, Y_test, "Test");
	} else {
		cout << "Error loading trained teacher parameters: " << teacher.error_description() << "\n";
		return -1;
	}

	// send logits to cpu memory
	Y_logits.to_cpu();


	// select student's training set
	enum {
		All,
		Only_7s_and_8s,
		LowVariance,
		Random,
	};

	int select = Random;
	if (select==All) {
		X_student = X; 
		Y_student = Y;
	} else if (select==Only_7s_and_8s) {
		// train only on 7s and 8s (α=0.01, T=5)
		umat<dtype> Y_log_tmp;
		uvec<int> idcs;
		idcs = df.select(y, 7);
		X_student = df.vstack(X_student, df.copy_rows(X, idcs));
		Y_student = df.vstack(Y_student, df.copy_rows(Y, idcs));
		Y_log_tmp = df.vstack(Y_log_tmp, df.copy_rows(Y_logits, idcs));
		idcs = df.select(y, 8);
		X_student = df.vstack(X_student, df.copy_rows(X, idcs));
		Y_student = df.vstack(Y_student, df.copy_rows(Y, idcs));	
		Y_log_tmp = df.vstack(Y_log_tmp, df.copy_rows(Y_logits, idcs));
		Y_logits  = Y_log_tmp;
	} else {
		umat<dtype> Y_softmax;
		vector<int> v;
		uvec<int> idcs;
		Y_softmax = Y_logits;
		apply_softmaxT(Y_softmax, dtype(5));
		dtype maxvar = 0.02;
		for (int i=0; i<Y_softmax.ydim(); ++i) {
			dtype var = variance<dtype>(Y_softmax.row(i), false);
			//log << i << " variance=" << var << ", " << Y_softmax.row(i).format(2) << "\n";
			if (var <= maxvar) v.push_back(i);
		}
		if (select==LowVariance) {
			// select samples with maximum information (low variance)
			idcs.resize(v.size());
			idcs.set(v);
			log << "indeces selected with variance <= " << maxvar << ": " << v.size() << "\n";
		} else {
			const int N = (int)v.size();
			build_shuffled_indeces(v, N);
			idcs.resize(v.size());
			idcs.set(v);
			log << "selected " << v.size() << " random samples.\n";
		}
		umat<dtype> Y_log_tmp;
		Y_log_tmp = df.copy_rows(Y_logits, idcs);
		X_student = df.copy_rows(X, idcs);
		Y_student = df.copy_rows(Y, idcs);
		Y_logits  = Y_log_tmp;
	}

	// concatenate Y_student and Y_logits for KDLoss
	umat<dtype> Y_prev;
	Y_prev = Y_student;
	Y_student.resize(Y_logits.ydim(), 2*Y_logits.xdim());
	Y_student.copy_cols(Y_prev, 0, 10, 0);
	Y_student.copy_cols(Y_logits, 0, 10, 10);
	log << "Y_student: " << Y_student.shape() << "\n";
	if (use_gpu) { 
		X_student.to_gpu(); Y_student.to_gpu(); 
	}


	//
	// Student with KD
	//
	if (1) {
		FFNN<dtype> student = create_student(STUDENT_NAME, false);
		//FFNN<dtype> student = create_teacher(STUDENT_NAME, false);
		if (!student) {
			cout << "Error creating student neural network: " << student.error_description() << "\n";
			return -1;
		}
		log << student.info() << "\n";

		lossfunc::kdloss<dtype> loss(0.01, 5.0); // best for training only in 7s & 8s
		//lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(student, log, loss, st, sh, 110, X_student, Y_student, X_valid, Y_valid);

		// evaluate
		umat<dtype> Y_pred1hot = evaluate_network(student, log, X_test, Y_test, "Test");
		confmat<> cm;
		const int cm_mode = CM_Macro;
		if (!enc.decode(Y_pred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
		cm = confusion_matrix<>(y_test, y_pred, cm_mode);
		log << "Accuracy  = " << accuracy<>(cm) << "\n";
		log << "Precision = " << precision<>(cm) << "\n";
		log << "Recall    = " << recall<>(cm) << "\n";
		log << "F1        = " << F1<>(cm) << "\n";
		log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";
	}

	//
	// Student without KD
	//
	if (1) {
		FFNN<dtype> student = create_student(STUDENT_NAME, true);
		//FFNN<dtype> student = create_teacher(STUDENT_NAME, false);
		if (!student) {
			cout << "Error creating student neural network: " << student.error_description() << "\n";
			return -1;
		}
		log << student.info() << "\n";

		lossfunc::softmaxce<dtype> loss;
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		Y_student.reshape(Y_student.ydim(), Y_student.xdim()/2);
		train_network(student, log, loss, st, sh, 58, X_student, Y_student, X_valid, Y_valid);

		// evaluate
		umat<dtype> Y_pred1hot = evaluate_network(student, log, X_test, Y_test, "Test");
		confmat<> cm;
		const int cm_mode = CM_Macro;
		if (!enc.decode(Y_pred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
		cm = confusion_matrix<>(y_test, y_pred, cm_mode);
		log << "Accuracy  = " << accuracy<>(cm) << "\n";
		log << "Precision = " << precision<>(cm) << "\n";
		log << "Recall    = " << recall<>(cm) << "\n";
		log << "F1        = " << F1<>(cm) << "\n";
		log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";
	}
	

	END_TEST


	return 0;
}
