// selects samples with maximum information
// STEP 1: create teacher and student networks, save initial student weights
// STEP 2: train teacher and save logits
// STEP 3: use a GA to find the optimal training samples of a subset of the training set
// STEP 4: train a neural network to learn the selection of training samples that the GA produced
// STEP 5: use the neural network of step 4 to the whole dataset
// STEP 6: train the student and evaluate

#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "stats.hpp"
#include "nn/backprop.hpp"
#include "bio/ga.hpp"
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
#define TRAINED_TEACHER_NAME "mnist-kd-trained-teacher" TEACHER_DESCR
#define STUDENT_NAME         "mnist-kd-student"
#define SELECTION_NAME       "mnist-kd-selection"
#define GASAVE_NAME          "mnist-kd-ga"
#define LOG_NAME             "mnist-kd-ga.log"

// GA parameters
constexpr int GA_TOT_SAMPLES  = 2000;
constexpr int GA_KEEP_SAMPLES = 1000;
constexpr int GA_POP_SIZE     = 500;
constexpr int GA_ITERS        = 50;
constexpr int GA_TRAIN_EPOCHS = 50;
constexpr int GA_BATCH_SIZE   = 10;


// Datatype for data and neural network
typedef float dtype;

// Saved trainable parameters
typedef std::pair<uvec<dtype>, uvec<dtype>> weights_and_biases;


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


// teacher neural network: MNIST {28,28} -> {10}
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

// student neural network: MNIST {28,28} -> {10}
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

// selection neural network logits {10} -> 0/1 (selected or not)
FFNN<dtype> create_selection(const std::string& name)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(10));
	net.add(new DenseLayer<dtype>(16, fReLU));
	net.add(new SoftmaxLayer<dtype>(2));
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


string tostring(const vector<int>& chain)
{
	stringstream ss;
	for (int a : chain) ss << a << " ";
	return ss.str();
}

string output_member(const GA<int>::member& m, int i)	{
	stringstream ss;
	ss << std::fixed;
   	ss << std::setprecision(6);
	ss << "member " << std::setw(3) << i+1 << ", fitness=" << m.fitness << ", " << tostring(m.chain) << "\n";
	return ss.str();
}

// converts a chain to a solution
// selects the training samples
template <typename Type>
void solution(const std::vector<int>& chain, 
			  const umat<Type>& X, const umat<Type>& Y, 
			  umat<Type>& X_train, umat<Type>& Y_train)
{
	int n = (int)chain.size();
	X_train.resize(n, X.xdim());
	Y_train.resize(n, Y.xdim());
	for (int i=0; i<n; ++i) {
		X_train.set_row(i, X.row_offset(chain[i]), X.xdim());
		Y_train.set_row(i, Y.row_offset(chain[i]), Y.xdim());
	}
}

// fitness function: trains a neural network using backpropagation
void train_member(const std::vector<int>& chain, FFNN<dtype>& net, 
		   const umat<dtype>& X, const umat<dtype>& Y, int iters)
{
	umat<dtype> X_train, Y_train;
	solution(chain, X, Y, X_train, Y_train);

	Backprop<dtype> bp;
	{
	// stochastic must be set to false in order for the GA to work
	typename Backprop<dtype>::params opt;
	opt.batch_size = GA_BATCH_SIZE;
	opt.max_iters = iters;
	opt.info_iters = 1;
	opt.verbose = false;
	bp.set_params(opt);
	}
	lossfunc::kdloss<dtype> loss(0.01, 5.0);
	gdstep::adam<dtype> st(0.0001);
	shuffle::deterministic sh;
	bp.train(net, loss, st, sh, X_train, Y_train);
}

// fitness function
// determines how good a solution is
struct Fitness {
	umat<dtype> X, Y, Xv, Yv;
	std::vector<weights_and_biases> initial;
	double operator ()(const std::vector<int>& chain) {
		FFNN<dtype> net = create_student("", false);
		for (int i=1; i<net.nlayers(); ++i)
			net.get_layer(i)->set_trainable_parameters(initial[i-1].first, initial[i-1].second);
		train_member(chain, net, X, Y, GA_TRAIN_EPOCHS);
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

		return f;
	}
};



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


	//
	// Teacher
	//

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


	//
	// Student
	//

	FFNN<dtype> student = create_student(STUDENT_NAME, false);
	if (!student) {
		cout << "Error creating student neural network: " << student.error_description() << "\n";
		return -1;
	}
	log << student.info() << "\n";

	// save initial student's weights
	std::vector<weights_and_biases> initial;
	for (int i=1; i<student.nlayers(); ++i) {
		uvec<dtype> ws, bs;
		student.get_layer(i)->get_trainable_parameters(ws, bs);
		initial.push_back(std::make_pair(ws,bs));
	}

	// student's initial training set
	X_student = X_teacher;
	Y_student = Y_teacher;

	// concatenate Y_student and Y_logits for KDLoss
	umat<dtype> Y_prev;
	Y_prev = Y_student;
	Y_student.resize(Y_logits.ydim(), 2*Y_logits.xdim());
	Y_student.copy_cols(Y_prev, 0, 10, 0);
	Y_student.copy_cols(Y_logits, 0, 10, 10);
	log << "X_student: " << X_student.shape() << "\n";
	log << "Y_student: " << Y_student.shape() << "\n";
	if (use_gpu) { 
		X_student.to_gpu(); Y_student.to_gpu(); 
	}



	//
	// GA
	//

	int popsize = GA_POP_SIZE;
	int chainsize = GA_KEEP_SAMPLES;
	log << "\nNumber of samples in each chain: " << chainsize << "\n";
	
	GA<int> ga;
	// no openmp parallelization, results are inconsistent due to SGD
	GA<int>::params opt;
	opt.info_iters = 1;
	opt.max_iters = GA_ITERS;
	opt.elitism = 0.02;
	opt.filename = string(SAVES_FOLDER)+GASAVE_NAME+".ga";
	opt.autosave = 1;
	opt.parallel = GA<int>::SingleThread;
	ga.set_params(opt);
	ga.set_logging_stream(&log);

	initializer::null<int> noinit;
	initializer::multiset::random<int,0> rndinit({GA_KEEP_SAMPLES},{GA_TOT_SAMPLES}, true);
	crossover::multiset::onepoint<int> xo({GA_KEEP_SAMPLES}, true);
	mutation::multiset::replace<int> mut({GA_KEEP_SAMPLES},{GA_TOT_SAMPLES}, true, 0.1, 0.01);

	Fitness ff;
	ff.X.resize(GA_TOT_SAMPLES, X_student.xdim());
	ff.X.copy_rows(X_student, 0, GA_TOT_SAMPLES);
	ff.Y.resize(GA_TOT_SAMPLES, Y_student.xdim()); 
	ff.Y.copy_rows(Y_student, 0, GA_TOT_SAMPLES);
	ff.Xv.resize_like(X_valid);
	ff.Xv.set(X_valid);
	ff.Yv.resize_like(Y_valid);
	ff.Yv.set(Y_valid);
	ff.initial = initial;
	
	string answer = "n";
	if (1) if (check_file_exists(opt.filename)) {
		cout << "A population is found in " << opt.filename << ". Please choose:\n";
		cout << "[n]. Do NOT load it and procced with evolution.\n";
		cout << "[t]. Load it and continue evolution.\n";
		cout << "[y]. Load it and skip evolution.\n";
		for (;;) {
			cin >> answer;
			if (tolower(answer[0])=='n' || tolower(answer[0])=='t' || tolower(answer[0])=='y') break;
		}
		if (tolower(answer[0])=='y' || tolower(answer[0])=='t') {
			if (ga.load(opt.filename)) {
				cout << "Population loaded ok, " << ga.generations() << " generations.\n";
			} else {
				cout << "Error loading population: " << ga.error_description() << "\n";
				answer = "n";
			}
		}
	}
	if (tolower(answer[0]) != 'y') {
		log << "Evolving " << popsize << " members for " << opt.max_iters << " generations...\n";
		if (tolower(answer[0])=='n') {
			ga.init(chainsize, popsize, rndinit);
		} else {
			ga.init(chainsize, popsize, noinit);
			ga.evolve(xo, mut);
		}
		for (;;) {
			ga.evaluate(ff);
			ga.sort();
			if (ga.done()) break;
			ga.evolve(xo, mut);
		}
		ga.finish();
	}

	log << "Evolution finished, top-10 members:\n";
	for (int i=0; i<std::min(10,popsize); ++i) log << output_member(ga.get_member(i), i);
	if (popsize > 10) log << output_member(ga.get_member(popsize-1), popsize-1);


	//
	// Student with KD
	//
	if (0) {
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
	// Selection network
	//

	// create dataset from a member's chain
	umat<dtype> X_sel, X_null;
	umat<dtype> Y_sel, Y_null;
	std::vector<int> chain = ga.get_member(0).chain;
	X_sel.resize(GA_TOT_SAMPLES, Y_logits.xdim());
	X_sel.copy_rows(Y_logits, 0, GA_TOT_SAMPLES);
	Y_sel.resize(GA_TOT_SAMPLES, 2);
	for (int i=0; i<GA_TOT_SAMPLES; ++i) {
		Y_sel(i,0) = 0;
		Y_sel(i,1) = 1;
	}
	for (int i : chain) {
		Y_sel(i,0) = 1;
		Y_sel(i,1) = 0;
	}

	// train the network
	FFNN<dtype> selection = create_selection(SELECTION_NAME);
	lossfunc::softmaxce<dtype> loss;
	gdstep::adam<dtype> st(0.0001);
	shuffle::deterministic sh;
	train_network(selection, log, loss, st, sh, 100, X_sel, Y_sel, X_null, Y_null);
	evaluate_network(selection, log, X_sel, Y_sel, "Training");


	return 0;
}
