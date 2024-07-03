#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "stats.hpp"
#include "kmeans.hpp"
#include "nn/backprop.hpp"
#include "bio/ga.hpp"
#include "datasets/mnistloader.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }



#define DESCR_STR "average"


#define SAVES_FOLDER         "../../saves/mnist/"
#define STUDENT_NAME         "mnistkdmerge-student"
#define LOG_NAME             "mnistkdmerge(" DESCR_STR ").log"
#define GASAVE_NAME          "mnistkdmerge(" DESCR_STR ").ga"
#define TRAINED_TEACHER_NAME "mnistkdmerge-trained-teacher"


// Datatype for data and neural network
typedef float dtype;

// Seed for local RNG
constexpr int SEED = 48;

// Batch size
constexpr int BATCH_SIZE = 10;

// GA parameters
constexpr int  GA_POP_SIZE = 1000;
constexpr int  GA_ITERS = 150;


// GA chromosome encoding
// only layers 1 and 3 needs to be encoded
// Nt and Ns need the input size to be known
const std::vector<int> layers = { 1, 3 };
const std::vector<int> Nt = { 18, 84 };
const std::vector<int> Ns = {  6, 14 }; 
typedef std::pair<uvec<dtype>, uvec<dtype>> weights_and_biases;

// Activation functions
constexpr int fconv = fReLU;
constexpr int cnn_ffc = fReLU;
constexpr int ff_ffc = fLogistic;

FFNN<dtype> create_teacher(const std::string& name, bool softmax)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(Nt[0],5,1,Valid,fconv)); 
	net.add(new MaxPool2DLayer<dtype>(4,4));
	net.add(new DenseLayer<dtype>(Nt[1], cnn_ffc));
	if (softmax) net.add(new SoftmaxLayer<dtype>(10));
	else net.add(new DenseLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_student(const std::string& name, bool softmax)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(Ns[0],5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(4,4));
	net.add(new DenseLayer<dtype>(Ns[1], cnn_ffc));
	if (softmax) net.add(new SoftmaxLayer<dtype>(10));
	else net.add(new DenseLayer<dtype>(10));
	return net;
}



// -------------- GA -------------------
// input is teacher's weights and biases
// output is which teacher's weights will be used in student network

string tostring(const vector<int>& chain)
{
	stringstream ss;
	for (int a : chain) {
		ss << a << " ";
	}
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
// 
void solution(const std::vector<int>& chain, const FFNN<dtype>& teacher, FFNN<dtype>& student)
{
	// create student's wbs using member's chain (combine with teacher weights)
	std::vector<weights_and_biases> swbs;
	int c=0;
	std::vector<int> all = layers;
	all.push_back(layers.back()+1);

	for (size_t i=0; i<all.size(); ++i) {
		int l = all[i];
		dims4 sd = student.get_layer(l)->_w.dims();
		dims4 td = teacher.get_layer(l)->_w.dims();
		uvec<dtype> sw, sb, tw, tb;
		student.get_layer(l)->get_trainable_parameters(sw, sb);
		teacher.get_layer(l)->get_trainable_parameters(tw, tb);
		int p = 0;
		for (int j=0; j<sd.y; ++j) {
			int v[2];
            if (i >= layers.size()) {
                v[0] = v[1] = j;
            } else {
            	v[0] = j;
				v[1] = chain[c++];
            }

			int twidx = v[1]*td.x;
			for (int k=0; k<sd.x; ++k) {
				dtype avg = (sw(p+k) + tw(twidx+k)) / 2.0;
				sw(p+k) = avg;
			}
			p += sd.x;
		}

		swbs.push_back(std::make_pair(sw, sb));
	}

	// set student's network weights from wbs
	assert(swbs.size()==all.size());
	for (size_t i=0; i<swbs.size(); ++i) {
		int l = all[i];
		student.get_layer(l)->set_trainable_parameters(swbs[i].first, swbs[i].second);
	}
}


// fitness function
// determines how good a solution is
struct Fitness {
	umat<dtype> X, Y, Xv, Yv;
	FFNN<dtype> teacher;
	FFNN<dtype> student;

	double operator ()(const std::vector<int>& chain) {
		FFNN<dtype> net;
		student.clone_to(net);

		solution(chain, teacher, net);

		umat<dtype> Y_pred;
		umat<dtype> Ypred1hot;
		net.predict(Xv, Y_pred);
		Ypred1hot.resize_like(Y_pred);
		Ypred1hot.argmaxto1hot(Y_pred);
		double acc = accuracy(Yv, Ypred1hot);
		double f = acc;
		//double f = std::exp(6*acc-6.0);
		//double f = std::pow(1+acc, 5) / std::pow(2,5);
		//double f = (acc < 0.95 ? std::tanh(2.5*acc) : acc);

		return f;
	}
};


// -------------- KD -------------------

template <typename Type, class Loss, class Stepper, class Shuffle>
void train_network(FFNN<Type>& net, Logger& log, Loss& loss, Stepper& st, Shuffle& sh, int epochs,
				   const umat<Type>& X_train, const umat<Type>& Y_train, 
				   const umat<Type>& X_valid, const umat<Type>& Y_valid)
{
	steady_clock::time_point t1, t2;
	Backprop<dtype> bp;
	{
	Backprop<dtype>::params opt;
	opt.batch_size = 20;
	opt.max_iters = epochs;
	opt.info_iters = 1;
	opt.verbose = true;
	opt.autosave = 5;
	opt.multisave = false;
	opt.filename = string(SAVES_FOLDER)+net.get_name()+".sav";
	bp.set_params(opt);
	}
	bp.set_logging_stream(&log);
	log << "Backprop: " << bp.info() << ", " << st.info() << ", " << loss.info() << "\n";
	log << "Training on " << device_name(X_train.dev()) << "...\n";

	t1 = chrono::steady_clock::now();
	bp.train(net, loss, st, sh, X_train, Y_train, X_valid, Y_valid);
	t2 = chrono::steady_clock::now();
	log << "Trained in " << format_duration(t1, t2) << ".\n";
}

template <typename Type>
double evaluate_network(FFNN<Type>& net, Logger& log, const umat<Type>& X, const umat<Type>& Y, const string& setname)
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
	return acc;
}

template <typename Type>
std::string statistics(FFNN<Type>& net)
{
	std::stringstream ss;
	histogram<Type> h;
	ss << "Stats for " << net.get_name() << "\n";
	for (int l=1; l<net.nlayers(); ++l) {
		uvec<Type> w, b;
		net.get_layer(l)->get_trainable_parameters(w, b);
		if (!w.empty()) {
			ss << "Layer " << l << " " << net.get_layer(l)->get_name() << "\n";
			ss << "variance: " << variance(w) << "\n";
			h.fit(w, 10);
			ss << h.format(80,3,6);
		}
	}
	return ss.str();
}


// -------------- Main -------------------

int main() 
{
	umat<dtype> X, X_teacher, X_student, X_valid, X_test;
	umat<dtype> Y, Y_teacher, Y_student, Y_valid, Y_test;
	uvec<int> y, y_train, y_valid, y_test;
	steady_clock::time_point t1, t2;
	double teacher_acc, student_acc, best_acc, worst_acc, random_acc;

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

	// encode digits with one-hot encoding
	onehot_enc<> enc;
	enc.fit_encode(y, Y);
	enc.encode(y_valid, Y_valid);
	enc.encode(y_test, Y_test);

	// teacher's and student's training set
	X_teacher = X; 
	Y_teacher = Y;
	X_student = X; 
	Y_student = Y;

	bool use_gpu = false;
	if (use_gpu) { 
		X_teacher.to_gpu(); X_student.to_gpu(); X_valid.to_gpu(); X_test.to_gpu();
		Y_teacher.to_gpu(); Y_student.to_gpu(); Y_valid.to_gpu(); Y_test.to_gpu();
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
	log << "\n" << teacher.info() << "\n";
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
			train_network(trained_teacher, log, loss, st, sh, 15, X_teacher, Y_teacher, X_valid, Y_valid);
		} else {
			cout << "Nothing more to do.\n";
			return 0;
		}
	}
	if (teacher.load(string(SAVES_FOLDER)+teacher.get_name()+".sav")) {
		cout << "Trained teacher parameters loaded ok.\n";
		log << statistics(teacher);
		// Calculate logits for X_train
		cout << "Calculating training set logits..\n";
		umat<dtype> Y_logits(X_student.dev());
		teacher.predict(X_student, Y_logits);
		// concatenate Y_student and Y_logits for KDLoss
		umat<dtype> Y_prev;
		Y_student.to_cpu();
		Y_logits.to_cpu();
		Y_prev = Y_student;
		Y_student.resize(Y_logits.ydim(), 2*Y_logits.xdim());
		Y_student.copy_cols(Y_prev, 0, 10, 0);
		Y_student.copy_cols(Y_logits, 0, 10, 10);
		Y_student.to_device(X_student.dev());
		cout << "Y_student: " << Y_student.shape() << "\n\n";
	} else {
		cout << "Error loading trained teacher parameters: " << teacher.error_description() << "\n";
		return -1;
	}
	teacher_acc = evaluate_network(teacher, log, X_test, Y_test, "Trained teacher test");
	log << "\n";
	// save trained teacher's weights
	std::vector<weights_and_biases> twbs;
	for (int i=1; i<teacher.nlayers(); ++i) {
		uvec<dtype> ws, bs;
		teacher.get_layer(i)->get_trainable_parameters(ws, bs);
		if (ws.len()) twbs.push_back(std::make_pair(ws,bs));
	}

	FFNN<dtype> student = create_student(STUDENT_NAME, false);
	if (!student) {
		cout << "Error creating student neural network: " << student.error_description() << "\n";
		return -1;
	}
	log << "\n" << student.info() << "\n";
	if (!check_file_exists(string(SAVES_FOLDER)+student.get_name()+".sav")) {
		log << "\nTrain student with normal KD\n";
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(student, log, loss, st, sh, 25, X_student, Y_student, X_valid, Y_valid);
	} else if (student.load(string(SAVES_FOLDER)+student.get_name()+".sav")) {
		cout << "Trained student parameters loaded ok.\n";
		log << statistics(student);
	} else {
		cout << "Error loading trained teacher parameters: " << student.error_description() << "\n";
		return -1;
	}
	student_acc = evaluate_network(student, log, X_test, Y_test, "Student with KD: Test");
	log << "\n";




	DO_TEST(true)

	// ========================
	// GA parameters
	// ========================
	int popsize = GA_POP_SIZE;
	int chainsize = 0;
	for (int n: Ns) chainsize += n;
	log << "Number of values in each chain: " << chainsize << "\n";
	
	GA<int> ga;
	// no openmp parallelization, results are inconsistent due to SGD
	GA<int>::params opt;
	opt.info_iters = 1;
	opt.max_iters = GA_ITERS;
	opt.elitism = 0.02;
	opt.filename = string(SAVES_FOLDER)+GASAVE_NAME;
	opt.autosave = 1;
	opt.parallel = GA<int>::SingleThread;
	ga.set_params(opt);
	ga.set_logging_stream(&log);

	initializer::null<int> noinit;
	initializer::multiset::random<int,0> rndinit(Ns, Nt, true);
	crossover::multiset::onepoint<int> xo(Ns, true);
	mutation::multiset::replace<int,0> mut(Ns, Nt, true, 0.1, 0.01);
	//mutation::null<int> mut;
	//crossover::null<int> xo;

	Fitness ff;
	constexpr int Nsamples = 10000;
	ff.X.resize(Nsamples, X_student.xdim());
	ff.X.copy_rows(X_student, 0, Nsamples);
	ff.Y.resize(Nsamples, Y_student.xdim()); 
	ff.Y.copy_rows(Y_student, 0, Nsamples);
	ff.Xv.resize_like(X_valid);
	ff.Xv.set(X_valid);
	ff.Yv.resize_like(Y_valid);
	ff.Yv.set(Y_valid);
	teacher.clone_to(ff.teacher);
	student.clone_to(ff.student);
	
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

	// student with KD but filters averaged by the GA (best member)
	if (1) {
		log << "\nstudent with KD, filters averaged by the GA (best member)\n";
		GA<int>::member m = ga.get_member(0);
		log << "Best member: " << tostring(m.chain) << "\n";
		FFNN<dtype> net;
		student.clone_to(net);
		solution(m.chain, teacher, net);
		/*
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, 25, X_student, Y_student, X_valid, Y_valid);
		*/
		best_acc = evaluate_network(net, log, X_test, Y_test, "Student with KD (GA optimized): Test");
		log << "\n";
	}

	// student with KD but filters averaged by the GA (worst member)
	if (1) {
		log << "\nstudent with KD, filters averaged by the GA (worst member)\n";
		GA<int>::member m = ga.get_member(popsize-1);
		log << "Worst member: " << tostring(m.chain) << "\n";
		FFNN<dtype> net;
		student.clone_to(net);
		solution(m.chain, teacher, net);
		/*
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, 25, X_student, Y_student, X_valid, Y_valid);
		*/
		worst_acc = evaluate_network(net, log, X_test, Y_test, "Student with KD (GA optimized): Test");
		log << "\n";
	}


	if (1) {
		double totacc = 0.0;
		int RUNS = 5;
		log << "\nstudent with KD, filters choosen randomly (full training)\n";
		for (int t=0; t<RUNS; ++t) {
			vector<int> r(chainsize);
			rndinit.apply(r);
			log << "Random" << t+1 << ": " << tostring(r) << "\n";
			FFNN<dtype> net;
			student.clone_to(net);
			solution(r, teacher, net);
			/*
			lossfunc::kdloss<dtype> loss(0.2, 5.0);
			gdstep::adam<dtype> st(0.0001);
			shuffle::deterministic sh;
			train_network(net, log, loss, st, sh, 25, X_student, Y_student, X_valid, Y_valid);
			*/
			totacc += evaluate_network(net, log, X_test, Y_test, "Student with KD (random filters): Test");
			log << "\n";
		}
		random_acc = totacc/RUNS;
		log << "Average (" << RUNS << " random runs): " << random_acc << "\n";
	}

	log << 	"\nResults:\n" << 
			"Teacher:      " << teacher_acc << "\n" <<
			"Student:      " << student_acc << "\n" <<
			"Best merged:  " << best_acc << "\n" <<
			"Worst merged: " << worst_acc << "\n" <<
			"Random (avg): " << random_acc << "\n";

	END_TEST




	return 0;
}
