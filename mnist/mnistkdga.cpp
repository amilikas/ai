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

#define FREEZE  false
#define DROPOUT true

#if DROPOUT==true
#define TEACHER_DESCR "(dropout)"
#else
#define TEACHER_DESCR "(nodropout)"
#endif

#if FREEZE==true
 #if DROPOUT==true
  #define DESCR_STR "(frozen-dropout)"
 #else
  #define DESCR_STR "(frozen-nodropout)"
 #endif
#else
 #if DROPOUT==true
  #define DESCR_STR "(all-dropout)"
 #else
  #define DESCR_STR "(all-nodropout)"
 #endif
#endif

#define SAVES_FOLDER         "../saves/"
#define STUDENT_NAME         "mnistkdga-student"
#define LOG_NAME             "mnistkdga" DESCR_STR ".log"
#define GASAVE_NAME          "mnistkdga" DESCR_STR 
#define TRAINED_TEACHER_NAME "mnistkdga-trained-teacher" TEACHER_DESCR


// Datatype for data and neural network
typedef float dtype;

// Seed for local RNG
constexpr int  SEED = 48;

// Batch size
constexpr int  BATCH_SIZE = 10;
constexpr int  TRAIN_EPOCHS = 25;

// GA parameters
constexpr int  GA_POP_SIZE = 100;
constexpr int  GA_ITERS = 50;
constexpr int  GA_TRAIN_EPOCHS = 20;
constexpr bool GA_FULL_TRAINING = false;

// Collecting results
typedef std::pair<double, std::string> result;


// GA encoding
#if DROPOUT==true
const std::vector<int> teacher_layers = { 1, 4, 7 };
#else
const std::vector<int> teacher_layers = { 1, 3, 5 };
#endif
const std::vector<int> student_layers = { 1, 3, 5 };
const std::vector<int> Nt = { 16, 32, 128 };
const std::vector<int> Ns = {  4,  8,  16 };
typedef std::pair<uvec<dtype>, uvec<dtype>> weights_and_biases;

// Activation functions
constexpr int fconv = fReLU;
constexpr int cnn_ffc = fReLU;
constexpr int ff_ffc = fLogistic;

FFNN<dtype> create_teacher(const std::string& name, bool softmax)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(Nt[0],3,1,Valid,fconv)); 
	net.add(new MaxPool2DLayer<dtype>(2,2));
	#if DROPOUT==true
	net.add(new DropoutLayer<dtype>(0.25));
	#endif
	net.add(new Conv2DLayer<dtype>(Nt[1],3,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	#if DROPOUT==true
	net.add(new DropoutLayer<dtype>(0.25));
	#endif
	net.add(new Conv2DLayer<dtype>(Nt[2],3,1,Valid,fconv));
	net.add(new DenseLayer<dtype>(84, cnn_ffc));
	if (softmax) net.add(new SoftmaxLayer<dtype>(10));
	else net.add(new DenseLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_student(const std::string& name, bool softmax)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(Ns[0],3,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(Ns[1],3,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(Ns[2],3,1,Valid,fconv));
	net.add(new DenseLayer<dtype>(42, cnn_ffc));
	if (softmax) net.add(new SoftmaxLayer<dtype>(10));
	else net.add(new DenseLayer<dtype>(10));
	return net;
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
			ss << " variance : " << variance(w) << "\n";
			ss << " magnitude: " << w.magnitude() << "\n";
			dims4 wd = net.get_layer(l)->weights_dims()[0];
			int depth = wd.x / 9;
			ss << " filters  : " << wd.y << "\n";
			ss << " depth    : " << depth << "\n";
			int p = 0;
			for (int f=0; f<wd.y; ++f) {
				ss << " f_" << std::setw(-2) << f << ": sum^2: ";
				for (int d=0; d<depth; ++d) {
					dtype sum = 0;
					for (int k=0; k<9; ++k) {
						sum += w(p)*w(p);
						p++;
					}
					ss << std::fixed;
					ss << setprecision(2);
					ss << sum << ",";
				}
				ss << "\n";
			}
			//h.fit(w, 10);
			//ss << h.graphic(80,3,6);
		}
	}
	return ss.str();
}



// -------------- GA -------------------
// input is teacher's weights and biases
// output is which teacher's weights will be used in student network

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

// maps student's channels to teacher's channels
int map_channel(int sl, int z, const std::vector<int>& chain)
{
	if (sl==0) return -1;
	else if (sl==1) return z;
	return Ns[sl-2]+z;
}

// converts a chain to a solution
void solution(const std::vector<int>& chain, const FFNN<dtype>& teacher, FFNN<dtype>& student, bool full_training)
{
	assert(std::accumulate(Ns.begin(), Ns.end(), 0) == (int)chain.size());

	std::vector<weights_and_biases> swbs;
	int nlayers = (int)student_layers.size();
	int c=0;

	for (int i=0; i<nlayers; ++i) {
		int sl = student_layers[i];
		int tl = teacher_layers[i];
		dims4 sd = student.get_layer(sl)->_w.dims();
		dims4 td = teacher.get_layer(tl)->_w.dims();
		uvec<dtype> sw, sb, tw, tb;
		student.get_layer(sl)->get_trainable_parameters(sw, sb);
		teacher.get_layer(tl)->get_trainable_parameters(tw, tb);
		int p = 0;
		for (int j=0; j<sd.y; ++j) {
			int twidx = chain[c]*td.x;

			constexpr int ksize = 3*3;
			// works only for 3x3 kernels!!!
			assert(sd.x % ksize == 0);
			for (int z=0; z<sd.x/ksize; ++z) {
				for (int x=0; x<ksize; ++x) {
					int cm = map_channel(i,z,chain);
					if (cm==-1) {
						sw(p+z*ksize+x) = tw(twidx+z*ksize+x);
					} else {
						sw(p+z*ksize+x) = tw(twidx+cm*ksize+x);
					}
				}
			}

/*
			for (int k=0; k<sd.x; ++k) {
				dtype value = tw(twidx+k);
				sw(p+k) = value;
			}
*/

			c++;
			p += sd.x;
		}

		student.get_layer(sl)->set_trainable_parameters(sw, sb);
		if (!full_training) student.get_layer(sl)->set_trainable(false);
	}
}

// trains a neural network using backpropagation
void train_member(const std::vector<int>& chain, FFNN<dtype>& net, 
		   		  const umat<dtype>& X, const umat<dtype>& Y, int iters)
{
	Backprop<dtype> bp;
	{
	// stochastic must be set to false in order for the GA to work
	typename Backprop<dtype>::params opt;
	opt.batch_size = BATCH_SIZE;
	opt.max_iters = iters;
	opt.info_iters = 1;
	opt.verbose = false;
	bp.set_params(opt);
	}
	lossfunc::kdloss<dtype> loss(0.2, 5.0);
	gdstep::adam<dtype> st(0.0001);
	shuffle::deterministic sh;
	
	bp.train(net, loss, st, sh, X, Y);
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
		solution(chain, teacher, net, GA_FULL_TRAINING);
		train_member(chain, net, X, Y, GA_TRAIN_EPOCHS);
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


// -------------- Main -------------------

int main() 
{
	umat<dtype> X, X_teacher, X_student, X_valid, X_test;
	umat<dtype> Y, Y_teacher, Y_student, Y_valid, Y_test;
	uvec<int> y, y_train, y_valid, y_test;
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
			evaluate_network(trained_teacher, log, X_teacher, Y_teacher, "Training");
		} else {
			cout << "Nothing more to do.\n";
			return 0;
		}
	}
	if (teacher.load(string(SAVES_FOLDER)+teacher.get_name()+".sav")) {
		cout << "Trained teacher parameters loaded ok.\n";
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
		evaluate_network(teacher, log, X_test, Y_test, "Trained teacher test");
		log << "\n";
	} else {
		cout << "Error loading trained teacher parameters: " << teacher.error_description() << "\n";
		return -1;
	}

	FFNN<dtype> student = create_student(STUDENT_NAME, false);
	if (!student) {
		cout << "Error creating student neural network: " << student.error_description() << "\n";
		return -1;
	}
	// save initial student's weights
	std::vector<weights_and_biases> initial;
	for (int i=1; i<student.nlayers(); ++i) {
		uvec<dtype> ws, bs;
		student.get_layer(i)->get_trainable_parameters(ws, bs);
		initial.push_back(std::make_pair(ws,bs));
	}

	log << teacher.info() << "\n";
	log << statistics(teacher) << "\n";
	log << student.info() << "\n\n";
	log << statistics(student) << "\n";


	std::vector<result> results;



	DO_TEST(true)

	// ========================
	// GA parameters
	// ========================
	int popsize = GA_POP_SIZE;
	int chainsize = 0;
	for (int n: Ns) chainsize += n;
	log << "Number of indeces in each chain: " << chainsize << "\n";
	std::string ga_fname = string(SAVES_FOLDER)+GASAVE_NAME+"p"+to_string(GA_POP_SIZE)+".ga";
	
	GA<int> ga;
	// no openmp parallelization, results are inconsistent due to SGD
	GA<int>::params opt;
	opt.info_iters = 1;
	opt.max_iters = GA_ITERS;
	opt.elitism = 0.02;
	opt.filename = ga_fname;
	opt.autosave = 1;
	opt.parallel = GA<int>::SingleThread;
	ga.set_params(opt);
	ga.set_logging_stream(&log);

	initializer::null<int> noinit;
	initializer::multiset::random<int,0> rndinit(Ns, Nt, true);
	crossover::multiset::onepoint<int> xo(Ns, true);
	mutation::multiset::replace<int> mut(Ns, Nt, true, 0.1, 0.1);

	Fitness ff;
	constexpr int Nsamples = 5000; // 50000
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

	if (1) {
		log << "\nTrain student without KD but with GA initialized filters\n";
		GA<int>::member m = ga.get_member(0);
		log << "Best member: " << tostring(m.chain) << "\n";
		FFNN<dtype> net = create_student(STUDENT_NAME, true);
		for (int i=1; i<net.nlayers(); ++i)
			net.get_layer(i)->set_trainable_parameters(initial[i-1].first, initial[i-1].second);
		solution(m.chain, teacher, net, true);
		lossfunc::softmaxce<dtype> loss;
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, TRAIN_EPOCHS, X, Y, X_valid, Y_valid);
		double acc = evaluate_network(net, log, X_test, Y_test, "Student without KD (but GA initialization): Test");
		results.push_back(std::make_pair(acc, "Student without KD (but GA initialization)"));
		log << "\n";
	}

	if (1) {
		log << "\nTrain student with KD, filters initialized by GA's best member\n";
		GA<int>::member m = ga.get_member(0);
		log << "Best member: " << tostring(m.chain) << "\n";
		FFNN<dtype> net;
		student.clone_to(net);
		solution(m.chain, teacher, net, GA_FULL_TRAINING);
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, TRAIN_EPOCHS, X_student, Y_student, X_valid, Y_valid);
		double acc = evaluate_network(net, log, X_test, Y_test, "Student with KD (GA optimized): Test");
		results.push_back(std::make_pair(acc, "Student with KD (best)"));
		log << "\n";
	}
	if (1 && GA_FULL_TRAINING==false) {
		log << "\nTrain student with KD, filters initialized by GA's best member\n";
		GA<int>::member m = ga.get_member(0);
		log << "Best member: " << tostring(m.chain) << "\n";
		FFNN<dtype> net;
		student.clone_to(net);
		solution(m.chain, teacher, net, true);
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, TRAIN_EPOCHS, X_student, Y_student, X_valid, Y_valid);
		double acc = evaluate_network(net, log, X_test, Y_test, "Student with KD (GA optimized): Test");
		results.push_back(std::make_pair(acc, "Student with KD (best)"));
		log << "\n";
	}

	if (1) {
		log << "\nTrain student with KD, filters choosen by GA's worst member\n";
		GA<int>::member m = ga.get_member(popsize-1);
		log << "Worst member: " << tostring(m.chain) << "\n";
		FFNN<dtype> net;
		student.clone_to(net);
		solution(m.chain, teacher, net, GA_FULL_TRAINING);
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, TRAIN_EPOCHS, X_student, Y_student, X_valid, Y_valid);
		double acc = evaluate_network(net, log, X_test, Y_test, "Student with KD (GA optimized): Test");
		results.push_back(std::make_pair(acc, "Student with KD (worst)"));
		log << "\n";
	}

	constexpr int RUNS = 5;
	double acc_total = 0.0;
	if (1) for (int t=0; t<RUNS; ++t) {
		log << "\nTrain student with KD, filters choosen randomly (full training)\n";
		vector<int> r(chainsize);
		rndinit.apply(r);
		log << "Random" << t+1 << ": " << tostring(r) << "\n";
		FFNN<dtype> net;
		student.clone_to(net);
		solution(r, teacher, net, GA_FULL_TRAINING);
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, TRAIN_EPOCHS, X_student, Y_student, X_valid, Y_valid);
		double acc = evaluate_network(net, log, X_test, Y_test, "Student with KD (Random): Test");
		acc_total += acc;
		log << "\n";
	}
	results.push_back(std::make_pair(acc_total/RUNS, "Student with KD (Random, Average)"));

	END_TEST


	bool normal_kd = true;

	// train student without KD
	if (normal_kd) {
		log << "\nTrain student without KD\n";
		FFNN<dtype> net = create_student(STUDENT_NAME, true);
		for (int i=1; i<net.nlayers(); ++i)
			net.get_layer(i)->set_trainable_parameters(initial[i-1].first, initial[i-1].second);
		lossfunc::softmaxce<dtype> loss;
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, TRAIN_EPOCHS, X, Y, X_valid, Y_valid);
		double acc = evaluate_network(net, log, X_test, Y_test, "Student without KD: Test");
		results.push_back(std::make_pair(acc, "Student without KD (normal initialization)"));
		log << "\n";
	}

	// train student with KD
	if (normal_kd) {
		log << "\nTrain student with normal KD\n";
		FFNN<dtype> net = create_student(STUDENT_NAME, false);
		for (int i=1; i<net.nlayers(); ++i)
			net.get_layer(i)->set_trainable_parameters(initial[i-1].first, initial[i-1].second);
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		train_network(net, log, loss, st, sh, TRAIN_EPOCHS, X_student, Y_student, X_valid, Y_valid);
		double acc = evaluate_network(net, log, X_test, Y_test, "Student with KD: Test");
		results.push_back(std::make_pair(acc, "Student with KD (normal initialization)"));
		log << "\n";
	}

	// results
	log << "RESULTS:\n";
	for (auto r : results) {
		log << std::fixed << std::setprecision(6) << r.first << " " << r.second << "\n";
	}

	return 0;
}
