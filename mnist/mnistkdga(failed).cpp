/*
 1. KMeans to generate N clusters
 2. GA to find which clusters to use in KD
 */

#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "kmeans.hpp"
#include "nn/backprop.hpp"
#include "bio/ga.hpp"
#include "datasets/mnistloader.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }

#define SAVES_FOLDER         "../../saves/mnist/"
#define TRAINED_TEACHER_NAME "mnistkdga-trained-teacher(clustering)"
#define STUDENT_NAME         "mnistkdga-student"
#define CLUSTERING_FNAME     "mnist_kmeans_clusters_12.dat"

// Datatype for data and neural network
typedef float dtype;

// Seed for local RNG
constexpr int SEED = 48;

// Number of clusters for KMeans
constexpr int NClusters = 12;

// Number of samples per cluster for GA. Set to 0 to use all samples.
constexpr int NSPC = 20;

// Batch size for GA training (fitness function)
constexpr int BATCH_SIZE  = 10;

// GA population size
constexpr int GA_POP_SIZE = 15;

// GA generations
constexpr int GA_ITERS = 5;

// GA encoding
typedef std::pair<uvec<dtype>, uvec<dtype>> weights_and_biases;


// Activation functions
constexpr int fconv = fReLU;
constexpr int cnn_ffc = fReLU;
constexpr int ff_ffc = fLogistic;

FFNN<dtype> create_teacher(const std::string& name, bool softmax)
{
	// 6, 16, 120, 84
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(12,5,1,Valid,fconv)); 
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(32,5,1,Same,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(120,4,1,Same,fconv));
	net.add(new DenseLayer<dtype>(84, cnn_ffc));
	if (softmax) net.add(new SoftmaxLayer<dtype>(10));
	else net.add(new DenseLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_student(const std::string& name, bool softmax)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(6,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(8,5,1,Same,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(30,4,1,Same,fconv));
	net.add(new DenseLayer<dtype>(20, cnn_ffc));
	if (softmax) net.add(new SoftmaxLayer<dtype>(10));
	else net.add(new DenseLayer<dtype>(10));
	return net;
}



// -------------- GA -------------------

string tostring(const vector<bool>& chain)
{
	stringstream ss;
	for (auto a : chain) ss << (a ? "1" : "0");
	return ss.str();
}

// converts a chain to a solution
template <typename dtype=float>
void solution(const std::vector<bool>& chain, 
			  const umat<dtype>& X, const umat<dtype>& Y, const uvec<int>& y_cl, int spc,
			  umat<dtype>& X_train, umat<dtype>& Y_train)
{
	assert(X.ydim() == Y.ydim());
	assert(X.ydim() == y_cl.len());
	int nclusters = (int)chain.size();
	std::vector<int> counts(nclusters);
	std::vector<int> selected;
	std::vector<int> shuffled;
	rng32 local_rng;
	local_rng.seed(SEED, SEED+13);
	build_shuffled_indeces(shuffled, y_cl.len(), local_rng);
	for (int i=0; i<nclusters; ++i) {
		if (chain[i]) {
			for (int j=0; j<y_cl.len(); ++j) {
				if (y_cl(shuffled[j])==i) {
					selected.push_back(shuffled[j]);
					counts[i]++;
					if (counts[i]==spc) break;
				}
			}
		}
	}
	int n = (int)selected.size();
	X_train.resize(n, X.xdim());
	Y_train.resize(n, Y.xdim());
	for (int i=0; i<n; ++i) {
		X_train.set_row(i, X.row_offset(i), X.xdim());
		Y_train.set_row(i, Y.row_offset(i), Y.xdim());
	}
}

// trains a neural network using backpropagation
template <typename dtype=float>
void train_member(const std::vector<bool>& chain, FFNN<dtype>& net, 
		   		  const umat<dtype>& X, const umat<dtype>& Y, const uvec<int>& y_cl, int iters)
{
	umat<dtype> X_train, Y_train;

	solution(chain, X, Y, y_cl, NSPC, X_train, Y_train);

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
	
	bp.train(net, loss, st, sh, X_train, Y_train);
}

// fitness function
// determines how good a solution is
struct Fitness {
	umat<dtype> X, Y, Xv, Yv;
	uvec<int> y_cl;
	std::vector<weights_and_biases> wbs;
	double operator ()(const std::vector<bool>& chain) {
		FFNN<dtype> net = create_student(STUDENT_NAME, false);
		for (int i=1; i<net.nlayers(); ++i)
			net.get_layer(i)->set_trainable_parameters(wbs[i-1].first, wbs[i-1].second);
		train_member(chain, net, X, Y, y_cl, 20);
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


// -------------- Main -------------------

int main() 
{
	umat<dtype> X, X_teacher, X_student, X_valid, X_test;
	umat<dtype> Y, Y_teacher, Y_student, Y_valid, Y_test;
	uvec<int> y, y_train, y_valid, y_test, y_pred, y_train_pred;
	uvec<int> clustering;
	steady_clock::time_point t1, t2;

	Logger log("mnistkdga-clustering.log");

	// set openmp threads to maximum
	umml_set_openmp_threads();

	// seed RNG
	umml_seed_rng(SEED);

	// file names
	string GASAVE_FNAME  = "mnistkdga.ga";
	string NOTUSED_FNAME = "clusters_not_used.csv";

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

	// choose 38K random samples (to test if GA really works)
	if (0) {
		df.shuffle(X, y);
		X.reshape(38000, X.xdim());
		y.reshape(38000);
	}

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

	// select student's training set
	if (false) {
		uvec<int> idcs;
		idcs = df.select(y, 7);
		X_student = df.vstack(X_student, df.copy_rows(X, idcs));
		Y_student = df.vstack(Y_student, df.copy_rows(Y, idcs));
		idcs = df.select(y, 8);
		X_student = df.vstack(X_student, df.copy_rows(X, idcs));
		Y_student = df.vstack(Y_student, df.copy_rows(Y, idcs));	
	} else {
		X_student = X; 
		Y_student = Y;
	}

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
		cout << "Trainined teacher network not found. Do you want to start training it? (y/n) ";
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
		cout << "Trainined teacher parameters loaded ok.\n";
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

	// save student's initial weights
	std::vector<weights_and_biases> wbs;
	for (int i=1; i<student.nlayers(); ++i) {
		uvec<dtype> ws, bs;
		student.get_layer(i)->get_trainable_parameters(ws, bs);
		wbs.push_back(std::make_pair(ws,bs));
	}

	log << teacher.info() << "\n";
	log << student.info() << "\n\n";



	DO_TEST(false)

	// ========================
	// Clustering (KMeans)
	// ========================
	KMeans<dtype> km;
	{
	KMeans<dtype>::params opt;
	opt.info_iters = 10;
	km.set_params(opt);
	}
	bool random_init = true;
	bool recluster = true;
	if (km.load(string(SAVES_FOLDER)+CLUSTERING_FNAME) && km.n_clusters()==NClusters) {
		cout << km.n_clusters() << " clusters found in the disk file.\n";
		recluster = false;
	} else {
		cout << "Clusters not found (expecting " << NClusters << ", found "
			 << km.n_clusters() << "). Do you want to do a reclustering (y/n)? ";
		string answer;
		cin >> answer;
		if (answer[0]=='n' || answer[0]=='N') {
			cout << "Nothing more to do.\n";
			return 0;
		}
	}
	if (recluster) {
		km.unitialize();
		if (!random_init) {
			umat<dtype> X_seed(NClusters, X.xdim());
			for (int i=0; i<NClusters; ++i) {
				int digit = i % 10;
				int pos = y.find_random(digit);
				X_seed.set_row(i, X.row_offset(pos), X_seed.xdim());
			}
			km.seed(NClusters, X_seed);
		}
		log << "Clustering...\n";
		t1 = chrono::steady_clock::now();
		clustering = km.fit(NClusters, X);
		log << "clustering.len=" << clustering.len() << "\n";
		t2 = chrono::steady_clock::now();
		log << NClusters << " clusters created in " << format_duration(t1, t2) << ".\n";
		km.save(string(SAVES_FOLDER)+CLUSTERING_FNAME);
	} else {
		clustering = km.cluster(X);
	}

	{
	umat<dtype> centr;
	uvec<int> spc;
	km.clustering_info(centr, spc);
	//log << "Samples per cluster: " << spc.format() << " (silhouette coefficient: " << silhouette_coefficient(X, centr) << "\n\n";
	}


	// ========================
	// GA parameters
	// ========================
	int popsize = GA_POP_SIZE;
	int tot_bits = NClusters;
	log << "Number of bits in each bitstring: " << tot_bits << "\n";
	log << "Max number of samples per cluster in training: " << NSPC << "\n";
	
	GA<bool> ga;
	// no openmp parallelization, results are inconsistent due to SGD
	GA<bool>::params opt;
	opt.info_iters = 1;
	opt.max_iters = GA_ITERS;
	opt.elitism = 0.02;
	opt.filename = string(SAVES_FOLDER)+GASAVE_FNAME;
	opt.autosave = 1;
	opt.parallel = GA<bool>::SingleThread;
	ga.set_params(opt);
	ga.set_logging_stream(&log);

	initializer::null<bool> noinit;
	initializer::values::random<bool> rndinit;
	//crossover::values::onepoint<bool> xo;
	//crossover::values::twopoint<bool> xo;
	crossover::values::uniform<bool> xo;
	mutation::values::flip<bool> mut(0.1, 0.01);

	Fitness ff;
	ff.X.resize_like(X_student); 
	ff.X.set(X_student);
	ff.Y.resize_like(Y_student); 
	ff.Y.set(Y_student);
	ff.Xv.resize_like(X_valid);
	ff.Xv.set(X_valid);
	ff.Yv.resize_like(Y_valid);
	ff.Yv.set(Y_valid);
	ff.y_cl = clustering;
	ff.wbs = wbs;
	
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
			ga.init(tot_bits, popsize, rndinit);
		} else {
			ga.init(tot_bits, popsize, noinit);
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
	for (int i=0; i<std::min(10,popsize); ++i) {
		GA<bool>::member m = ga.get_member(i);
		log << std::fixed;
    	log << std::setprecision(6);
		log << "member " << std::setw(3) << i+1 << ", fitness=" << m.fitness << ", " << tostring(m.chain) << "\n";
	}

	// train student with KD but with a subset of training set choosen by the GA
	if (1) {
		log << "\nTrain student with KD but with a subset of training set choosen by the GA\n";
		GA<bool>::member m = ga.get_member(0);
		log << "Best member: " << tostring(m.chain) << "\n";
		umat<dtype> X_ga, Y_ga;
		solution(m.chain, X_student, Y_student, clustering, 0, X_ga, Y_ga);
		FFNN<dtype> student = create_student(STUDENT_NAME, false);
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		for (int i=1; i<student.nlayers(); ++i)
			student.get_layer(i)->set_trainable_parameters(wbs[i-1].first, wbs[i-1].second);
		train_network(student, log, loss, st, sh, 50, X_ga, Y_ga, X_valid, Y_valid);
		umat<dtype> Y_pred1hot = evaluate_network(student, log, X_test, Y_test, "Student with KD (GA optimized): Test");
		log << "\n";
	}

	// create a csv file with the indeces of clusters that didn't used in training
	ofstream os(string(SAVES_FOLDER)+NOTUSED_FNAME);
	os << "cluster,index,digit\n";
	for (int i=0; i<NClusters; ++i) {
		GA<bool>::member m = ga.get_member(0);
		if (!m.chain[i]) {
			for (int j=0; j<clustering.len(); ++j) {
				if (clustering(j)==i) {
					os << i << "," << j << "," << y(j) << "\n";
				}
			}
		}
	}

	END_TEST


	// train student without KD
	if (1) {
		log << "\nTrain student without KD\n";
		FFNN<dtype> student = create_student(STUDENT_NAME, true);
		lossfunc::softmaxce<dtype> loss;
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		for (int i=1; i<student.nlayers(); ++i)
			student.get_layer(i)->set_trainable_parameters(wbs[i-1].first, wbs[i-1].second);
		train_network(student, log, loss, st, sh, 25, X, Y, X_valid, Y_valid);
		umat<dtype> Y_pred1hot = evaluate_network(student, log, X_test, Y_test, "Student without KD: Test");
		log << "\n";
	}

	// train student with KD
	if (1) {
		log << "\nTrain student with KD\n";
		FFNN<dtype> student = create_student(STUDENT_NAME, false);
		lossfunc::kdloss<dtype> loss(0.2, 5.0);
		gdstep::adam<dtype> st(0.0001);
		shuffle::deterministic sh;
		for (int i=1; i<student.nlayers(); ++i)
			student.get_layer(i)->set_trainable_parameters(wbs[i-1].first, wbs[i-1].second);
		train_network(student, log, loss, st, sh, 25, X_student, Y_student, X_valid, Y_valid);
		umat<dtype> Y_pred1hot = evaluate_network(student, log, X_test, Y_test, "Student with KD: Test");
		log << "\n";
	}


	return 0;
}
