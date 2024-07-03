#include "cpuinfo.hpp"
#include "metrics.hpp"
#include "dataframe.hpp"
#include "datasets/mnistloader.hpp"
#include "nn/ffnn.hpp"
#include "nn/backprop.hpp"
#include "preproc.hpp"
#include "bio/ga.hpp"
#include "bio/bitstr.hpp"
#include <iostream>

using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }


typedef float dtype;

#define   SAVES_FOLDER  "../../saves/mnist/"
#define   NET_NAME      "mnistpermut-cnn"
#define   GASAVE_FNAME  "mnistpermut.ga"

constexpr int     OMP_THREADS   = 12;

constexpr int     BATCH_SIZE    = 10;
constexpr double  LEARNING_RATE = 0.01;
constexpr int     TRAIN_EPOCHS  = 2;

constexpr int     SEED          = 48;
constexpr int     NSPC          = 30*BATCH_SIZE;
constexpr int     POP_SIZE      = 25;
constexpr int     GENERATIONS   = 15;



string tostring(const vector<int>& chain)
{
	stringstream ss;
	for (auto a : chain) ss << a << " ";
	return ss.str();
}

template <typename Type>
void shuffle(uvec<Type>& v, int first, int last, rng32& rng)
{
	for (int i=last; i>first; --i) {
		int j = (int)(rng.generate() % (i+1));
		Type tmp = v(i);
		v(i) = v(j);
		v(j) = tmp;
	}
}

template <typename Type>
void make_sets(umat<Type>& X_train, umat<Type>& Y_train, 
			   const umat<Type>& X, const umat<Type>& Y, const uvec<int>& y, 
			   const std::vector<int>& ordering, int spc)
{
	std::vector<int> rows;
	std::vector<int> counts(ordering.size(), 0);
	for (int k: ordering) {
		if (spc > 0 && counts[k] >= spc) continue;
		for (int i=0; i<y.len(); ++i) {
			int digit = y(i);
			if (digit==k) {
				rows.push_back(i);
				counts[k]++;
				if (spc > 0 && counts[k] >= spc) break;
			}
		}
	}
	int n = (int)rows.size() / ordering.size();
	X_train.resize(n*ordering.size(), X.xdim());
	Y_train.resize(n*ordering.size(), Y.xdim());
	int k = 0;
	for (int i=0; i<n; ++i) {
		for (int j=0; j<(int)ordering.size(); ++j) {
			int idx = i + j*n;
			X_train.set_row(k, X.row_offset(rows[idx]).get_cmem(), X.xdim());
			Y_train.set_row(k, Y.row_offset(rows[idx]).get_cmem(), Y.xdim());
			k++;
		}
	}
}


int fconv = fReLU;
int cnn_ffc = fReLU;
int ff_ffc = fLogistic;

FFNN<dtype> create_cnn(const string& name)
{
	FFNN<dtype> net(name);
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

FFNN<dtype> create_ffnn(const string& name)
{
	FFNN<dtype> net(name);
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	//net.add(new DenseLayer<dtype>(32, ff_ffc));
	//net.add(new DenseLayer<dtype>(32, ff_ffc));
	net.add(new SoftmaxLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_network(const string& name)
{
	return create_ffnn(name);
	//return create_cnn(name);
}



// Training
// ------------------------------

template <typename Type, class Loss, class Stepper, class Shuffle>
void train_network(FFNN<Type>& net, Logger& log, Loss& loss, Stepper& st, Shuffle& sh, int epochs,
				   const umat<Type>& X_train, const umat<Type>& Y_train, 
				   const umat<Type>& X_valid, const umat<Type>& Y_valid)
{
	steady_clock::time_point t1, t2;
	Backprop<dtype> bp;
	{
	Backprop<dtype>::params opt;
	opt.batch_size = BATCH_SIZE;
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
umat<Type> evaluate_network(FFNN<Type>& net, Logger& log, const umat<Type>& X, const umat<Type>& Y,
							onehot_enc<>& enc, const uvec<int>& y, const string& setname)
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

	uvec<int> y_pred;
	if (!enc.decode(Y_pred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
	confmat<> cm = confusion_matrix<>(y, y_pred, CM_Macro);
	log << "Accuracy  = " << accuracy<>(cm) << "\n";
	log << "Precision = " << precision<>(cm) << "\n";
	log << "Recall    = " << recall<>(cm) << "\n";
	log << "F1        = " << F1<>(cm) << "\n";
	log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";

	return Y_pred1hot;
}


 
// GA
// ------------------------------

// converts a chain to a solution
template <typename Type=float>
void solution(const std::vector<int>& chain, int spc,
			  const umat<Type>& X, const umat<Type>& Y, const uvec<int>& clusters,
			  umat<Type>& X_train, umat<Type>& Y_train)
{
	rng32 rng;
	rng.seed(SEED, SEED+13);
	std::vector<int> idcs;
	build_shuffled_indeces(idcs, X.ydim(), rng);
	umat<Type> tmpX = X;
	umat<Type> tmpY = Y;
	uvec<int>  tmpc = clusters;
	tmpX.copy_rows(X, idcs);
	tmpY.copy_rows(Y, idcs);
	tmpc.copy(clusters, idcs);
	make_sets(X_train, Y_train, tmpX, tmpY, tmpc, chain, spc);
}


// trains a neural network using backpropagation
template <typename dtype=float>
void train_member(const std::vector<int>& chain, FFNN<dtype>& net, 
		   const umat<dtype>& X, const umat<dtype>& Y, const uvec<int>& clusters, int spc, int iters)
{
	umat<dtype> X_train, Y_train;

	solution(chain, spc, X, Y, clusters, X_train, Y_train);

	Backprop<dtype> bp;
	{
	// stochastic must be set to false in order for the GA to work
	typename Backprop<dtype>::params opt;
	opt.batch_size = BATCH_SIZE;
	opt.max_iters = iters;
	opt.info_iters = 1;
	opt.verbose = true;
	bp.set_params(opt);
	}
	lossfunc::softmaxce<dtype> loss;
	//gdstep::adam<dtype> st(0.0001);
	gdstep::learnrate<dtype> st(LEARNING_RATE);
	shuffle::null sh;
	
	bp.train(net, loss, st, sh, X_train, Y_train);
}


// fitness function
// determines how good a solution is
struct Fitness {
	umat<dtype> X, Y, Xv, Yv;
	uvec<int> clusters;
	std::vector<uvec<dtype>> weights;
	std::vector<uvec<dtype>> biases;
	double operator ()(const std::vector<int>& chain) {
		FFNN<dtype> net = create_network("");
		for (int i=1; i<net.nlayers(); ++i)
			net.get_layer(i)->set_trainable_parameters(weights[i-1], biases[i-1]);
		train_member(chain, net, X, Y, clusters, NSPC, 1);
		umat<dtype> Y_pred;
		umat<dtype> Y_pred1hot;
		net.predict(Xv, Y_pred);
		Y_pred1hot.resize_like(Y_pred);
		Y_pred1hot.argmaxto1hot(Y_pred);
		double acc = accuracy(Yv, Y_pred1hot);
		double f = acc;
		//double f = std::pow(1+acc, 5) / std::pow(2,5);
		//double f = std::exp(6*acc-6.0);
		//double f = (acc < 0.95 ? std::tanh(2.5*acc) : acc);

		return f;
	}
};



int main()
{
	umat<dtype> X, X_train, X_valid, X_test;
	umat<dtype> Y, Y_train, Y_valid, Y_test;

	uvec<int> y, y_train, y_valid, y_test;
	steady_clock::time_point t1, t2;

	Logger log("mnistpermut.log");

	umml_set_openmp_threads(OMP_THREADS);
	umml_seed_rng(48);


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

	// re-arrange samples
	constexpr int NSAMPLES = 4500;
	int first=0, last;
	std::vector<std::pair<int,int>> classes;
	for (int cls=0; cls<=9; ++cls) {
		uvec<int> idcs;
		idcs = df.select(y, cls);
		idcs.reshape(NSAMPLES);
		X_train = df.vstack(X_train, df.copy_rows(X, idcs));
		Y_train = df.vstack(Y_train, df.copy_rows(Y, idcs));
		y_train = df.append(y_train, df.copy(y, idcs));
		last = X_train.ydim()-1;
		log << "first:" << first << ", last:" << last << ", count=" << last-first+1 << "\n";
		classes.push_back(std::pair<int,int>(first, last));
		first = last+1;
	}
	log << "X_train: " << X_train.shape() << "\n";


	// create the neural network
	FFNN<dtype> net = create_network(NET_NAME);
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


	DO_TEST(true)

	// ========================
	// GA parameters
	// ========================
	int popsize = POP_SIZE;
	int N = 10;
	log << "Number of clusters in each bitstring: " << N << "\n";
	
	GA<int> ga;
	// no openmp parallelization, results are inconsistent due to SGD
	GA<int>::params opt;
	opt.info_iters = 1;
	opt.max_iters = GENERATIONS;
	opt.elitism = 0.02;
	opt.filename = string(SAVES_FOLDER)+GASAVE_FNAME;
	opt.autosave = 1;
	opt.parallel = GA<int>::SingleThread;
	ga.set_params(opt);
	ga.set_logging_stream(&log);

	initializer::null<int> noinit;
	initializer::permut::random<int,0> rndinit(N);
	crossover::permut::onepoint<int> xo;
	mutation::permut::swap<int> mut(0.1, 0.01);

	Fitness ff;
	ff.X.resize_like(X_train); 
	ff.X.set(X_train);
	ff.Y.resize_like(Y_train); 
	ff.Y.set(Y_train);
	ff.clusters.resize_like(y_train); 
	ff.clusters.set(y_train);
	ff.Xv.resize_like(X_valid);
	ff.Xv.set(X_valid);
	ff.Yv.resize_like(Y_valid);
	ff.Yv.set(Y_valid);
	ff.weights = weights;
	ff.biases = biases;
	
	string answer = "n";
	if (1) if (check_file_exists(opt.filename)) {
		cout << "A population is found in `" << opt.filename << "`. Please choose:\n";
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
			ga.init(N, popsize, rndinit);
		} else {
			ga.init(N, popsize, noinit);
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
		GA<int>::member m = ga.get_member(i);
		log << "member " << std::setw(3) << i+1 << " [" << tostring(m.chain) << "], fitness=" << m.fitness << "\n";
	}

	// training and test accuracy with the best member
	umat<dtype> Y_pred;
	umat<int> Y_pred1hot;
	uvec<int> y_pred;
	t1 = chrono::steady_clock::now();
	train_member(ga.get_member(0).chain, net, X_train, Y_train, y_train, 0, TRAIN_EPOCHS);
	t2 = chrono::steady_clock::now();
	log << "Trained in " << format_duration(t1, t2) << ".\n";

	evaluate_network(net, log, X_valid, Y_valid, enc, y_valid, "Validation");
	log << "\n";
	
	evaluate_network(net, log, X_test, Y_test, enc, y_test, "Test");
	log << "\n";

	END_TEST




	// train with standard SGD
	if (1) {
		FFNN<dtype> net = create_network(NET_NAME);
		lossfunc::softmaxce<dtype> loss;
		//gdstep::adam<dtype> st(0.0001);
		gdstep::learnrate<dtype> st(LEARNING_RATE);
		shuffle::deterministic sh;
		//shuffle::null sh;
		for (int i=1; i<net.nlayers(); ++i)
			net.get_layer(i)->set_trainable_parameters(weights[i-1], biases[i-1]);
		log << "\n";
		train_network(net, log, loss, st, sh, TRAIN_EPOCHS, X_train, Y_train, X_valid, Y_valid);
		evaluate_network(net, log, X_test, Y_test, enc, y_test, "Standard KD: Test");
		log << "\n";
	}

	return 0;
}
