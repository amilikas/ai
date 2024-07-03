// Train a FFNN with a GA+PSO
//
// Creates popsize networks in parallel, trains them for some epochs
// and then combines them via crossover (averaging)
//
// Represantation: floats (weights and biases)
//   (I1)  (I2) 
//    
// (H1) (H2) (H3)
//  
//   (O1) (O2)
//
// Genome:
//  w1h1, w2h2, w3h3, v1o1, v2o2  =>  w11w12h1, w21w22h2, w31w32h3, v11v12v13o1, v21v22v23o2
//                                    0  1  2   3  4  5   6  7  8   9  10 11 12  13 14 15 16 
// Genes (pos,ws,bs)
//                                    {0,2,1}   {3,2,1}   {6,2,1}   {9,3,1}      {13,3,1}
// Genome (layers & genes)
//                                    [0,3]                         [9,2] 
// Crossover/Mutation points
// - Macro (neuron level)             0(0)      1(3)      2(6)      3(9)         4(13)

#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "stats.hpp"
#include "bio/ga.hpp"
#include "bio/pso.hpp"
#include "nn/ffnn.hpp"
#include "datasets/mnistloader.hpp"
#include <iostream>

using namespace std;
using namespace std::chrono;
using namespace umml;

#define DO_TEST(x) if(x){
#define END_TEST   }


// data type
typedef float dtype;

// Gene type
// layer
// weights: (pos,wsize)
// biases:  (pos+wsize, bsize)
struct Gene {
	int layer;
	int pos;
	int wsize;
	int bsize;
};

// genome type
using Genome = std::vector<Gene>;


#define LOG_NAME      "mnist-gapso.log"
#define SAVES_FOLDER  "../../saves/mnist/"
#define GASAVE_NAME   "mnist-gapso.ga"

// Seed for local RNG
constexpr int    SEED = 48;

// GA parameters
constexpr int    GA_POP_SIZE        = 50;
constexpr int    GA_ITERS           = 150;
constexpr int    GA_MUT_TRAIN_ITERS = 10;
constexpr double GA_MUT_PROB        = 0.2;
constexpr double GA_MUT_SWARM_SIZE  = 1000;


// activation functions
int fconv = fReLU;
int ffc = fReLU;

FFNN<dtype> create_cnn()
{
	FFNN<dtype> net("mnist-gapso-cnn");
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	net.add(new Conv2DLayer<dtype>(6,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(16,5,1,Valid,fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new Conv2DLayer<dtype>(120,4,1,Valid,fconv));
	net.add(new DenseLayer<dtype>(84, ffc));
	net.add(new SoftmaxLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_ffnn()
{
	FFNN<dtype> net("mnist-gapso-ff");
	net.add(new InputLayer<dtype>(dims3{28,28,1}));
	//net.add(new DenseLayer<dtype>(64, ffc));
	//net.add(new DenseLayer<dtype>(32, ffc));
	net.add(new SoftmaxLayer<dtype>(10));
	return net;
}


FFNN<dtype> create_network()
{
	//return create_cnn();
	return create_ffnn();
}


void   train_nn(FFNN<dtype>& nn, int iters, int info, const umat<dtype>& X, const umat<dtype>& Y);
double evaluate_nn(FFNN<dtype>& nn, const umat<dtype>& X, const umat<dtype>& Y);

void nn_to_genome(const FFNN<dtype>& net, Genome& genome)
{
	int pos = 0;
	genome.clear();
	for (int l=1; l<net.nlayers(); ++l) {
		int wsize = net.get_layer(l)->weights_size();
		int bsize = net.get_layer(l)->biases_size();
		dims4 wdims = net.get_layer(l)->weights_dims()[0];
		if (wsize > 0) {
			int ng = wsize / wdims.x;
			int gw = wsize / wdims.y;
			int gb = bsize / wdims.y;
			for (int i=0; i<ng; ++i) {
				Gene g;
				g.layer = l;
				g.pos   = pos;
				g.wsize = gw;
				g.bsize = gb;
				genome.push_back(g);
				pos += (g.wsize + g.bsize);
			}
		}
	} 
}

void nn_to_chain(const FFNN<dtype>& net, std::vector<dtype>& chain)
{
	chain.clear();
	for (int l=1; l<net.nlayers(); ++l) {
		uvec<dtype> ws, bs;
		net.get_layer(l)->get_trainable_parameters(ws, bs);
		int wsize = ws.len();
		int bsize = bs.len();
		dims4 wdims = net.get_layer(l)->weights_dims()[0];
		if (wsize > 0) {
			int ng = wsize / wdims.x;
			int gw = wsize / wdims.y;
			int gb = bsize / wdims.y;
			for (int i=0; i<ng; ++i) {
				for (int j=0; j<gw; ++j) chain.push_back(ws(i*gw+j));
				for (int j=0; j<gb; ++j) chain.push_back(bs(i*gb+j));
			}
		}
	} 
}

void chain_to_nn(const std::vector<dtype>& chain, FFNN<dtype>& net)
{
	int pos = 0;
	for (int l=1; l<net.nlayers(); ++l) {
		uvec<dtype> ws, bs;
		ws.resize(net.get_layer(l)->weights_size());
		bs.resize(net.get_layer(l)->biases_size());
		int wsize = ws.len();
		int bsize = bs.len();
		dims4 wdims = net.get_layer(l)->weights_dims()[0];
		if (wsize > 0) {
			int ng = wsize / wdims.x;
			int gw = wsize / wdims.y;
			int gb = bsize / wdims.y;
			for (int i=0; i<ng; ++i) {
				for (int j=0; j<gw; ++j) ws(i*gw+j) = chain[pos++];
				for (int j=0; j<gb; ++j) bs(i*gb+j) = chain[pos++];
			}
		}
		net.get_layer(l)->set_trainable_parameters(ws, bs);
	} 
}

string format_genome(const Genome& genome)
{
	stringstream ss;
	ss << std::fixed;
	for (const Gene& g : genome)
		ss << "{" << g.layer << ":" << g.pos << "," << g.wsize << "," << g.bsize << "} ";
	return ss.str();
}

string format_chain(const vector<dtype>& chain)
{
	stringstream ss;
	ss << std::fixed;
	ss << std::setprecision(4);
	for (dtype a : chain) ss << a << " ";
	return ss.str();
}

string output_member(const GA<dtype>::member& m, int i) 
{
	stringstream ss;
	ss << std::fixed;
   	ss << std::setprecision(6);
	//ss << "member " << std::setw(3) << i+1 << ", fitness=" << m.fitness << ", " << tostring(m.chain) << "\n";
	ss << "member " << std::setw(3) << i+1 << ", fitness=" << m.fitness << "\n";
	return ss.str();
}

int find_gene_with_pos(const Genome& genome, int pos)
{
	for (int s=0; s<(int)genome.size(); ++s) {
		if (genome[s].pos==pos) return s;
	}
	return -1;
}

int find_first_gene_with_layer(const Genome& genome, int layer)
{
	for (int s=0; s<(int)genome.size(); ++s) {
		if (genome[s].layer==layer) return s;
	}
	return -1;
}

int find_last_gene_with_layer(const Genome& genome, int first, int layer)
{
	if (first >= 0) {
		int s = first+1;
		while (s < (int)genome.size() && genome[s].layer==layer) s++;
		return s-1;
	}
	return -1;
}

/*
// gene similarity (to handle hidden layers permutations)
// uses sum of squares to determine similar genes
int similar_gene(const Genome& genome, const std::vector<dtype>& chain1, int pos1, const std::vector<dtype>& chain2)
{
	int s1 = find_gene_with_pos(genome, pos1);
	if (s1 >= 0) {
		const Gene& g1 = genome[s1];
		int first = find_first_gene_with_layer(genome, g1.layer);
		int last  = find_last_gene_with_layer(genome, first, g1.layer);
		assert(last != -1);
		dtype sumsq1 = dtype(0);
		for (int i=g1.pos; i<g1.pos+g1.wsize; ++i) sumsq1 += chain1[i]*chain1[i];
		int dmin_pos = 0;
		dtype dmin = std::numeric_limits<dtype>::max();
		for (int s=first; s<=last; ++s) {
			const Gene& g2 = genome[s];
			dtype sumsq2 = dtype(0);
			for (int i=g2.pos; i<g2.pos+g2.wsize; ++i) sumsq2 += chain2[i]*chain2[i];
			dtype d = std::abs(sumsq1-sumsq2);
			if (d < dmin) {
				dmin = d;
				dmin_pos = g2.pos;
			}
		}
		return dmin_pos;
	}
	return -1;
}
*/


dtype similarity_metric(const std::vector<dtype>& chain, int pos, int n)
{
	dtype sumsq = dtype(0);
	for (int i=pos; i<pos+n; ++i) sumsq += chain[i]*chain[i];
	dtype stdev = std::sqrt(sumsq);
	sumsq = dtype(0);
	for (int i=pos; i<pos+n; ++i) {
		dtype v = chain[i] / stdev; 
		sumsq += v * v;
	}
	return sumsq;
	/*
	dtype sumsq = dtype(0);
	for (int i=pos; i<pos+n; ++i) sumsq += chain[i]*chain[i];
	return sumsq;
	*/
}

// gene similarity (to handle hidden layers permutations)
// uses sum of squares of nomalized vectors yo determine similar genes
int similar_gene(const Genome& genome, const std::vector<dtype>& chain1, int pos1, const std::vector<dtype>& chain2)
{
	int s1 = find_gene_with_pos(genome, pos1);
	if (s1 >= 0) {
		const Gene& g1 = genome[s1];
		int first = find_first_gene_with_layer(genome, g1.layer);
		int last  = find_last_gene_with_layer(genome, first, g1.layer);
		assert(last != -1);

		dtype sim1 = similarity_metric(chain1, g1.pos, g1.wsize);
		int dmin_pos = 0;
		dtype dmin = std::numeric_limits<dtype>::max();
		for (int s=first; s<=last; ++s) {
			const Gene& g2 = genome[s];
			dtype sim2 = similarity_metric(chain2, g2.pos, g2.wsize);
			dtype d = std::abs(sim1-sim2);
			if (d < dmin) {
				dmin = d;
				dmin_pos = g2.pos;
			}
		}
		return dmin_pos;
	}
	return -1;
}

// member initialization
struct nninit {
	using Vals = std::vector<dtype>;
	nninit() {}
	void apply(Vals& m) {
		nn_to_chain(create_network(), m);
	}
	std::string info() const { return "nninit"; }
};


// nntrain test 
enum {
	nnmerge_AvgFull,
	nnmerge_AvgSome,
	nnmerge_Uniform,
	nnmerge_AvgFullAvgSome,
	nnmerge_AvgFullUniform,
	nnmerge_AvgSomeUniform,
	nnmerge_All,
};

std::string test_mode_str(int test_mode) {
	switch (test_mode) {
		case nnmerge_AvgFull: return "AvgFull";
		case nnmerge_AvgSome: return "AvgSome";
		case nnmerge_Uniform: return "Uniform";
		case nnmerge_AvgFullAvgSome: return "Full+Some";
		case nnmerge_AvgFullUniform: return "Full+Uniform";
		case nnmerge_AvgSomeUniform: return "Some+Uniform";
	}
	return "Full+Some+Uniform";
}

// crossover: merge two neural networks
struct nnmerge {
	using Vals = std::vector<dtype>;

	Genome genome;
	int test;

	nnmerge(const Genome& __genome, int __test): genome(__genome), test(__test) {}

	void average_all(const Vals& p1, const Vals& p2, Vals& c1) {
		int n = (int)p1.size();
		for (int i=0; i<n; ++i) c1[i] = (p1[i]+p2[i])/2.0;
	}

	void average_some(const Vals& p1, const Vals& p2, Vals& c1) {
		c1 = p1;
		for (size_t s=0; s<genome.size(); ++s) {
			const Gene& g = genome[s];
			double coin = uniform_random_real<double>(0.0, 1.0);
			if (coin <= 0.5) {
				int n = g.wsize + g.bsize;
				int p2_pos = similar_gene(genome, p1, g.pos, p2);
				for (int j=0; j<n; ++j) c1[g.pos+j] = (p1[g.pos+j]+p2[p2_pos+j])/2.0;
			}
		}
	}

	void average_some(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		c1 = p1;
		c2 = p2;
		std::vector<int> averaged;
		for (size_t s=0; s<genome.size(); ++s) {
			const Gene& g = genome[s];
			double coin = uniform_random_real<double>(0.0, 1.0);
			if (coin <= 0.5) {
				averaged.push_back(s);
				int n = g.wsize + g.bsize;
				int p2_pos = similar_gene(genome, p1, g.pos, p2);
				for (int j=0; j<n; ++j) c1[g.pos+j] = (p1[g.pos+j]+p2[p2_pos+j])/2.0;
				for (int j=0; j<n; ++j) c2[g.pos+j] = (p1[g.pos+j]+p2[p2_pos+j])/2.0;
			}
		}
		/*
		for (size_t s=0; s<genome.size(); ++s) {
			const Gene& g = genome[s];
			if (std::find(averaged.begin(), averaged.end(), s)==averaged.end()) {
				int n = g.wsize + g.bsize;
				int p1_pos = similar_gene(genome, p2, g.pos, p1)
				for (int j=0; j<n; ++j) c2[g.pos+j] = (p2[g.pos+j]+p1[p1_pos+j])/2.0;
			}
		}
		*/
	}

	void mix(const Vals& p1, const Vals& p2, Vals& c1) {
		crossover::values::uniform<dtype> xo;
		xo.apply(p1, p2, c1);
	}

	void mix(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		crossover::values::uniform<dtype> xo;
		xo.apply(p1, p2, c1, c2);
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1) {
		double wheel = uniform_random_real<double>(0.0, 1.0);

		if (test==nnmerge_AvgFull) {
			average_all(p1, p2, c1);
		} else if (test==nnmerge_AvgSome) {
			average_some(p1, p2, c1);
		} else if (test==nnmerge_Uniform) {
			mix(p1, p2, c1);
		} else if (test==nnmerge_AvgFullAvgSome) {
			if (wheel <= 0.5) {
				average_all(p1, p2, c1);
			} else {
				average_some(p1, p2, c1);
			}
		} else if (test==nnmerge_AvgFullUniform) {
			if (wheel <= 0.5) {
				average_all(p1, p2, c1);
			} else {
				mix(p1, p2, c1);
			}
		} else if (test==nnmerge_AvgSomeUniform) {
			if (wheel <= 0.5) {
				average_some(p1, p2, c1);
			} else {
				mix(p1, p2, c1);
			}
		} else {
			if (wheel <= 0.33) {
				average_all(p1, p2, c1);
			} else if (wheel <= 0.66) {
				average_some(p1, p2, c1);
			} else {
				mix(p1, p2, c1);
			}
		}

		/*
		if (wheel <= 0.33) {
			// average all neurons
			average_all(p1, p2, c1);
		} else if (wheel <= 0.66) {
			// average some neurons, others choosen from p1 or p2
			average_some(p1, p2, c1);
		} else {
			// mix
			mix(p1, p2, c1);
		}
		*/
	}

	void apply(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		assert(false);
		double wheel = uniform_random_real<double>(0.0, 1.0);
		if (wheel <= 0.33) {
			// average all neurons
			average_all(p1, p2, c1);
			average_some(p1, p2, c2);
		} else if (wheel <= 0.66) { // 0.6
			average_some(p1, p2, c1, c2);
		} else {
			// mix
			mix(p1, p2, c1, c2);
		}
	}

	std::string info() const { return "nnmerge"; }
};


// mutation, train member for more epochs
template <typename Type>
struct nntrain {
	using Vals = std::vector<Type>;
	double prob;
	int epochs;
	umat<dtype> X_train, Y_train;
	nntrain(double prb, int epch, const umat<dtype>& X, const umat<dtype> Y): 
		prob(prb), epochs(epch), X_train(X), Y_train(Y) {}
	void apply(Vals& m) {
		if (uniform_random_real<double>(0.0, 1.0) < prob) {
			FFNN<dtype> net = create_network();
			chain_to_nn(m, net);
			train_nn(net, epochs, 0, X_train, Y_train);
			nn_to_chain(net, m);
		}
	}
	std::string info() const { return "nntrain"; }
};


// converts a chain to a solution
void solution(const vector<dtype>& chain, FFNN<dtype>& net)
{
	chain_to_nn(chain, net);
}

// fitness function
// determines how good a solution is
struct Fitness {
	umat<dtype> X, Y, Xv, Yv;
	double operator ()(const std::vector<dtype>& chain) {
		FFNN<dtype> net = create_network();
		solution(chain, net);
		double acc = evaluate_nn(net, Xv, Yv);
		//double acc = evaluate_nn(net, X, Y);
		return acc;
	}
};


void train_nn(FFNN<dtype>& nn, int iters, int info, const umat<dtype>& X, const umat<dtype>& Y)
{
	struct PSO_Fitness {
		umat<dtype> X, Y;
		double operator ()(const std::vector<dtype>& chain) {
			FFNN<dtype> net = create_network();
			solution(chain, net);
			double acc = evaluate_nn(net, X, Y);
			return 1.0 - acc;
		}
	};

	// member initialization
	struct PSO_nninit {
		using Vals = std::vector<dtype>;
		Vals src;
		PSO_nninit(const Vals& __src): src(__src) {}
		void apply(Vals& m) {
			initializer::values::gaussian<dtype> g(0.0, 2.0);
			m = src;
			g.apply(m);
		}
		std::string info() const { return "PSO_nninit"; }
	};


	constexpr int BATCH_SIZE = 250;
	PSO<dtype> pso;

	PSO<dtype>::params opt;
	opt.max_iters  = iters*X.ydim()/BATCH_SIZE;
	opt.threshold  = 1e-5;
	opt.info_iters = info*X.ydim()/BATCH_SIZE;
	pso.set_params(opt);

	std::vector<dtype> chain;
	nn_to_chain(nn, chain);

	PSO_nninit rndinit(chain);

	PSO_Fitness ff;
	umat<dtype> Xb, Yb;

	pso.init(GA_MUT_SWARM_SIZE, chain.size(), rndinit);
	for (;;) {
		int batch_size = BATCH_SIZE;
		int nbatches = X.ydim() / batch_size;
		int nleftover = X.ydim() % batch_size;
		bool done = false;
		for (int b=0; b<nbatches && !done; ++b) {
			int bs = batch_size;
			if (b==nbatches-1) bs += nleftover;
			Xb.resize(bs, X.xdim()); Xb.copy_rows(X, b*batch_size, bs);
			Yb.resize(bs, Y.xdim()); Yb.copy_rows(Y, b*batch_size, bs);
			ff.X = Xb;
			ff.Y = Yb;
			pso.evaluate(ff);
			pso.update();
			done = pso.done();
			if (done) break;
			pso.step();
		}
		if (done) break;
	}
	pso.finish();

	chain_to_nn(pso.solution(), nn);
}

double evaluate_nn(FFNN<dtype>& nn, const umat<dtype>& X, const umat<dtype>& Y)
{
	umat<dtype> Y_pred, Y_pred1hot;
	nn.predict(X, Y_pred);
	Y_pred1hot.resize_like(Y_pred);
	Y_pred1hot.argmaxto1hot(Y_pred);
	return accuracy(Y, Y_pred1hot);
}


int main()
{
	umat<dtype> X, X_valid, X_test;
	umat<dtype> Y, Y_valid, Y_test;
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

	// create the FFNN
	FFNN<dtype> net = create_network();
	if (!net) {
		cout << "Error creating FFNN network: " << net.error_description() << "\n";
		return -1;
	}
	log << net.info() << "\n";

	log << "Mutation probability: " << GA_MUT_PROB << "\n";

	Genome genome;
	nn_to_genome(net, genome);
	cout << "Genome: " << format_genome(genome) << "\n";

	if (1) {
		for (int t=0; t<1; ++t) {
			log << "\nSEED=" << SEED+t << "\n";
			umml_seed_rng(SEED+t);
			int epochs = 0;
			FFNN<dtype> nn = create_network();
			// MLP:60, CNN:20: 
			train_nn(nn, 15, 1, X, Y);
			double vacc = evaluate_nn(nn, X_valid, Y_valid);
			double tacc = evaluate_nn(nn, X_test, Y_test);
			log << "epochs: " << epochs << ", accuracy validation set=" << vacc << ", test set=" << tacc << "\n";
		}
	} else {
		log << "\n";
		log << "BP only (MLP), best validation set accuracy=" << 0.9771 << " (260 epochs), mean=" << "0.9750 (5 runs)\n";
		log << "BP only (CNN), best validation set accuracy=" << 0.9896 << "  (70 epochs), mean=" << "0.9885 (5 runs)\n";
	}


	/*
	// Debug
	std::vector<dtype> chain;
	nn_to_chain(net, chain);
	cout << "Initial:    " << format_chain(chain) << "\n";
	chain_to_nn(chain, net);
	nn_to_chain(net, chain);
	cout << "Transfered: " << format_chain(chain) << "\n";
	*/


	//
	// GA
	//

	// determine chain size
	int popsize = GA_POP_SIZE;
	int chainsize = 0;
	for (const Gene& g : genome) chainsize += (g.wsize + g.bsize);
	log << "Total values needed to encode all trainable parameters: " << chainsize << "\n\n";
	
	GA<dtype> ga;
	GA<dtype>::params opt;
	opt.info_iters = 1;
	opt.max_iters = GA_ITERS;
	opt.elitism = 0.02;
	opt.children = 1;
	opt.filename = string(SAVES_FOLDER)+GASAVE_NAME;
	opt.autosave = 1;
	//opt.parallel = GA<dtype>::SingleThread;
	ga.set_params(opt);
	ga.set_logging_stream(&log);

	Fitness ff;
	ff.X  = X;
	ff.Y  = Y;
	ff.Xv = X_valid;
	ff.Yv = Y_valid;



	DO_TEST(false)

	constexpr int N_SEEDS = 1;
	double sums_valid[nnmerge_All+1];
	double sums_test[nnmerge_All+1];
	for (int i=0; i<=nnmerge_All; ++i) sums_valid[i] = sums_test[i] = 0.0;

	for (int t=0; t<N_SEEDS; ++t) {
	//for (int t=N_SEEDS-1; t<N_SEEDS; ++t) {
		int seed = SEED + SEED*t;
		log << "\nSeed: " << seed << "\n";
		log << "----------\n";

/*
		nnmerge_AvgFull,
		nnmerge_AvgSome,
		nnmerge_Uniform,
		nnmerge_AvgFullAvgSome,
		nnmerge_AvgFullUniform,
		nnmerge_AvgSomeUniform,
		nnmerge_All,
*/
		for (int xo_mode=nnmerge_All; xo_mode<=nnmerge_All; ++xo_mode) {
			umml_seed_rng(seed);

			nninit rndinit;
			nnmerge xo(genome, xo_mode);
			nntrain<dtype> mut(GA_MUT_PROB, GA_MUT_TRAIN_ITERS, X, Y);
			log << "\nCrossover: " << test_mode_str(xo_mode) << ", PSO swarm size: " << GA_MUT_SWARM_SIZE << ", PSO iters: " << GA_MUT_TRAIN_ITERS << "\n";
			log << "evolving " << popsize << " members for " << opt.max_iters << " generations...\n";
			ga.solve(chainsize, popsize, rndinit, ff, xo, mut);
			log << "evolution finished, top-5 members:\n";
			for (int i=0; i<std::min(5,popsize); ++i) log << output_member(ga.get_member(i), i);
			if (popsize > 5) log << output_member(ga.get_member(popsize-1), popsize-1);

			FFNN<dtype> net = create_network();
			umat<dtype> Y_pred, Y_pred1hot;
			uvec<int> y_train_pred, y_pred;
			solution(ga.get_member(0).chain, net);
			net.predict(X, Y_pred);
			Y_pred1hot.resize_like(Y_pred);
			Y_pred1hot.argmaxto1hot(Y_pred);
			if (!enc.decode(Y_pred1hot, y_train_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
			double tr_acc = accuracy<>(y, y_train_pred);
			log << "Training set predicted with " << tr_acc << " accuracy.\n";
			net.predict(X_valid, Y_pred);
			Y_pred1hot.resize_like(Y_pred);
			Y_pred1hot.argmaxto1hot(Y_pred);
			if (!enc.decode(Y_pred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
			double val_acc = accuracy<>(y_valid, y_pred);
			log << "Validation set predicted with " << val_acc << " accuracy.\n";
			net.predict(X_test, Y_pred);
			Y_pred1hot.resize_like(Y_pred);
			Y_pred1hot.argmaxto1hot(Y_pred);
			if (!enc.decode(Y_pred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
			double acc = accuracy<>(y_test, y_pred);
			log << "Test set predicted with " << acc << " accuracy.\n";
			//log << "Test: " << y_test.format() << "\n";
			//log << "Vald: " << y_pred.format() << "\n";

			sums_valid[xo_mode] += ga.get_member(0).fitness;
			sums_test[xo_mode] += acc;
		}

		for (int i=0; i<=nnmerge_All; ++i) {
			log << "sums_valid[" << i << "] = " << sums_valid[i] << "\n";
			log << "sums_test[" << i << "] = " << sums_test[i] << "\n";
		}
	}

	log << "\nRESULTS:\n";
	for (int i=0; i<=nnmerge_All; ++i) {
		log << test_mode_str(i) << ":\n";
		log << " mean valid accuracy: " << sums_valid[i]/N_SEEDS << "\n";
		log << " mean test accuracy:  " << sums_test[i]/N_SEEDS << "\n";
	}

	END_TEST




	#if 0
	DO_TEST(false)

	nninit rndinit;
	nnmerge xo(genome, nnmerge_All);
	initializer::null<dtype> noinit;
	//mutation::null<dtype> mut;
	nntrain<dtype> mut(GA_MUT_PROB, GA_MUT_TRAIN_ITERS, X, Y);

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

	// training set and test set accuracy with the best member
	for (int i=0; i<2; ++i) {
		umat<dtype> Y_pred, Y_pred1hot;
		uvec<int> y_train_pred, y_pred;
		solution(ga.get_member(i).chain, net);
		net.predict(X, Y_pred);
		Y_pred1hot.resize_like(Y_pred);
		Y_pred1hot.argmaxto1hot(Y_pred);
		if (!enc.decode(Y_pred1hot, y_train_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
		double tr_acc = accuracy<>(y, y_train_pred);
		log << "\nTraining set predicted with " << tr_acc << " accuracy.\n";
		net.predict(X_test, Y_pred);
		Y_pred1hot.resize_like(Y_pred);
		Y_pred1hot.argmaxto1hot(Y_pred);
		if (!enc.decode(Y_pred1hot, y_pred, onehot_enc<>::SkipFaulty)) log << "error in decoding\n";
		confmat<> cm;
		const int cm_mode = CM_Macro;
		cm = confusion_matrix<>(y_test, y_pred, cm_mode);
		log << "Test set predictions:\n";
		log << "Accuracy  = " << accuracy<>(cm) << "\n";
		log << "Precision = " << precision<>(cm) << "\n";
		log << "Recall    = " << recall<>(cm) << "\n";
		log << "F1        = " << F1<>(cm) << "\n";
		log << "Confusion matrix:\n" << cm.m.format(0, 4) << "\n";
	}

	END_TEST
	#endif


	return 0;
}
