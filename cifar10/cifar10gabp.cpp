// Train a FFNN with a GA+backprop
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
#include "nn/ffnn.hpp"
#include "nn/backprop.hpp"
#include "datasets/cifarloader.hpp"
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
struct Genome {
	int nlayers;
	std::vector<Gene> genes;
};


#define LOG_NAME      "c10-bpga-adam-convdo.log"
#define SAVES_FOLDER  "../../saves/cifar10/"
#define GASAVE_NAME   "c10-bpga-adam-do.ga"

// Seed for local RNG
constexpr int  SEED = 48;

// GA parameters
constexpr int    GA_POP_SIZE        = 20;
constexpr int    GA_ITERS           = 75;
constexpr int    GA_BATCH_SIZE      = 30;
constexpr int    GA_MUT_TRAIN_ITERS = 5;
constexpr double GA_MUT_LRATE       = 0.0005;
constexpr double GA_MUT_PROB        = 0.2;


// activation functions
int fconv = fReLU;
int ffc = fReLU;

FFNN<dtype> create_cnn()
{
	FFNN<dtype> net("c10-bpga-cnn");
	net.add(new InputLayer<dtype>(dims3{32,32,3}));
	net.add(new Conv2DLayer<dtype>(32, 5,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DropoutLayer<dtype>(0.25));
	net.add(new Conv2DLayer<dtype>(64, 3,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DropoutLayer<dtype>(0.25));
	net.add(new Conv2DLayer<dtype>(128, 3,1,Valid, fconv));
	net.add(new MaxPool2DLayer<dtype>(2,2));
	net.add(new DenseLayer<dtype>(128, ffc));
	//net.add(new DropoutLayer<dtype>(0.5));
	net.add(new SoftmaxLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_ffnn()
{
	FFNN<dtype> net("c10-bpga-mlp");
	net.add(new InputLayer<dtype>(dims3{32,32,3}));
	net.add(new DenseLayer<dtype>(256, ffc));
	net.add(new DenseLayer<dtype>(128, ffc));
	//net.add(new DenseLayer<dtype>(32, ffc));
	net.add(new SoftmaxLayer<dtype>(10));
	return net;
}

FFNN<dtype> create_network()
{
	return create_cnn();
	//return create_ffnn();
}

void train_nn(FFNN<dtype>& nn, int iters, int info, const umat<dtype>& X, const umat<dtype>& Y)
{
	Backprop<dtype> bp;
	{
	typename Backprop<dtype>::params opt;
	opt.batch_size = GA_BATCH_SIZE;
	opt.max_iters = iters;
	opt.info_iters = info;
	opt.verbose = (info > 0);
	bp.set_params(opt);
	}
	lossfunc::softmaxce<dtype> loss;
	gdstep::adam<dtype> st(GA_MUT_LRATE);
	//gdstep::learnrate<dtype> st(GA_MUT_LRATE);
	shuffle::deterministic sh;
	//shuffle::stochastic sh;
	
	bp.train(nn, loss, st, sh, X, Y);
}

double evaluate_nn(FFNN<dtype>& nn, const umat<dtype>& X, const umat<dtype>& Y)
{
	umat<dtype> Y_pred, Y_pred1hot;
	nn.predict(X, Y_pred);
	Y_pred1hot.resize_like(Y_pred);
	Y_pred1hot.argmaxto1hot(Y_pred);
	return accuracy(Y, Y_pred1hot);
}

void nn_to_genome(const FFNN<dtype>& net, Genome& genome)
{
	int pos = 0;
	genome.nlayers = net.nlayers(); 
	genome.genes.clear();
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
				genome.genes.push_back(g);
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
	for (const Gene& g : genome.genes)
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
	for (int i=0; i<(int)genome.genes.size(); ++i) {
		if (genome.genes[i].pos==pos) return i;
	}
	return -1;
}

int find_first_gene_with_layer(const Genome& genome, int layer)
{
	for (int i=0; i<(int)genome.genes.size(); ++i) {
		if (genome.genes[i].layer==layer) return i;
	}
	return -1;
}

int find_last_gene_with_layer(const Genome& genome, int first, int layer)
{
	if (first >= 0) {
		int i = first+1;
		while (i < (int)genome.genes.size() && genome.genes[i].layer==layer) i++;
		return i-1;
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


// cosine similaritybetween two genes
dtype similarity_metric(const std::vector<dtype>& chain1, int pos1, 
						const std::vector<dtype>& chain2, int pos2, int n)
{
	uv_ref<dtype> v1(chain1, pos1, n);
	uv_ref<dtype> v2(chain2, pos2, n);
	return v1.dot(v2) / (v1.magnitude()*v2.magnitude());
}

int similar_gene(const Genome& genome, int gpos1, const std::vector<dtype>& chain1, const std::vector<dtype>& chain2)
{
	const Gene& g1 = genome.genes[gpos1];
	int first = find_first_gene_with_layer(genome, g1.layer);
	int last  = find_last_gene_with_layer(genome, first, g1.layer);
	assert(last != -1);
	int n = g1.wsize+g1.bsize;
	int gpos2 = 0;
	dtype best = std::numeric_limits<dtype>::min();
	for (int i=first; i<=last; ++i) {
		const Gene& g2 = genome.genes[i];
		dtype sim = similarity_metric(chain1, g1.pos, chain2, g2.pos, n);
		if (sim > best) {
			best = sim;
			gpos2 = i;
		}
	}
	return gpos2;
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
	int mode;

	nnmerge(const Genome& __genome, int __mode): genome(__genome), mode(__mode) {}


	void average_all(const Vals& p1, const Vals& p2, Vals& c1) {
		int n = (int)p1.size();
		for (int i=0; i<n; ++i) c1[i] = (p1[i]+p2[i])/2.0;
	}

	void average_some(const Vals& p1, const Vals& p2, Vals& c1) {
		c1 = p1;
		double perc = uniform_random_real<double>(0.2, 0.8);
		for (int i=0; i<(int)genome.genes.size(); ++i) {
			double coin = uniform_random_real<double>(0.0, 1.0);
			if (coin <= perc) {
				const Gene& g1 = genome.genes[i];
				const Gene& g2 = genome.genes[similar_gene(genome, i, p1, p2)];
				int n = g1.wsize + g1.bsize;
				for (int j=0; j<n; ++j) c1[g1.pos+j] = (p1[g1.pos+j]+p2[g2.pos+j])/2.0;
			}
		}
	}

	void average_some(const Vals& p1, const Vals& p2, Vals& c1, Vals& c2) {
		c1 = p1;
		c2 = p2;
		/*
		std::vector<int> averaged;
		for (size_t s=0; s<genome.genes.size(); ++s) {
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
		*/
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

		if (mode==nnmerge_AvgFull) {
			average_all(p1, p2, c1);
		} else if (mode==nnmerge_AvgSome) {
			average_some(p1, p2, c1);
		} else if (mode==nnmerge_Uniform) {
			mix(p1, p2, c1);
		} else if (mode==nnmerge_AvgFullAvgSome) {
			if (wheel <= 0.5) {
				average_all(p1, p2, c1);
			} else {
				average_some(p1, p2, c1);
			}
		} else if (mode==nnmerge_AvgFullUniform) {
			if (wheel <= 0.5) {
				average_all(p1, p2, c1);
			} else {
				mix(p1, p2, c1);
			}
		} else if (mode==nnmerge_AvgSomeUniform) {
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

	// split training/validation sets
	X_valid = df.copy_rows(X, 40000, 10000);
	y_valid = df.copy(y, 40000, 10000);
	X.reshape(40000, X.xdim());
	y.reshape(40000);

	// scale data
	X.mul(1.0/255);
	X.plus(-0.5);
	X_valid.mul(1.0/255);
	X_valid.plus(-0.5);
	X_test.mul(1.0/255);
	X_test.plus(-0.5);

	// encode labels with one-hot encoding
	onehot_enc<> enc;
	enc.fit_encode(y, Y);
	enc.encode(y_valid, Y_valid);
	enc.encode(y_test, Y_test);

	// create the FFNN
	FFNN<dtype> net = create_network();
	if (!net) {
		log << "Error creating FFNN network: " << net.error_description() << "\n";
		return -1;
	}
	log << net.info() << "\n";

	log << "Mutation probability: " << GA_MUT_PROB << "\n";
	log << "Learning rate: " << GA_MUT_LRATE << "\n";

	Genome genome;
	nn_to_genome(net, genome);
	//cout << "Genome: " << format_genome(genome) << "\n";

	if (1) {
		for (int t=0; t<1; ++t) {
			int seed = SEED+SEED*t;
			log << "\nSEED=" << seed << "\n";
			umml_seed_rng(seed);
			int epochs = 0;
			FFNN<dtype> nn = create_network();
			// MLP:60, CNN:20: 
			for (int i=0; i<GA_ITERS; ++i) {
				train_nn(nn, 5, 0, X, Y);
				double  acc = evaluate_nn(nn, X, Y);
				double vacc = evaluate_nn(nn, X_valid, Y_valid);
				double tacc = evaluate_nn(nn, X_test, Y_test);
				epochs += 5;
				log << "epochs: " << epochs << ", accuracy (training, validation, test): " << acc << ", " << vacc << ", " << tacc << "\n";
			}
		}
	} else {
		log << "\n";
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
	for (const Gene& g : genome.genes) chainsize += (g.wsize + g.bsize);
	log << "Total values needed to encode all trainable parameters: " << chainsize << "\n";
	
	GA<dtype> ga;
	GA<dtype>::params opt;
	opt.info_iters = 1;
	opt.max_iters = GA_ITERS;
	opt.elitism = 0.02;
	opt.children = 1;
	opt.filename = string(SAVES_FOLDER)+GASAVE_NAME;
	opt.autosave = 1;
	opt.parallel = GA<dtype>::SingleThread;
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

	for (int t=N_SEEDS-1; t<N_SEEDS; ++t) {
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
		//for (int xo_mode=nnmerge_AvgFull; xo_mode<=nnmerge_All; ++xo_mode) {
		const std::vector<int> xo_list = { nnmerge_All };
		for (int xo_mode : xo_list) {
			umml_seed_rng(seed);

			nninit rndinit;
			nnmerge xo(genome, xo_mode);
			nntrain<dtype> mut(GA_MUT_PROB, GA_MUT_TRAIN_ITERS, X, Y);
			log << "\nCrossover: " << test_mode_str(xo_mode) << ", learning rate: " << GA_MUT_LRATE << ", mutation epochs: " << GA_MUT_TRAIN_ITERS << "\n";
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


	return 0;
}
