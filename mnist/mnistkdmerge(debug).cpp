#include "cpuinfo.hpp"
#include "dataframe.hpp"
#include "preproc.hpp"
#include "metrics.hpp"
#include "kmeans.hpp"
#include "nn/backprop.hpp"
#include "bio/ga.hpp"


using namespace std;
using namespace std::chrono;
using namespace umml;

// Datatype for data and neural network
typedef float dtype;

// Seed for local RNG
constexpr int SEED = 48;

// Batch size
constexpr int BATCH_SIZE = 10;

// GA parameters
constexpr int GA_POP_SIZE = 1000;
constexpr int GA_ITERS = 150;


const std::vector<int> layers = { 1, 2 };
const std::vector<int> Nt = { 4, 8, 2 }; 
const std::vector<int> Ns = { 2, 4, 2 }; 
typedef std::pair<uvec<dtype>, uvec<dtype>> weights_and_biases;

FFNN<dtype> create_teacher(const std::string& name, bool softmax)
{
	dims4 d;
	uvec<dtype> ws, bs;
	FFNN<dtype> net(name);
	Layer<dtype>* l;

	net.add(new InputLayer<dtype>(dims3{3,3,1}));

	net.add(l = new Conv2DLayer<dtype>(Nt[0],2,1,Valid,fLinear)); 
		l->get_trainable_parameters(ws, bs);
		d = l->_w.dims();
		for (int y=0; y<d.y; ++y) for (int x=0; x<d.x; ++x) ws(y*d.x+x) = 2+y;
		l->set_trainable_parameters(ws, bs);
		cout << l->_w.shape() << "\n" << ws.format() << "\n";

	//net.add(new MaxPool2DLayer<dtype>(4,4));

	net.add(l = new DenseLayer<dtype>(Nt[1], fLinear));
		l->get_trainable_parameters(ws, bs);
		d = l->_w.dims();
		for (int y=0; y<d.y; ++y) for (int x=0; x<d.x; ++x) ws(y*d.x+x) = 2+y;
		l->set_trainable_parameters(ws, bs);
		cout << l->_w.shape() << "\n" << ws.format() << "\n";

	if (softmax) l = new SoftmaxLayer<dtype>(2);
	else l = new DenseLayer<dtype>(2);
	net.add(l);
		l->get_trainable_parameters(ws, bs);
		d = l->_w.dims();
		for (int y=0; y<d.y; ++y) for (int x=0; x<d.x; ++x) ws(y*d.x+x) = 2+y;
		l->set_trainable_parameters(ws, bs);
		cout << l->_w.shape() << "\n" << ws.format() << "\n";

	return net;
}

FFNN<dtype> create_student(const std::string& name, bool softmax)
{
	dims4 d;
	uvec<dtype> ws, bs;
	FFNN<dtype> net(name);
	Layer<dtype>* l;

	net.add(new InputLayer<dtype>(dims3{3,3,1}));

	net.add(l = new Conv2DLayer<dtype>(Ns[0],2,1,Valid,fLinear));
		l->get_trainable_parameters(ws, bs);
		d = l->_w.dims();
		for (int y=0; y<d.y; ++y) for (int x=0; x<d.x; ++x) ws(y*d.x+x) = 2+y;
		l->set_trainable_parameters(ws, bs);
		cout << l->_w.shape() << "\n" << ws.format() << "\n";

	//net.add(new MaxPool2DLayer<dtype>(4,4));

	net.add(l = new DenseLayer<dtype>(Ns[1], fLinear));
		l->get_trainable_parameters(ws, bs);
		d = l->_w.dims();
		for (int y=0; y<d.y; ++y) for (int x=0; x<d.x; ++x) ws(y*d.x+x) = 2+y;
		l->set_trainable_parameters(ws, bs);
		cout << l->_w.shape() << "\n" << ws.format() << "\n";

	if (softmax) l = new SoftmaxLayer<dtype>(2);
	else l = new DenseLayer<dtype>(2);
	net.add(l);
		l->get_trainable_parameters(ws, bs);
		d = l->_w.dims();
		for (int y=0; y<d.y; ++y) for (int x=0; x<d.x; ++x) ws(y*d.x+x) = 2+y;
		l->set_trainable_parameters(ws, bs);
		cout << l->_w.shape() << "\n" << ws.format() << "\n";

	return net;
}

std::string analyze(const std::vector<weights_and_biases>& wbs) 
{
	std::stringstream ss;
	int i=1;
	for (auto wb: wbs) {
		uvec<dtype>& w = wb.first;
		uvec<dtype>& b = wb.second;
		if (!w.empty()) {
			ss << "w" << i << ":" << w.len() << " Σ^2=" << w.sum_squared();
			if (!b.empty()) ss << " b" << i << ":" << b.len() << "/" << b.count(0, 1e-3);
			ss << "\n";
		}
		i++;
	}
	return ss.str();
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
// twbs and initial stored weights from layer=1 and above
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
		cout << "student Ns=" << Ns[i] << "(" << sd.y << ")\n";
		cout << "teacher Nt=" << Nt[i] << "(" << td.y << ")\n";
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

            //int swidx = v[0]*Ns[i];
            //int twidx = v[1]*Nt[i];

			int twidx = v[1]*td.x;
			for (int k=0; k<sd.x; ++k) {
				dtype avg = (sw(p+k) + tw(twidx+k)) / 2.0;
				sw(p+k) = avg;
			}
			p += sd.x;
		}

		cout << "sw: " << sw.format(1) << "\n";
		swbs.push_back(std::make_pair(sw, sb));
	}

	// set student's network weights from wbs
	assert(swbs.size()==all.size());
	for (size_t i=0; i<swbs.size(); ++i) {
		int l = all[i];
		student.get_layer(l)->set_trainable_parameters(swbs[i].first, swbs[i].second);
	}
}




// -------------- Main -------------------

int main() 
{
	umat<dtype> X, X_teacher, X_student, X_valid, X_test;
	umat<dtype> Y, Y_teacher, Y_student, Y_valid, Y_test;
	uvec<int> y, y_train, y_valid, y_test;
	steady_clock::time_point t1, t2;

	// set openmp threads to maximum
	umml_set_openmp_threads();

	// seed RNG
	umml_seed_rng(SEED);


	FFNN<dtype> teacher = create_teacher("TRAINED_TEACHER", false);
	cout << teacher.info() << "\n\n";

	FFNN<dtype> student = create_student("STUDENT", false);
	cout << student.info() << "\n\n";
	//                    0,1  0,1,2,3
	vector<int> chain = { 1,3, 2,3,4,5 };
	solution(chain, teacher, student);
	

	return 0;
}
