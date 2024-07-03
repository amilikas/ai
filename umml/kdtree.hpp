#ifndef UMML_KDTREE_INCLUDED
#define UMML_KDTREE_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 k-dimensional tree (k-d tree)

 FILE:     kdtree.hpp
 AUTHOR:   Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:     2021-2024
 KEYWORDS: K-D Tree 
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 umml vector
 umml matrix
 STL string
 
 Internal dependencies:
 umml algo (pqueue)
 STL vector
 STL algorithm
 STL queue

 Notes
 ~~~~~
 * Uses std::nth_element()
 Τhe actual data are stored in the matrix outside the kdtree class and they are
 accessed through their row's index.
 
 * Nodes are stored in an internal buffer like this:
   0      1      2      3      4      5
  row   left   right   row   left   right  ...
 index  child  child  index  child  child
 
 * For large dimensions (20 is already large) do not expect this to run significantly faster than 
 brute force. High-dimensional nearest-neighbor queries are a substantial open problem in computer 
 science.
 
 References
 ~~~~~~~~~~
 [1] Steven Schmatz: How does a k-d tree find the K nearest neighbors
 https://www.quora.com/How-does-a-k-d-tree-find-the-K-nearest-neighbors
 
 Usage example
 ~~~~~~~~~~~~~ 
 kdtree<> kd(X);
 kd.build();
 uvec<int> indeces;
 uvec<float> dists;
 kd.k_nearest(3, point, indeces, dists);
*/

#include "umat.hpp"
#include "algo.hpp"
#include <queue>
#include <iostream>


namespace umml {


template <typename Type=float>
class kdtree
{
 public:
	typedef int node_t;
	typedef int idx_t;

	// null pointer definition
	enum { null = -1 };

	// modes for converting a tree to a string
	enum { single_line, multiple_rows };

	/// constructors, destructor
	kdtree() { 
		pM = nullptr;
		heap = nullptr;
		capacity = 0;
		n_items = 0;
		root = null;
	}
	
	kdtree(const umat<Type>& M) { 
		pM = &M;
		heap = nullptr;
		capacity = 0;
		n_items = 0;
		root = null;
	}
	
	~kdtree() { 
		if (heap) delete[] heap; 
	}
	
	void use(const umat<Type>& M) { pM = &M; }

	// access nodes
	idx_t&  index(node_t node) { return heap[node]; }
	node_t& left(node_t node)  { return heap[node+1]; }
	node_t& right(node_t node) { return heap[node+2]; }

	// build the k-d tree
	void build();

	// finds the node in the tree where the point 'pt' is stored.
	node_t find(idx_t pt);

	// finds the nearest neighbor of the point 'pt'
	idx_t nearest(const uvec<Type>& pt) { 
		uvec<idx_t> knn;
		uvec<Type> dist;
		k_nearest(1, pt, knn, dist);
		return knn(0);
	}

	// finds the k-nearest neighbors and their distances from the point 'pt'
	template <class Vector>
	void k_nearest(int k, const Vector& pt, uvec<idx_t>& knn, uvec<Type>& dists);

	// finds the k-nearest neighbors and their distances from the point indexed by 'target'
	void k_nearest(int k, idx_t target, uvec<idx_t>& knn, uvec<Type>& dists) {
		const uv_ref<Type> pt = (*pM).row(target);
		return k_nearest(k, pt, knn, dists);
	}

	// convert the tree to a printable string.
	// * single_line mode:
	// * multiple_rows mode:
	std::string dump(int mode=multiple_rows);

 private:
	// private methods
	node_t  append(idx_t it);
	void    build_branch(int col, int start, int end);
	void    tree_to_string(node_t node, std::string& str);
	template <class Vector>
	Type    distance(idx_t it, const Vector& pt);
	template <class Vector>
	void    neighbors(node_t node, const Vector& pt, int col, pqueue<std::pair<idx_t,Type>>& q);

 private:
	Type    tolerance;
	node_t* heap;
	int     capacity;
	int     n_items;
	node_t  root;
	const umat<Type>* pM;
	std::vector<idx_t> rows_pool;
	size_t nodes;
};


////////////////////////////////////////////////////////////
//             I M P L E M E N T A T I O N                //
////////////////////////////////////////////////////////////

template <typename IdxT, typename Type>
int median(const umat<Type>& M, const std::vector<IdxT>& rows_pool, int col, int start, int end) {
	typedef std::pair<Type,IdxT> ElemIdx;
	typedef std::vector<ElemIdx> ElemIdxVec;
	int n = end-start+1;
	ElemIdxVec colv(n);
	for (int i=0; i<n; ++i) colv[i] = std::make_pair(M(rows_pool[start+i],col) , (IdxT)(start+i));
    typename ElemIdxVec::iterator m = colv.begin() + n/2;
    std::nth_element(colv.begin(), m, colv.end(), [](ElemIdx& a, ElemIdx& b){ return a.first < b.first; });
	while (m != colv.begin() && m->first==(m-1)->first) m--;
    return m->second;
}

template <typename Type>
void kdtree<Type>::build_branch(int col, int start, int end) 
{
	if (start > end) return;
	int m = median<idx_t,Type>(*pM, rows_pool, col % (*pM).xdim(), start, end);
	append(rows_pool[m]);
	build_branch(col+1, start, m-1);
	build_branch(col+1, m+1, end);
}

template <typename Type>
void kdtree<Type>::build()
{
	const umat<Type>& M = *pM;
	int n = M.ydim();
	// make the rows_pool vector
	rows_pool.resize(n);
	for (idx_t i=0; i<n; ++i) rows_pool[i] = i;
	// allocate internal buffer
	if (capacity != n) {
		if (heap) delete[] heap;
		capacity = n;
		heap = new node_t [capacity*3];
	}
	// build the tree
	n_items = 0;
	root = -1;
	build_branch(0, 0, n-1);
}

template <typename Type>
typename kdtree<Type>::node_t kdtree<Type>::append(idx_t it)
{
	if (n_items >= capacity) return null;

	const umat<Type>& M = *pM;
	int n = M.xdim();

	node_t new_node = n_items * 3;
	index(new_node) = it;
	left(new_node)  = null;
	right(new_node) = null;

	// if this is the first node, update root
	if (n_items==0) {
		root = new_node;
		n_items = 1;
		return new_node;
	}

	// link the new node with the correct leaf 
	int col = 0;
	node_t node = root;
	while (node != null) {
		node_t next_node;
		if (M(it, col%n) < M(index(node), col%n)) {
			next_node = left(node);
			if (next_node==null) left(node) = new_node;
		} else {
			next_node = right(node);
			if (next_node==null) right(node) = new_node;
		}
		col++;
		node = next_node;
	}
	n_items++;
	return new_node;
}

template <typename Type>
typename kdtree<Type>::node_t kdtree<Type>::find(idx_t target) 
{ 
	int col = 0;
	node_t node = root;
	while (node != null) {
		if (index(node)==target) break;
		if ((*pM)(target, col) < (*pM)(index(node), col)) node = left(node);
		else node = right(node);
		col++;
		if (col > pM->ncols()) col = 0;
	}
	return node;
}

template <typename Type>
void kdtree<Type>::tree_to_string(node_t node, std::string& str)
{
	if (node==null) return;
	str += std::to_string(index(node));
	if (left(node)==null && right(node)==null) return;

	// left subtree
	str.push_back('(');
	tree_to_string(left(node), str);
	str.push_back(')');
	// only if right child is present (to avoid extra parenthesis)
	if (right(node)) {
		str.push_back('(');
		tree_to_string(right(node), str);
		str.push_back(')');
	}
}

template <typename Type>
std::string kdtree<Type>::dump(int mode)
{
	if (mode==single_line) {
		std::string str;
		tree_to_string(root, str);
		return str;
	}
	std::stringstream ss;
	std::queue<node_t> q;
	q.push(root);
	while (!q.empty()) {
		int size = (int)q.size();
		for (int i = 0; i < size; ++i) {
			node_t n = q.front();
			q.pop();				
			if (n != null) {
				ss << index(n) << ", ";
				q.push(left(n));
				q.push(right(n));
			} else {
				ss << "#, ";
			}
		}
		ss << "\n";
	}
	return ss.str();
}

// this is distance squared
template <typename Type>
template <class Vector>
Type kdtree<Type>::distance(idx_t it, const Vector& pt)
{
	Type sum = static_cast<Type>(0);
	for (int j=0; j<pM->xdim(); ++j) {
		Type diff = (*pM)(it,j) - pt(j);
		sum += diff*diff;
	}
	return sum;
}

// finds the nearest node of the point 'pt'
template <typename Type>
template <class Vector>
void kdtree<Type>::neighbors(node_t node, const Vector& pt, int col, pqueue<std::pair<idx_t,Type>>& q)
{
	if (node==null) return;
	nodes++;
	Type dist = distance(index(node), pt);
	q.enqueue(dist, std::make_pair(index(node),(double)dist));

	const umat<Type>& M = *pM;
	int n = M.xdim();

	// Recursively search the half of the tree that contains 'pt'
	node_t other;
	if (pt(col%n) < M(index(node), col%n)) {
		neighbors(left(node), pt, col+1, q);
		other = right(node);
	} else {
		neighbors(right(node), pt, col+1, q);
		other = left(node);
	}

	if (q.len() < q.size() || std::abs(pt(col%n) - M(index(node), col%n)) < q.worst()) {
		// Recursively search the other half of the tree if necessary
		neighbors(other, pt, col+1, q);
	}
}

// finds the k-nearest nodes of the point 'pt'
template <typename Type>
template <class Vector>
void kdtree<Type>::k_nearest(int k, const Vector& pt, uvec<idx_t>& knn, uvec<Type>& dists)
{
	pqueue<std::pair<idx_t,Type>> q(k);
	nodes=0;
	neighbors(root, pt, 0, q);
//std::cout << "nodes=" << nodes << "\n";
	knn.resize(k);
	dists.resize(k);
	int i = 0;
	while (q.len()) {
		std::pair<idx_t,Type> nn = q.dequeue();
		knn(i) = nn.first;
		dists(i) = std::sqrt(nn.second);
		++i;
	}
}


};     // namespace umml

#endif // UMML_KDTREE_INCLUDED
