CIFAR-10 and MNIST datasets can be downloaded from:
- MNIST: http://yann.lecun.com/exdb/mnist/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

Files must be unzipped and placed in 'data/MNIST' and 'data/CIFAR10' folders

cifar10:<br />
   make             : help<br />
   make cifar10gabp : CIFAR-10, train with a hybrid BP+GA.<br />
   make cifar10kdga : CIFAR-10, pruning with a GA and knowledge distillation (KD).<br />
   <br />
mnist:<br />
   make             : help<br />
   make mnistga     : MNIST (digits), outlier detection with GA.<br />
   make mnistpso    : MNIST (digits), outlier detection with PSO.<br />
   make mnistbpga   : MNIST (digits), BP+GA hybrid training.<br />
   make mnistkdga   : MNIST (digits), pruning with a GA and knowledge distillation.<br />
