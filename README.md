CIFAR-10 and MNIST datasets can be downloaded from:
- MNIST: http://yann.lecun.com/exdb/mnist/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

Files must be unzipped and placed in 'data/MNIST' and 'data/CIFAR10' folders

cifar10:
   make             : help
   make cifar10gabp : CIFAR-10, train with a hybrid BP+GA.
   make cifar10kdga : CIFAR-10, pruning with a GA and knowledge distillation (KD).
   
mnist:
   make             : help
	make mnistga     : MNIST (digits), outlier detection with GA.
	make mnistpso    : MNIST (digits), outlier detection with PSO.
	make mnistbpga   : MNIST (digits), BP+GA hybrid training.
	make mnistkdga   : MNIST (digits), pruning with a GA and knowledge distillation.
