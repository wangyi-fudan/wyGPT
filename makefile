all:	train gpu cpu
train:	train.cu makefile config
	/usr/local/cuda-12.2/bin/nvcc train.cu -o train -arch=sm_86 -lcublas -Xptxas -O3 --std=c++11
gpu:	gpu.cu makefile config
	/usr/local/cuda-12.2/bin/nvcc gpu.cu -o gpu -arch=sm_86 -lcublas -Xptxas -O3 --std=c++11
cpu:	cpu.cpp makefile config
	g++ cpu.cpp -o cpu -Ofast -march=native -static -s -fopenmp
