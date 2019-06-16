# Play with Neural Network using C++ #

## MNIST CSV dataset ##

[https://pjreddie.com/projects/mnist-in-csv/](https://pjreddie.com/projects/mnist-in-csv/)

## Eigen ##

Using Eigen to deal with Matrix:

[http://eigen.tuxfamily.org/](http://eigen.tuxfamily.org/)

You can use Intel MKL to optimize the performance:

[https://software.intel.com/en-us/mkl](https://software.intel.com/en-us/mkl)

## Make ##

```
make
```

With Intel MKL:

```
make mkl
```

## Usage ##

```
usage:
    ?, h, help          show this help
    q, quit, exit       exit program
    <num>               view data at index <num>
    p[:]<num>           predict data at index <num>
    auc[:]<count>       evaluate accuracy use <num> records
    train[:]<loop>      train <loop>s use loaded dataset
    save[:]<file>       save model to <file>
    load[:]<file>       load model from <file>
```
