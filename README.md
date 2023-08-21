# FE<sup>2</sup> Computations With Deep Neural Networks: Algorithmic Structure, Data Generation, and Implementation

This repository features codes and data employed for developing deep neural networks (DNN) based surrogate models for FE<sup>2</sup> computations. To gain a significant speed-up of the FE<sup>2</sup> computations, an efficient implementation of the trained neural network in a FORTRAN finite element code is provided using [Forpy](https://github.com/ylikx/forpy). This is achieved by drawing on state-of-the-art high-performance computing libraries, e.g. [JAX](https://github.com/google/jax), and just-in-time (JIT) compilation yielding a maximum speed-up of a factor of more than 5,000 compared to a reference FE<sup>2</sup> computation. More details about the implementation and the results are available in ["FE<sup>2</sup> Computations with Deep Neural Networks: Algorithmic Structure, Data Generation, and Implementation (2023)" H Eivazi, JA Tröger, S Wittek, S Hartmann, A Rausch](https://doi.org/10.3390/mca28040091).

![](Graphical_abstract.png)

## Citation

```
@article{dnn_fe2,
AUTHOR = {Eivazi, Hamidreza and Tröger, Jendrik-Alexander and Wittek, Stefan and Hartmann, Stefan and Rausch, Andreas},
TITLE = {{FE}² Computations with Deep Neural Networks: Algorithmic Structure, Data Generation, and Implementation},
JOURNAL = {Mathematical and Computational Applications},
VOLUME = {28},
YEAR = {2023},
NUMBER = {4},
ARTICLE-NUMBER = {91},
DOI = {10.3390/mca28040091}
}
```
