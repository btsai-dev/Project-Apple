
# Project Apple

Project Apple is a Data Generation and Variational Autoencoder created to support a paper in Information Theory.

Currently, only the data generation step is being worked on. A working Variational Autoencoder has not been implemented. A prototype default Variational Autoencoder can be found in the src/Legacy Code folder (note that it is the standard Variational Autoencoder to generate handwritten images from the MNIST database).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

#### Tensorflow
Used for machine learning.
```console
python -m pip install --user tensorflow
```
#### Matplotlib
Used for data visualization.
```console
python -m pip install --user matplotlib
```
Further installation details can be found [here](https://matplotlib.org/users/installing.html).

#### Numpy
Used for scientific computing.
```console
python -m pip install --user numpy
```

#### PyPer
Used for R integration.
```
python -m pip install --user pyper
```

Further installation details can be found [here](https://buildmedia.readthedocs.org/media/pdf/pyper/latest/pyper.pdf).

### Installing and Running

- Clone the github repository
```console
git clone https://github.com/godonan/Project-Apple.git
```
- Navigate to the repository's ```src``` folder and execute the data generation program.
```console
python3 Training_Generation.py
```
- Execute the VAE program (TODO, FILE DOES NOT EXIST YET).
```console
python3 VAE_Execution.py
```
The program can also be run in an IDLE environment.


## Additional Requirements

* [Python](https://www.python.org/doc/) - Programming Language
* [Tensorflow 2](https://www.tensorflow.org/) - AI Framework
* [R](https://www.r-project.org/) - Statistical Computing Language

## Papers
* **Algebraic Properties of Wyner Common Information Solution under Graphical Constraints** - *Related Paper* - [https://arxiv.org/abs/2001.02712](https://arxiv.org/abs/2001.02712)
* **Latent Factor Analysis of Gaussian Distributions under Graphical Constraints** - *Related Paper* - [https://arxiv.org/abs/2001.02712](https://arxiv.org/abs/2001.02712)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

