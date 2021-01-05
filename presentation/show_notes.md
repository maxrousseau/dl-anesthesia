Theory and Concepts
=================

Deep Learning
-------------


**Statistics -> Machine Learning -> Deep learning**


As you all know, statistics allow scientist to draw inference from data.
Machine learning is a branch of statistics which aims at making predictions
from data. This is a gross oversimplification of both fields but I believe that
this distinction is important as it will affect how one would design a study.


There two main types of machine learning: supervised and unsupervised. In
supervised learning, the data has been labeled and we are trying to teach the
models to recognize and predict the labels. In unsupervised learning,
algorithms are trained without any label and try to organize the data without
any external knowledge.
I will not talk about unsupervised learning as it has not shown to be effective
in real world applications yet although this may change in the future.

Supervised learning and deep learning will be the central topic of discussion
of this talk as they have the most practical applications currently.

	SIDE NOTE: there is a third category which is reinforcement learning,
	which has had some success in games such as chess, Go and video games (atari,
	DoTa and Starcraft).


High level view of deep learning
The essential idea behind deep learning is that the a complex neural network
with many layers can gain higher level abstractions about complex or high
dimension data. For example, a network trained on images would learn to
recognize edges and shapes in its firsts layers then in deeper layers put those
primitives concepts together and detect ears eyes and other complex features to
determine if the image contains a cat or a dog.


Why does deep learning work
There are two main factors which have set the stage for deep learning's rise:
the increase in data availability and increase in computation power (the most
important factor most likely being the latter).


Deep Learning Building Blocks
-----------------------------

Deep learning can be boiled down to a series of tensor operation. The tensor is
the neural network input. Computers only understand numbers hence, data must be
transformed into tensors prior in order to be trained on a deep learning
moddel.

### Tensors

Tensor are simply arrays of digits they can have a variety of forms:
- 1D tensor: list of metrics from a patient
- 2D tensor: facial image from a patient
- 3D tensor: video data or 3D model

The type of tensor will guide which type of neural network you can create and
use. In practice the tensor represents the whole dataset so a sample of
grayscale images will actually be represented as a 3D tensor.


### Tensor Operation

The neural network will perform various operation on the input tensor.
Depending on what you are trying to achive the output may vary. In binary
classification you will end up with an output of either 1 or 0 for a given
input.

- addition/substraction
- multiplication
- dot product


### The Neuron

The _neuron_ is the core building block of a neural networks. Deep neural
networks will chain multiple neurons to transform and learn high level
abstractions from complex data.

$Y = ReLU(w\cdotX+b)$

- ReLU: an activation function
- w: weight
- b: bias
- X: input (tensor)
- Y: output (label)


Backpropagation
---------------

**How does the network learn?**

The main mechanism behind deep learning is called backpropagation. In
supervised deep learning, it is how the network can improve its performance
during the training loop. The backpropagation algorithm has a few components:

- loss function
- gradient descent
- optimizer


### Terminology regarding training - batch and epoch

The epoch paramter defines the number of iterations of the training loop. It
would be inefficient to perform backpropagation on your whole dataset at once
during each iterations of the training loop. This is why we use batches during
which are subsets of your training data. Let's say you have 100 training
samples, in this cases you could choose batches of 20. In that case, for each
epoch you would perform backpropagation 5 times over 20 samples.


Architectures
-------------

Deep neural networks going beyond backpropagation and fully connected layers.

### Convolutions

Convolutional neural networks have been very successfull at image tasks.
Convolutions are performaed on images to extract features. They create feature
maps which represent the dataset and can then be processed in a final fully
connected layer.

### Recurrent Neural Networks

Recurrent neural networks such as Long Short Term Memory (LSTM) and Gated
Recurrent Unit (GRU) netwoorks are mainly used for text and sequence task. They
process input in a sequence updating a hidden state.

### Transformers

Introduced in 2018, this type of architecture is slowly taking over the field
improving the performance of deep neural networks on text tasks but also on
image problems. The attention mechanisms allows the network to look back at
different part of the input and create relatioships between the different parts
of the input sequence.


Practical
=========

Material and Setup
------------------


You need a basic understanding of programming to start deep learning: basic
syntax, control flow, *data structures*, etc.


Required:
- A programming language: Python 3.7+
- A deep learning framework: Tensorflow/*Keras* 
- A place to run experiments and communicate results: Jupyter notebook (or google colab)
- Some hardware: GPU vs cloud compute


Deep learning is code, you need to be able to debug your scripts and
prepare your data.


Datasets
--------


Things to consider before gathering data:
- what real problem are we trying to solve
- what type of deep learning task does this problem fall into
- what information is required to solve the problem


0. Data Preprocessing: Computers can only understand numbers. Preprocessing is
   the transformation complex data types into tensors which can be used by a
   model.
1. Distribution: how to create a good dataset -> balanced, out of sample
   distribution
2. Sample size: what is a minimum sample size? will depend on how complex the
   data is (i.e. hihger dimension will tend to require more data)
3. Split: training, validation and test (and how each are created and used)


Workflow
--------

Projects tend to be iterative in deep learning. A lot of experimentation is
usually required in order to create a model with good performance.

1. Data preprocessing (view and get familiar with your dataset)
2. Define model (start with the simplest possible model, which will be easier to debug)
3. Validate and tune hyperparameters
4. Test model and evaluate performance
5. Revise hypothese and go back to step 2


Potential Issues
----------------


+ Overfitting -> data augmentation, architecture changes, etc.
+ Debugging -> check dimensions
+ Lack of data -> Transfer learning


Tip: Set a clear goal and determine the data type you can use. Build a small
dataset and begin experimenting. Begin with the simplest form of the
problem you wish to solve and as you gain experience and become more
comfortable with deep learning.


Personal Tips
-------------


Set a clear goal and determine the data type you can use. Build a small
dataset and begin experimenting. Start small with the simplest form of the
problem you wish to solve and as you gain experience and become more
comfortable with deep learning.


[Anderj Kaparthy](https://karpathy.github.io/2019/04/25/recipe/) offers some
great insight on how to tackle deep learning projects.


Ressources
==========

Introductory Material
---------------------

* [Nature - Deep
  Learning](https://www.nature.com/articles/nature14539?source=post_page---------------------------)
* [Prediction, Not Association, Paves the Road to Precision
  Medicine](https://jamanetwork.com/journals/jamapsychiatry/article-abstract/2769487)

	In the unfolding era of big data in medicine, the phrase “association is
	not  prediction” should become as important as “correlation is not
	causation.”

Python
------

* [Official python beguinner's guide](https://wiki.python.org/moin/BeginnersGuide)
* [Python Crash Course (book)](https://nostarch.com/pythoncrashcourse2e)


Deep Learning
-------------


- Books 
	* [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
	* [Deep Learning - Ian Goodfellow](https://www.deeplearningbook.org/)
- Online courses:
	* [Coursera Deep Learning](https://www.deeplearning.ai/)
	* [Fastai](https://www.fast.ai/)
- Misc:
	* [Papers with Code](https://paperswithcode.com/)
	* [Kaggle](https://www.kaggle.com/)


Frameworks and Libraries
------------------------


* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)

* [Keras](https://keras.io/)
	- [Keras examples](https://keras.io/examples/)
* [Pytorch](https://pytorch.org/)
