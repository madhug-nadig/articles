---
layout: post
title:  "Implementing K Nearest Neighbours in Parallel from scratch"
date:   2017-02-01 03:34:56 +0530
description: 
categories: Machine Learning, Parallel Processing

---

K Nearest Neighbours is one of the most commonly implemented Machine Learning classification algorithms. In in previous blog post, [I had implemented the algorithm from scratch in Python](/). If you are not very familiar with the algorithm or it's implementation, do check my previous post.

One of the prime drawbacks of the k-NN algorithm is it's efficiency. Being a supervised **[lazy learning](https://en.wikipedia.org/wiki/Lazy_learning)** algorithm, the k-NN waits till the end to compute. On top of this, due to its [non-parametric](https://en.wikipedia.org/wiki/Non-parametric_statistics) 'nature', the k-NN considers the entire dataset as it's model. 

So, the algorithms works on the _entire_ dataset at the _very end_ for _each prediction_. This considerably slows down the performace of k-NN and for larger datasets, it is excruciatingly difficult to apply k-NN due to its inability to scale.

Now, let's see if we can speed up our [previous algorithm](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/K%20Nearest%20Neighbours.py) by applying the concepts of parallel programming.


## Proposal

The brute force version of k-NN that was written previously is [highly parallelizable](http://web.cs.ucdavis.edu/~amenta/pubs/bfknn.pdf). This is due to the fact the computation of the distances between the data points is completely _independent_ of one another. Furthermore, if there are _n_ points in the test set, all of the computation regarding the classification of these _n_ points is independent of one another and can be easily accomplished in parallel. This allows for partitioning the computation work with least synchronization effort. The distance computations can be calculated seperately and then brought together or the dataset itself can be split up into multiple factions to be run in parallel. 

That is, the brute force k-NN has high potential to work faster under [data parallelism](https://en.wikipedia.org/wiki/Data_parallelism):

> Data parallelism is a form of parallelization across multiple processors in parallel computing environments. It focuses on distributing the data across different nodes, which operate on the data in parallel.

The idea is to split the data amongst different processor and then combine them later for procuring final results. The ideal scenario is the case where the processors do not interact with each other, this is the case with brute-force k-NN.

## Implementation

As stated before, there are two options that could be implemented whilst parallizing brute force k-NN. The first is to parallelize the distance finding part within _each_ incoming datapoint, the second is to divide the test data and process on it in parallel. I going to go ahead and implement the latter, simple because it's much easier to implement and the code will be less cluttered.

The implementation revolves around applying data parallelism to the distance finding part of the algorithm. In the parallelizable part, if there are _n_ data points on whom the distance algorithm is to be applied, we will divide the data intp _p_ datasets of size _n/p_ and then let each processor work _independently_ on a data of size _n/p_. In the serial part of the algorithm, we will be dividing the dataset, setting up the code to run in parallel, collect the output from the paralleized region and then continue with the k-NN algorithm.

### Parallel processing in Python

Parallel programming in Python isn't as straight foward as it is in mainstream languages such as Java or C/C++. This is due to the fact that the default python interpreter(Cpython) was designed with simplicity in mind and with the notion that multithreading is [tricky and dangerous](http://www.softpanorama.org/People/Ousterhout/Threads/index.shtml).  The python interpreter has a thread-safe mechanism, the **Global interpreter lock**. 

>Global interpreter lock (GIL) is a mechanism used in computer language interpreters to synchronize the execution of threads so that only one native thread can execute at a time. An interpreter that uses GIL always allows exactly one thread to execute at a time, even if run on a multi-core processor.

Python is restricted to a single OS thread; therefore, it cannot make use of the multiple cores and processors available on modern hardware. Hence, using threads for parallel processing will _not_ work.

As a result, I am using the invaluable **[multiprocessing](http://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#multiprocessing)** module in Python for parallel processing. [I have previously written about working with the multiprocessing library](), do have a look if you are unsure on the working of the module.

### Parallelizable region

The parallelizable in brute force k-NN is the distance finding part. More specifically this code snippet:


	for group in training_data:
			for features in training_data[group]:
				euclidean_distance = np.linalg.norm(np.array(features)- np.array(to_predict))
				distributions.append([euclidean_distance, group])
		

The above for loop is the bottleneck of the k-NN algorithm. We need to parallelize the above for loop. Since we are going to be applying data parallelism, we needn't worry about the actual functions used; we will uilize the same functions again. Applying data parallelism will not affect the actual results in any way. 


