---
layout: post
title:  "Implementing K NEarest Neighbours in Parallel from scratch"
date:   2017-01-19 00:34:56 +0530
description: 
categories: Machine Learning, Parallel Processing
---

K Nearest Neighbours is one of the most commonly implemented Machine Learning classification algorithms. In in previous blog post, [I had implemented the algorithm from scratch in Python](/). If you are not very familiar with the algorithm or it's implementation, do check my previous post.

One of the prime drawbacks of the k-NN algorithm is it's efficiency. Being a supervised **[lazy learning](https://en.wikipedia.org/wiki/Lazy_learning)** algorithm, the k-NN waits till the end to compute. On top of this, dure to its [non-parametric](https://en.wikipedia.org/wiki/Non-parametric_statistics) 'nature',the k-NN considers the entire dataset as it's model. 

So, the algorithms works on the _entire_ dataset at the _very end_ for _each prediction_. This considerably slows down the performace of k-NN and for larger datasets, it is excruciatingly difficult to apply k-NN due to its inability to scale.

Now, let's see if we can speed up our [previous algorithm](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/K%20Nearest%20Neighbours.py) by applying the concepts of parallel programming.


## Proposal

The brute force version of k-NN that was written previously is [highly parallelizable](http://web.cs.ucdavis.edu/~amenta/pubs/bfknn.pdf). This is due to the fact the computation of the distances between the datapoints is completely _independent_ of one another. As such the distance computations can be calculated seperately and then brought together. 

That is, the brute force k-NN has high potential to work faster under [data parallelism](https://en.wikipedia.org/wiki/Data_parallelism):

>Data parallelism is a form of parallelization across multiple processors in parallel computing environments. It focuses on distributing the data across different nodes, which operate on the data in parallel.