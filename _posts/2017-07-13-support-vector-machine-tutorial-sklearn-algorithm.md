---
layout: post
title:  "SVM Tutorial: The Algorithm and sklearn Implementation"
date:   2017-07-13 11:34:56 +0530
description: Support Vector Machines are perhaps one of the most(if not the most) used classification algorithms. One of the prime advantages of SVM is that it works very good right out of the box. You can take the classifier in it’s generic form, without any explicit modifications, run it directly on your data and get good results.
categories: Machine-Learning

---

Support Vector Machines are perhaps one of the most(if not the most) used classification algorithms. They heralded the downfall of the Neural Networks (It was only in the late 2000s that Neural Nets caught on at the advent of Deep Learning and availability of powerful computers) in the 1990s by classifying images efficiently and more accurately. One of the prime advantages of SVM is that it works very good right out of the box. You can take the classifier in it's generic form, without any explicit modifications, run it directly on your data and get good results. In addition to their low error rate, support vector machines are computationally inexpensive in contrast to other calssification algorithms such as the K Nearest Neighbours. 

## So what is a support vector machine?

> A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples.

SVMs are supervised learning algorithms, hence we'll need to train the algorithm before running it on the actual use case. 

> Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.


Let us take the example of a linearly seperable set of points on a 2D plane. There is a possibility of many lines or more generally _hyperplanes_ that might cut across datapoints in which that it splits it into the two given classes.

![Support Vector Machines Seperating lines]({{site.baseurl}}/images/separating-lines.png)

<span style = "color: #dfdfdf; font-size:0.6em">Image courtesy: opencv.org</span>

As can be seen from the above image, there are multiple lines that split the data into the two classes. The question now is, **Which is the best line that seperates the two classes?** and how do we find it?

One practical assumption is that, the farther the datapoint is from this seperating line, the more confidence we have in our prediction. Naturally, we'd want to make all the points of each class as far as possible from the decision boundary. This can be made sure by having the decision boundary be the farthest from points closest to the decision boundary of each class. Unoptimized decision boundary could result in greater misclassifications on new data.


![Support Vector Machines Optimal Hyperplane]({{site.baseurl}}/images/optimal-hyperplane.png)

<span style = "color: #dfdfdf; font-size:0.6em">Image courtesy: opencv.org</span>

The distance between the closest point and the decision boundary is referred to as **margin**. In SVMs, all we are really doing is maximizing this margin. The points closest to the seperating boundary are referred to as **support vectors**. Thus, all SVM does is maximize the distance between the seperating hyperplane and the support vectors. Simple, yeah?

## What's a Hyperplane?

A hyperplane is a n-1 dimensional Euclidean space that can divide any n dimensional Eucildean space into 2 disconnected parts. A hyperplane can be a point - 0 dimensional, a line - 1 Dimensional, a plane - 2 Dimensional and so on. 

Let's break it down a bit. First, let's take a line. Now a single point could break that line into two disconnected parts - in this case the said point is the *seperating hyperplace*. When we take a 2D plane, we can have a line as a seperating hyperplane. Similarly, if we take a 3 dimensional euclidean space, we will need a plane to seperate the 3D space into two disconnected parts. Similarly, if we want to split a space with 512 dimensions into two disconnect parts, we will need a 511 Dimensional Euclidean space to do so.

In call the cases, in order to seperate an n dimensional Euclidean space, we used a n-1 dimensional Euclidean space. This n-1 Dimensional space is your hyperplane.

# How does the SVM work?

First some math:

d<sup>+</sup> : Shortest distance between the decision boundary and the positive support vector

d<sup>-</sup> : Shortest distance between the decision boundary and the negative support vector

So, d<sup>+</sup> + d<sup>-</sup> = Margin

Equation of the hyperplane: 

`w`<sup>`T`</sup>` x + b = 0` , where `w` is the feature vector.  


To make it easier to understand, let us take the example of linearly seperable binary sets. 

The goal here would be to find a hyperplane such that it splits the dataset into two classes, all while making sure that the margin is maximized. 


# Implementing SVMs in Python using scikit-learn 

Now that we have understood the algo, let's go ahead and implement it out of box in Python. We can use to the all-powerful `scikit-learn` library to implement SVM. 

> The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input.

In this tutorial, I am going to focus on classification problems that can be solved using SVMs. One could also use scikit-learn library to solve a variety of regression, density estimation and outlier detection. 

In scikit-learn, we can use the `sklearn.svm.SVC`, `sklearn.svm.NuSVC` and `sklearn.svm.LinearSVC` classes to perform multi-class classification on a dataset. SVC and NuSVC are based on libsvm and LinearSVC is based on liblinear. 

> SVC, NuSVC and LinearSVC take as input two arrays: an array X of size [n_samples, n_features] holding the training samples, and an array y of class labels (strings or integers), size [n_samples]

Let's try out a very simple example of SVC, with linearly seperable binary set:

	from sklearn import svm
	
	data = [[1,1], [6,6]]
	y = [1,6]
	clf = svm.SVC()
	clf.fit(data, y)
	# Print out the support vectors
	print(clf.support_vectors_)
	# Let us make a prediction
	print(clf.predict([[5.,5.]]))
	
Let's do the same for multi-class classification:

	from sklearn import svm
	data = [[1,1], [2,2], [3,3], [4,4]]
	y = [1,2,3,4]
	# For multi class calssification 
	clf = svm.SVC(decision_function_shape = 'ovr')
	clf.fit(data, y)
	# Print out the support vectors
	print(clf.support_vectors_)
	# Let us make some predictions
	print(clf.predict([[5.,5.]]))
	print(clf.predict([[1.,2.]]))

Now, let us work on some real data.

   

# Advantages and Disadvantages of SVMs

## Advantages 

1. Effective in higher dimensions 
2. Works well in extreme cases
3. Good performance right out of the box
4. It is memory efficient
5. Flexibility when Kernel functions are used.

## Disadvantages

1. SVMs can be quite cumbersome for multiclass problems, you would need a model per class.
2. Poor performance in cases where number of features > Number of samples.
3. SVMs, unlike Bayesian classifiers, do not directly provide probability estimates.
4. Choosing the Kernel can be quite tricky
