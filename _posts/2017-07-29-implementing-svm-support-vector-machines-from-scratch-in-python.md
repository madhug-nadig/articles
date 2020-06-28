---
layout: post
title:  "Implementing SVM from Scratch - in Python"
date:   2017-09-29 12:34:56 +0530
description:   Support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. SVMS is one of the most commonly implemented Machine Learning classification algorithms. In this post I will implement the SMV algorithm from scratch in Python.
categories: Machine-Learning

---


Support Vector Machine is one of the most popular Machine Learning algorithms for classification in data mining. One of the prime advantages of SVM is that it works very good right out of the box. You can take the classifier in it's generic form, without any explicit modifications, run it directly on your data and get good results. In addition to their low error rate, support vector machines are computationally inexpensive in contrast to other classification algorithms such as the K Nearest Neighbours.

Support Vector Machine algorithm is a [supervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) algorithm, ie. it needs training data. You will have to feed the algorithm training data for it make predictions on the actual data.

### How a Support Vector Machine works:

In my previous blog post, [I had explained the theory behind SVMs and had implemented the algorithm with Python's scikit learn](http://madhugnadig.com/articles/machine-learning/2017/07/13/support-vector-machine-tutorial-sklearn-algorithm.html). If you are not very familiar with the algorithm or its scikit-learn implementation, do check my previous post.

<script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<!-- Image AD -->
<ins class="adsbygoogle"
     style="display:inline-block;width:728px;height:90px"
     data-ad-client="ca-pub-3120660330925914"
     data-ad-slot="4462066103"></ins>
<script>
(adsbygoogle = window.adsbygoogle || []).push({});
</script>

## Implementing a Support Vector Machine from scratch:

The implementation can be divided into the following:

1. Handle Data: Clean the file, normalize the parameters, given numeric values to non-numeric attributes. Read data from the file and split the data for cross validation.
2. Find Initial Centroids: Choose _k_ centroids in random.
3. Distance Calculation: Finding the distance between each of the datapoints with each of the centroids. This distance metric is used to find the which cluster the points belong to.
4. Re-calculating the centroids: Find the new values for centroid.
5. Stop the iteration: Stop the algorithm when the difference between the old and the new centroids is negligible.

### Predict the presence of Chronic Kidney disease:

I've used the "Chronic Kidney Diseases" dataset from the UCI ML repository. We will be predicting the presence of chronic kidney disease based on many input parameters. The _predict class_ is binary: **"chronic"** or **"not chronic"**.

I shall visualize the algorithm using the mathplotlib module for python.

The dataset will be divided into _'test'_ and _'training'_ samples for **[cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))**. The training set will be used to 'teach' the algorithm about the dataset, ie. to build a model; which, in the case of k-NN algorithm happens during active runtime during prediction. The test set will be used for evaluation of the results.


<script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<ins class="adsbygoogle"
     style="display:block; text-align:center;"
     data-ad-layout="in-article"
     data-ad-format="fluid"
     data-ad-client="ca-pub-3120660330925914"
     data-ad-slot="1624596889"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>


### Setting up the class:

Before we move forward, let's create a class for the algorithm.

    class CustomSVM:
      def __init__(self):
        pass

We have the `CustomSVM`, that will be our main class for the algorithm. In the constructor, we do not have to initialize any value.


## Handling Data:


I've modified the original data set and have added the header lines. You can find the modified dataset [here](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/data/chronic_kidney_disease.csv).

The original dataset has the data description and other related metadata. You can find the original dataset from the UCI ML repo [here](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease).

The first thing to do is to read the csv file. To deal with the csv data data, let's import Pandas first. Pandas is a powerful library that gives Python R like syntax and functioning.

    import pandas as pd

Now, loading the data file:

    df = pd.read_csv(r".\data\chronic_kidney_disease.csv") #Reading from the data file

The first thing is to convert the non-numerical data elements into numerical formats. In this dataset, all the non-numerical elements are of Boolean type. This makes it easy to convert them to numbers. I've assigned the numbers '4' and '2' to positive and negative Boolean attributes respectively.

    def mod_data(df):

        df.replace('?', -999999, inplace = True)
        df.replace('yes', 4, inplace = True)
        df.replace('no', 2, inplace = True)
        df.replace('notpresent', 4, inplace = True)
        df.replace('present', 2, inplace = True)
        df.replace('abnormal', 4, inplace = True)
        df.replace('normal', 2, inplace = True)
        df.replace('poor', 4, inplace = True)
        df.replace('good', 2, inplace = True)
        df.replace('ckd', 4, inplace = True)
        df.replace('notckd', 2, inplace = True)

In `main.py`:

        mod_data(df)
        dataset = df.astype(float).values.tolist()
        #Shuffle the dataset
        random.shuffle(dataset) #import random for this

Next, we have split the data into test and train. In this case, I will be taking 25% of the dataset as the test set:

        #25% of the available data will be used for testing

        test_size = 0.25

        #The keys of the dict are the classes that the data is classfied into

        training_set = {2: [], 4:[]}
        test_set = {2: [], 4:[]}

Now, split the data into test and training; insert them into test and training dictionaries:

        #Split data into training and test for cross validation

        training_data = dataset[:-int(test_size \* len(dataset))]
        test_data = dataset[-int(test_size \* len(dataset)):]

        #Insert data into the training set

        for record in training_data:
				#Append the list in the dict will all the elements of the record except the class
                training_set[record[-1]].append(record[:-1])

        #Insert data into the test set

        for record in test_data:
				# Append the list in the dict will all the elements of the record except the class
                test_set[record[-1]].append(record[:-1])


<div class = "announcement" id = "announcement">
	<span>Still have questions? Find me on <a href='https://www.codementor.io/madhugnadig' target ="_blank" > Codementor </a></span>
</div>

## Defining Functions:

Now, let's define the functions that go inside the `CustomSVM` class. For the simplicity of this tutorial, we will not delve into Advanced SVM topics such as
[kernels](https://en.wikipedia.org/wiki/Kernel_method). For our  simple implementation, we will only need two functions - `fit` and `predict`.

As their names suggest, the `fit` function will use the incoming data to model a SVM - essentially calculating the values for  `W` (feature vector) and `b` (bias). If you are unfamiliar with the terminology or the theoretical fundamentals of SVMs, you can read about it [here](http://madhugnadig.com/articles/machine-learning/2017/07/13/support-vector-machine-tutorial-sklearn-algorithm.html). The predict function will predict the classification for the incoming parameters, deriving it from the model we 'fit' from the training dataset.


First we have the `fit` function. We only need the pre-processed data set as a param for fit, that's all we need to model a SVM at this point.

  def fit(self, dataset):
    pass



Let's define the `predict` function. The predict function take in the attributes, ie. the incoming values for which we need to make a prediction. The `predict` function will use the model that will by created by the `fit` function

  def predict(self, attrs):
    pass

### Distance Metric:

The k-means algorithm, like the k-NN algorithm, relies heavy on the idea of _distance_ between the data points and the centroid. This distance is computed is using the **distance metric**. Now, the decision regarding the decision measure is _very, very imperative_ in k-Means. A given incoming point can be predicted by the algorithm to belong one cluster or many depending on the distance metric used. From the previous sentence, it should be apparent that different distance measures may result in different answers.

There is no sure-shot way of choosing a distance metric, the results mainly depend on the dataset itself. The only way of surely knowing the right distance metric is to apply different distance measures to the same dataset and choose the one which is most accurate.

In this case, I will be using the **[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)** as the distance metric (through there are other options such as the **[Manhattan Distance](https://en.wiktionary.org/wiki/Manhattan_distance), [Minkowski Distance](https://en.wikipedia.org/wiki/Minkowski_distance)** ). The Euclidean distance is straight line distance between two data points, that is, the distance between the points if they were represented in an _n-dimensional Cartesian plane_, more specifically, if they were present in the _Euclidean space_.


### Implementing Euclidean distance for two features in python:

    import math

    def Euclidean_distance(feat_one, feat_two):

        squared_distance = 0

        #Assuming correct input to the function where the lengths of two features are the same

        for i in range(len(feat_one)):

                squared_distance += (feat_one[i] – feat_two[i])**2

        ed = sqrt(squared_distances)

        return ed;

The above code can be extended to _n_ number of features. In this example, however, I will rely on Python's numpy library's function: `numpy.linalg.norm`


<div class = "announcement" id = "announcement">
	<span>Still have questions? Find me on <a href='https://www.codementor.io/madhugnadig' target ="_blank" > Codementor </a></span>
</div>

## Clustering:

After figuring out the distances between the points, we will use the distances to find which cluster amongst the _k_ clusters a given data point belongs to.  

First, let's initialize the centroids randomly:

	#initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
	for i in range(self.k):
		self.centroids[i] = data[i]

Now, let's enter the main loop.

	for i in range(self.max_iterations):
			self.classes = {}
			for i in range(self.k):
				self.classes[i] = []

			#find the distance between the point and cluster; choose the nearest centroid
			for features in data:
				distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classes[classification].append(features)

The main loop executes `max_iterations` number of times at most. We are defining the  each cluster in the `classes` list. Then we iterate through the features in data and find the distance between the features of the data point and the features of the centroid. After finding the cluster nearest to the datapoint, we append the cluster list within `classes` with the data point's feature vector.

Now, let's re-calculate the cluster centroids.

	previous = dict(self.centroids)

	#average the cluster datapoints to re-calculate the centroids
	for classification in self.classes:
		self.centroids[classification] = np.average(self.classes[classification], axis = 0)

The dictionary `previous` stores the value of centroids that the previous iteration returned, we performed the clustering in this iteration based on these centroids. Then we iterate though the `classes` list and find the average of all the datapoints in the given cluster. This is, perhaps, the _machine learning_ part of k-means. The algorithm recomputes the centroids as long as it's optimal(or if there have been far too many interations in  attempting to do so).

Time to see if our algorithm has reached the optimal values of centroids. For this, let's have a flag `isOptimal`.

	isOptimal = True

Let's iterate though the new centroids and compare it with the older centroid values and see if it's converged.

	for centroid in self.centroids:

		original_centroid = previous[centroid]
		curr = self.centroids[centroid]

		if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
			isOptimal = False

		#break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)

	if isOptimal:
			break

We find the situation to be optimal if the percentage change in the centroid values is lower than our accepted value of tolerance. We break out of the main loop if we find that the algorithm has reached the optimal stage, ie. the changes in the values of centroids, if the algorithm continued to execute, is negligible.

## Visualizing the clusters

Now that we are done with clustering, let us visualize the datasets to see where these clusters stand at. I will be using python [matplotlib](http://matplotlib.org/) module to visualize the dataset and then color the different clusters for visual identification.

In main:


	km = K_Means(3)
	km.fit(X)

	# Plotting starts here, the colors
	colors = 10*["r", "g", "c", "b", "k"]

Lets mark our centroids with an `x`.

	for centroid in km.centroids:
		plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s = 130, marker = "x")

Now, let's go ahead and plot the datapoints and color them based on their cluster.


	for classification in km.classes:
		color = colors[classification]
		for features in km.classes[classification]:
			plt.scatter(features[0], features[1], color = color,s = 30)

Show the plot:

	plt.show()

Output:
<style>
.mpld3-yaxis line, .mpld3-yaxis path {
    shape-rendering: crispEdges;
    stroke: black;
    fill: none;
}

Here we have our scatter plot with clustering done on it using K Means clustering algorithm. The three clusters can be thought of as _Batsmen_ (<span style = "color:red">Red</span>), _Bowlers_(<span style = "color:green">Green</span>) and _Allrounders_(<span style = "color:blue">Blue</span>).

That's it for now; if you have any comments, please leave them below.


<br /><br />
