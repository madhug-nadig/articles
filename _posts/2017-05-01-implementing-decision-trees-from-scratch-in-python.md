---
layout: post
title:  "Implementing Decision Trees - in Python"
date:   2017-05-01 10:34:56 +0530
description:   Decision Tree learning is one of the most widely used and practical methods for inductive inference. Decision Trees are easily understood by human and can be developed/used without much pain. In this post I will walk through the basics and the working of decision trees In this post I will implement decision trees from scratch in Python.
categories: Machine-Learning

---

Decision tree learning is one of the most commonly known machine learning algorithms out there. One of the advantages of decision trees are that there are quite straight forward, easily understandable by humans. Decision trees provide a way to approximate discrete valued functions and are robust to noisy data. Decision trees can be represented using the typical [Tree](https://en.wikipedia.org/wiki/Tree_(data_structure)) Data Structure.

> Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning.

[Full code here](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch)

Decision tree is a [supervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) algorithm, ie. it needs training data. You will have to feed the algorithm training data for it make predictions on the actual data. Though decision trees can be utilized for both classification and regression, it’s primarily used for classification.  


### How Decision Trees work:

In my previous blog post, [I had explained the theory behind Decision Tree Learning ](http://madhugnadig.com/articles/machine-learning/2017/04/23/understanding-decision-trees-learning-a-primer.html). If you are not very familiar with the algorithm, do check my previous post.

Here’s an example of a classification tree (Titanic Dataset):


![Decision Trees - Titanic Dataset]({{site.baseurl}}/images/CART_tree_titanic_survivors.png)

<script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<!-- Image AD -->
<ins class="adsbygoogle"
     style="display:inline-block;width:728px;height:90px"
     data-ad-client="ca-pub-3120660330925914"
     data-ad-slot="4462066103"></ins>
<script>
(adsbygoogle = window.adsbygoogle || []).push({});
</script>

## Implementing a Decision Tree from scratch:

In this article, I will be focusing on the [Iterative Dichotomiser 3](https://en.wikipedia.org/wiki/ID3_algorithm), commonly know as the ID3 algorithm. Variants/Extensions of the ID3 algorithm, such as C4.5, are very much in practical use today.

The ID3 algorithm greedily builds the tree top-down, starting from the root by meticulously choosing which attribute that will be tested at each given node. Each attribute is evaluated through statistical means as to see which attribute splits the dataset the best. The best attribute is made the root, with it’s attribute values branching out. The process continues with the rest of the attributes. Once an attribute is selected, it is not possible to backtrack.

The implementation can be divided into the following:

1. Handle Data: Clean the file, normalize the parameters, given numeric values to non-numeric attributes. Read data from the file and split the data for cross validation.
2. Implement Entropy function : For this example, I'll be using the [Shannon Entropy](https://en.wiktionary.org/wiki/Shannon_entropy).
3. Implement the best feature split function : Using Shannon's entropy, we find the information gain, which we then use to choose the best feature to split the tree.
4. Implement the create tree: The recursive `createTree` function will build the actual decision tree based on previous functions.

### Predict whether a person would survive the titanic tragedy :

I've used the legendary "Titanic" dataset from the Kaggle. We will be create a model that predicts which passengers survived the [Titanic shipwreck](https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic) based on many input parameters. The _predict class_ is binary: **"survived"** or **"not-survived"**.  

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


## Handling Data:  


I've modified the original data set and removed some of the columns (like passenger_id, ticket number, name, boat number etc) and _I have chosen a very small and contrived subset of the data for this article_. You can find it my github repo [here](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/data/titanic-subset.csv). The remaining parameters in the data set are: `pclass`, `survived`, `sex`, and `embarked`


The original data set has the data description and other related metadata. You can find the original data set from the Kaggle [here](https://www.kaggle.com/c/titanic/data).

The first thing to do is to read the csv file. To deal with the csv data data, let's import Pandas first. Pandas is a powerful library that gives Python R like syntax and functioning.

    import pandas as pd

Now, loading the data file:

    df = pd.read_csv(r".\data\titanic.csv") #Reading from the data file

Now, our task is to convert the non-numerical data elements into numerical formats. Our dataset is a mixture of numbers and character strings. The character strings are binary (`sex` parameter) or ternary (`embarked` parameter) in nature. So, we will be assigning the values of `0`, `1`  and  `0`, `1`, `2` respectively.

      # Sex param
      df.replace('male', 0, inplace = True)
      df.replace('female', 1, inplace = True)

      # Embarked param
      df.replace('S', 0, inplace = True)
      df.replace('C', 1, inplace = True)
      df.replace('Q', 2, inplace = True)


In `main.py`:

        dataset = df.astype(float).values.tolist()
        #Shuffle the dataset
        random.shuffle(dataset) #import random for this


### Setting up the class:

Before we move forward, let's create a class for the algorithm.

      class CustomDecisionTree:
          def __init__(self):
              pass

We have the `CustomDecisionTree`, that will be our main class for the algorithm. In the constructor, we do not have to initialize any value.


## Defining Functions:

Now, let's define the functions that go inside the `CustomDecisionTree` class. For our  implementation, we will only need functions - `calcShannonEnt`, `chooseBestFeatureToSplit`, `createTree` and `predict`.

The `calcShannonEnt` function will use the training data to calculate the entropy of a given dataset. If you are unfamiliar with the terminology or the theoretical fundamentals of Decision Trees, you can refer my previous article [here](http://madhugnadig.com/articles/machine-learning/2017/04/23/understanding-decision-trees-learning-a-primer.html). The `chooseBestFeatureToSplit` function, as the name eloquently suggests, will predict the best feature that can split the decision tree at any given point in decision tree construction. The heuristic to define _best_ is the **information gain**, which is calculated through the entropy. The `createTree` is a recursive function that greedily builds our decision tree from top to bottom. The `createTree` chooses the differentiating attribute (the attribute the splits the tree at any given node) by the decreasing order of the information gain of the attribute. This is handled at the `chooseBestFeatureToSplit` function. The `predict` function will predict the classification for the incoming parameters, deriving it from the tree built by `createTree` from the training dataset.


First we have the `calcShannonEnt` function. We only need a data set as a param for fit, that's all we need to calculate the entropy at this point.

    def calcShannonEnt(self, dataset):
      pass


Let's define the `chooseBestFeatureToSplit` function. The `chooseBestFeatureToSplit` function also takes in a `dataset` param. It is up to the caller which _section_ or _subset_ of our training dataset to pass to find the best split attribute.

    def chooseBestFeatureToSplit(self, dataset):
      pass



Now we have our recursive `createTree` function. The `createTree` function takes in the training dataset. The function that uses the training dataset to model the decision tree.

    def createTree(self, dataset):
      pass


## Implementing the `calcShannonEnt` function:

The `calcShannonEnt` function is the core of how we model out decision tree. This is where we will calculate the entropy training data set provided. In the fit function, we will be be trying to find the entropy for for any given set of data points, essentially trying to find how much **information content** is present in a given set of data points.

> Entropy is a measure of unpredictability of the state, or equivalently, of its average information content.

The formula for this statistical metric:

**Entropy(S) = E(S) = -p<sub>positive</sub> log<sub>2</sub> p<sub>positive</sub> - p<sub>negative </sub>  log<sub>2</sub> p<sub>negative</sub>**


Where p<sub>positive</sub> is the proportion (probability) of positive examples in S and p<sub>negative</sub> is the proportion of negative examples in S. Entropy is 1 if the collection S contains equal number of examples from both classes, Entropy is 0 if all the examples in S contain the same example.


In general terms, when the classes of the target function may not always be boolean, entropy is defined as

![Entropy Formula]({{site.baseurl}}/images/EntropyFormula.png)

So for our case with the titanic data set, the formula would look like:

**Entropy(S) = E(S) = -p<sub>survived</sub> log<sub>2</sub> p<sub>survived</sub> - p<sub>not-survived </sub>  log<sub>2</sub> p<sub>not-survived</sub>**


Implementing this in code, first we will find the count of data points of each label.

	def calcShannonEnt(self, dataSet):
		numEntries = len(dataSet)
		labelCounts = {}
		for featVec in dataSet:
			currentLabel = featVec[-1]
			if currentLabel not in labelCounts.keys():
			 labelCounts[currentLabel] = 0
			 labelCounts[currentLabel] += 1

Now, lets apply the formula for calculating Shannon Entropy:

     shannonEnt = 0.0 # Initialize the entrypy to 0
     for key in labelCounts:
         prob = float(labelCounts[key])/numEntries
         shannonEnt -= prob * log(prob, 2)
     return shannonEnt

## Implementing the `chooseBestFeatureToSplit` function:

The `chooseBestFeatureToSplit` function, as the name suggests, will find the best feature to split the tree for an given set of feature vectors. The function will use the information gain metric to decide which feature most optimally will split the dataset. The information gain is calculated by using Shannon Entropy. We first calulate the `baseEntropy` for the whole dataset and then calculate the entropy of without each subsequent feature. In order to split the dataset, we will first implement the `splitDataSet` function.

  def splitDataSet(self, dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

Now let's get started on the `chooseBestFeatureToSplit` function, let's initialize some important params:

  def chooseBestFeatureToSplit(self, dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = self.calcShannonEnt(dataSet)
    bestInfoGain = -1
    bestFeature = 0

We've initialized the `bestInfoGain` to `-1` and chosen the first feature as the best by default. We've also calculated the base entropy of the entire dataset at this point. Next, we will calculate the `infoGain` for each feature.

      for i in range(numFeatures):
          featList = [example[i] for example in dataSet]
          uniqueVals = set(featList)
          newEntropy = 0.0
          for value in uniqueVals:
              subDataSet = self.splitDataSet(dataSet, i, value)
              prob = len(subDataSet)/float(len(dataSet))
              newEntropy += prob * self.calcShannonEnt(subDataSet)
          infoGain = baseEntropy - newEntropy

We now have the `infoGain` calculated for each feature, by calculating the `newEntropy` - the entropy for the feature in question by using the formula mentioned in the previous section. Now, we can select the feature with the highest information gain.

    if (infoGain > bestInfoGain):
      bestInfoGain = infoGain
      bestFeature = i

The whole funtion:

    def chooseBestFeatureToSplit(self, dataSet, labels):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = -1
        bestFeature = 0
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            print(infoGain, bestInfoGain)
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i

        print("the best feature to split is", labels[bestFeature])
        return bestFeature

## Implementing the `createTree` function:


<br /><br />
