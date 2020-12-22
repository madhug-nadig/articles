---
layout: post
title: "Understanding Genetic Algorithms and implementing it using a Python library"
date: 2017-11-04 09:23:00 +0530
description: Genetic algorithms are a class of machine learning algorithms which approximate the process of natural selection seen in nature. Genetic algorithms belong to a larger set Evolutionary algorithms, which take Charles Darwin's evolution theory as the center piece. Genetic algorithms are widely used to for solving variety of optimization algorithms in many different domains.  
categories: Machine-Learning
---

# What are Genetic Algorithms?

Genetic algorithms are a class of machine learning algorithms which approximate the process of natural selection seen in nature. Genetic algorithms belong to a larger set of [Evolutionary algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm), which take Charles Darwin's evolution theory as the center piece of inspiration. Genetic algorithms are widely used in solving variety of optimization algorithms in many different domains. Biological processes such as mutation, crossover and selection are heavily relied upon as a source of inspiration in the implementation of generic algorithms.  

The main use cases for genetic algorithms are **optimization**, **Classification** and **Human Comparable Behaviors**. Genetic algorithms are essentially a way for of performing **biologically inspired optimized trial and error**.  

In biology, the organisms evolve to suit their environment better, in genetic algorithms, we define the environment (the end result) and we evolve a list of potential solutions until they are converge into an ideal fit to our predefined environment.  

<div style="text-align:center">
![Antenna designed using Genetic Algorithms]({{site.baseurl}}/images/antenna.jpg)
</div>

<br />

<span style = "color: #dfdfdf; font-size:0.6em">Image courtesy:Wikipedia</span>  


> The 2006 NASA ST5 spacecraft antenna. This complicated shape was found by an evolutionary computer design program to create the best radiation pattern. It is known as an evolved antenna.  

Genetic algorithms are mostly utilized for the following use cases:
* Operations Research
* Biology
* Economics
* AI Design (Like the antenna above)
* Game theory
* Sociology

Genetic algorithms are especially potent in time tabling problems, scheduling problems and generating optimized design, such as designing a network topology.

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

# Prerequisites:

Lets quickly go over and acquaint ourselves with some of the underlying biological concepts that are abstracted in genetic algorithms.  

### Population and Gene Pool:

A gene pool is a collection of genetic information in any population. In biology, a gene pool represents the total genetic diversity of population. A gene is the unit information structure, a sequence of genes form a chromosome - which encapsulate all the information about the member - and a population consists of members.  

<div style="text-align:center">
![Genetic Algorithms]({{site.baseurl}}/images/genetic.png)
</div>


In the application of genetic algorithms, we can abstract the population as a **list of potential solutions**. In genetic algorithms, each member of a population is a potential solution and would _evolve_ to reach an optimal solution. A gene can be abstracted to a single feature and a chromosome can be abstracted to a single data point with a bunch of features that has all the information to describe it.  

### Fitness

Fitness involves the ability of populations or species to survive and reproduce in the environment in which they find themselves 6–9. The consequence of this survival and reproduction is that organisms contribute genes to the next generation. From the theory of evolution, only the fittest members of the species survive and pass on their genes to the new generation. This way, the generation are more likely to survive the environment.  

In the application of genetic algorithms, we abstract the notion of fitness into a **fitness function**, which gives a fitness score to each member of the population. The probability that a member will be selected for reproduction is based on its fitness score. In general, the fitness function evaluates how close a member of the population is in relation to the end state.   

### Cross-Over

Chromosomal crossover, or crossing over, is the exchange of genetic material during reproduction between two members of the population. The new generation created will have genes of both the original members.

In genetic algorithms, we use crossover whilst creating a new generation of population. Only certain members of the population that are more fit than others are chosen for cross over.   

### Mutation

Biologically, a Mutation occurs when a DNA gene is damaged or changed in such a way as to alter the genetic message carried by that gene. A Mutagen is an agent of substance that can bring about a permanent alteration to the physical composition of a DNA gene such that the genetic message is changed. There are 3 types of biological mutations:

1. **Base Substitutions**: Involve the swapping of one nucleotide for another during DNA replication ie. swapping one value with another.  
2. **Deletion**: Adding an nucleotide ie. removing a value from a member of the population.  
3. **Insertions**: Adding an extra nucleotide ie. Adding an extra value to a member of the population.  

More info on the actual biology can be found [here](http://www2.csudh.edu/nsturm/CHEMXL153/DNAMutationRepair.htm).  

In genetic algorithms, the process of mutation inserts a sense of randomness to the population - this enables the algorithm to expand it's solution set beyond the current values of the population. With mutation, an algorithm can look farther for a solution than the current set of features, since the process of crossover will only re-assemble the existing values.  

Mutation also ensures to maintain diversity within the population and prevents premature convergence or over fitting.  

### Termination

Although in nature evolution keeps going on, we only want to run an algorithm a finite amount of time. For this, genetic algorithms usually have a termination condition. Once this condition is met, the algorithm stops recreating populations and will emit the last created generation as the solution.  

Termination conditions usually fall into one of the three:

1. The overall fitness of the population is above a certain, predefined threshold.  
2. After many generations, there is no improvement seen in fitness.  
3. A predefined absolute number of generations have been created.  


# How Genetic Algorithms Work:

In genetic algorithms, **we generate a random list of population, each member of which can be our potential optimal solution. Using processes like mutation and cross over, we will evolve this population to reach some level "optimality" based on our Selection criteria.** Each application of mutation and cross over creates a new generation of population and each new generation of population is more fit and optimal than the previous generation.  


## Genetic Algorithm Lifecycle:

There are 5 main steps in genetic algorithms:

1. **Initial population**: Usually randomly generated.  
2. **Fitness function**: A metric to evaluate how fit an abritrary member of population is.  
3. **Selection**: Selecting the most "fit" members of population for the next iteration.  
4. **Crossover**: Reproduction of new generation of members based of the most fit members of the previous generations.  
5. **Mutation**: Adding changes to the new members to maintain diversity and to expand the solution space.  

We keep performing steps 2 through 5 until we have reach the optimal composition of the members, at which point we emit the "chromosomes" (The values of the solution).

In case of practical application, before we can create an initial population, first we would have to encode the problem, ie. find a way to represent the problem and the solution space as a sequence of characters.

## Solving using genetic algorithm:  

I will solve a problem using genetic algorithm to illustrate how the algorithm works.


## When should you use Genetic Algorithms?

-   When the solution space is less understood or is too big.  
-   When the solution space is unstructured.  
-   The presence of multiple conflicting solutions or partial solutions.  
-   When you need to simulate Human Comparable behaviors(trial and error).  
-   When the problem to be solved can be approximated to a search in combinatorial space.  


# Implementing Genetic Algorithms using Sklearn-genetic:

Now that we have understood the algorithm, let’s go ahead and implement it out of box in Python. We can use Python's all-powerful `scikit-learn` library to implement DBSCAN.

> DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Perform DBSCAN clustering from vector array or distance matrix. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.

In this tutorial, I am going to focus on contrived clustering problems that can be solved using BDSCAN. One could also use `scikit-learn` library to solve a variety of clustering, density estimation and outlier detection problems. I will be using two toy datasets that make DBSCAN Standout - Two Concentric circles and 2 moons - these are oddly shaped data sets where DBSCAN would outperform other clustering methods like K-Means.

In scikit-learn, we can use the `sklearn.cluster.DBSCAN` class to perform density based clustering on a dataset. The `scikit-learn` implimentation takes in a variety of input parameters that can be [found here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). The most interesting of them is the values of `eps` (which defaults to `0.5`) and `min_samples` (which defaults to 5). With the sklearn implementation, you can also provide option for distance metric (defaults to `Euclidean`), additional params for the metric, the type of clustering and so on.

## Creating the Dataset

As mentioned I will be creating two toy datasets from `scikit-learn` library. I will be using `make_circles` and `make_moons` from the dataset package:

```
from sklearn import cluster, datasets


np.random.seed(0)

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.08)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.08)
```

## Fitting the data to the model

Now that we have the datasets ready, we go ahead and fit the datasets into our model. For these examples I have chosen the value of `eps` as `0.2`. We will leave the rest of the params to their default value for the sake of simplicity.

```
eps = 0.2

datasets = [
    noisy_circles,
    noisy_moons,
]
```

Let's iterate through the datasets and normalize the dataset. I am going to use the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) from the library to do so,

```
for i_dataset, dataset in enumerate(datasets):

    X, y = dataset

    # normalize dataset
    X = StandardScaler().fit_transform(X)

```

Now we can just initialize the class and fit the data:

```
    dbscan = cluster.DBSCAN(eps=eps)

    dbscan.fit(X)
```

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

<div class = "announcement" id = "announcement">
	<span>Still have questions? Find me on <a href='https://www.codementor.io/madhugnadig' target ="_blank" > Codementor </a></span>
</div>

That's it for now, if you have any comments, please leave then below.

<br /><br />
