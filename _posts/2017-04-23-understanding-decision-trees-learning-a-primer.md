---
layout: post
title:  "Understanding Decision Tree Learning: A Primer"
date:   2017-03-23 11:34:56 +0530
description: Decision Tree learning is one of the most widely used and practical methods for inductive inference. Decision Trees are easily understood by human and can be developed/used without much pain. In this post I will walk through the basics and the working of decision trees.
categories: Machine-Learning

---

# What is decision tree learning?

Decision tree learning are one of the most commonly known machine learning algorithms out there. One of the advantages of decision trees are that there are quite staright forward, easily understandable by humans. Decision trees provide a way to approximate discrete valued functions and are robust to noisy data. Decision trees can be represented using the typical [Tree](https://en.wikipedia.org/wiki/Tree_(data_structure)) Data Structure. 

>Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning.

In decision tree learning, a decision tree can be used to visually and explicitly represent decisions and decision making. Though, it is common to use a tree-like model for decisions, learned trees can also be represented as sets of `if-else-then` rules. 

Though decision trees can be utilized for both classification and regression, it's primarily used for classification.

## Representating a Decision Tree

Decision trees perform classification after sorting the instances in a top-down approach - from the root to the leaf. Each non-leaf node _splits_ the set of instances based on a test of an attribute. Each branch emanting from a node corresponds to one of the possible values of the said attribute in the node. The leaves of the decision tree specifies the label or the class in which a given instance belongs to. 

Here's an example of a classification tree (Titanic Dataset):

![D Tree]({{site.baseurl}}/images/CART_tree_titanic_survivors.png)

<span style = "color: #dfdfdf; font-size:0.6em">Image courtesy: Wikipedia</span>

The above model uses three attributes namely : _Gender, age and number of spouses/children_. As can be seen from the example, the internal nodes have an attribute test associated with them. This test splits the data set based on the value of the said attribute of the incoming instance. The branches correspond to the values of the attribute in question. At the end, the leaf node represent the class of the instance - in this case the fate of the titanic passengers.  

> Decision Trees represent a disjunction of conjunctions of constraints on attributes values of instances. 

That is, Decision Trees represent a bunch of `AND` 'statements' chained by `OR` statements. For example, let's look at the titanic example above. The given tree can be represented by a disjunction of conjuections as:

	( female ) OR
	( male AND less than 9.5 years of age AND more than 2.5 siblings)

## When should you use a decision tree?

 - When it is imperative for the humans to understand and communicate the model.
 - When you'd like to make minimalistic assumptions from the dataset.
 - When you don't want to normalize the data.
 - When the dataset contains ample amount of noise (but not too much).
 - Presence of Skewed variables in the dataset.
 - When there are many missing attribute values in the dataset.
 - When _disjunctive_ descriptions are required



<br /><br />