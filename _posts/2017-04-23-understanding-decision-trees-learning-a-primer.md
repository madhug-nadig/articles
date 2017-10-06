---
layout: post
title:  "Machine Learning: Understanding Decision Tree Learning"
date:   2017-04-23 11:34:56 +0530
description: Decision Tree learning is one of the most widely used and practical methods for inductive inference. Decision Trees are easily understood by human and can be developed/used without much pain. In this post I will walk through the basics and the working of decision trees.
categories: Machine-Learning

---

<style>
table, th, td {
    border: 1px solid #efefef;
	padding:3px;
}
</style>

# What is decision tree learning?

Decision tree learning is one of the most commonly known machine learning algorithms out there. One of the advantages of decision trees are that there are quite staright forward, easily understandable by humans. Decision trees provide a way to approximate discrete valued functions and are robust to noisy data. Decision trees can be represented using the typical [Tree](https://en.wikipedia.org/wiki/Tree_(data_structure)) Data Structure. 

>Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning.

In decision tree learning, a decision tree - now known by the umbrella term [CART (Classification and Regression Tree)](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29) - can be used to visually and explicitly represent decisions and decision making. Though, it is common to use a tree-like model for decisions, learned trees can also be represented as sets of `if-else-then` rules. 

Though decision trees can be utilized for both classification and regression, it's primarily used for classification.

## Representating a Decision Tree

Decision trees perform classification after sorting the instances in a top-down approach - from the root to the leaf. Each non-leaf node _splits_ the set of instances based on a test of an attribute. Each branch emanting from a node corresponds to one of the possible values of the said attribute in the node. The leaves of the decision tree specifies the label or the class in which a given instance belongs to. 

Here's an example of a classification tree (Titanic Dataset):

![D Tree]({{site.baseurl}}/images/CART_tree_titanic_survivors.png)

<span style = "color: #dfdfdf; font-size:0.6em">Image courtesy: Wikipedia</span>

The above model uses three attributes namely : _Gender, age and number of spouses/children_. As can be seen from the example, the internal nodes have an attribute test associated with them. This test splits the data set based on the value of the said attribute of the incoming instance. The branches correspond to the values of the attribute in question. At the end, the leaf node represent the class of the instance - in this case the fate of the titanic passengers.  

> Decision Trees represent a disjunction of conjunctions of constraints on attributes values of instances. 

That is, Decision Trees represent a bunch of `AND` 'statements' chained by `OR` statements. For example, let's look at the titanic example above. The given tree can be represented by a disjunction of conjuctions as:
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
 - When you need to build and test fast
 - When the dataset is small is size

# How is a decision tree is built?

Before we start classifying, we first need to build the tree from the available dataset. 

> Most algorithms that have been developed for learning decision trees are variations of the core algorithm that employs a __top down__, __greedy__ search through the possible space of decision trees.

In this article, I will be focussing on the [Iterative Dichotomiser 3](https://en.wikipedia.org/wiki/ID3_algorithm), commonly know as the ID3 algorithm. Variants/Extensions of the ID3 algorithm, such as C4.5, are very much in practical use today. 

The ID3 algorithm builds the tree top-down, starting from the root by meticulously choosing which attribute that will be tested at each given node. Each attribute is evaluated through statistical means as to see __which attribute splits the dataset the best.__ The best attribute is made the root, with it's attribute values branching out. The process continues with the rest of the attributes. Once an attribute is selected, it is not possible to _backtrack_. 


## Choosing the attribute

### Entropy

> Entropy is a measure of _unpredictability_ of the state, or equivalently, of its _average information content._ 

Entropy is a statistical metric that measures that **impurity.** Given a collection of S, which contains two classes: _Positive_ and _Negative_, of some arbitrary target concept. The entropy with respect to this boolean classification is:
		
**Entropy(S) = E(S) = -p<sub>positive</sub>log<sub>2</sub> p <sub>positive</sub> - p<sub>negative</sub>log<sub>2</sub> p <sub>negative</sub>**

Where p<sub>positive</sub> is the proportion (probability) of positive examples in S and p<sub>negative</sub> is the proportion of negative examples in S. Entropy is 1 if the collection S contains equal number of examples from both classes, Entropy is 0 if all the examples in S contain the same example.

The entropy values vs the probabilities for a collection S follows a parabolic curve:

![Entropy vs Probablity]({{site.baseurl}}/images/EntropyGraph.png)

<span style = "color: #dfdfdf; font-size:0.6em">Image courtesy: MATLAB data science</span>

One interpretation of entropy is that, entropy specifies the minimum number of bits required to __encode__ the classification of any member of a collection S.

In general terms, when the classes of the target function may not always be boolean, entropy is defined as

![Entropy Formula]({{site.baseurl}}/images/EntropyFormula.png)

### Information Gain

Now that we know what entropy is, let's look at an attribute that is more attached to the building of the decision tree - _[Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)_. THe information gain is a metric that measures the expected reduction in the impurity of the collection S, caused by splitting the data according to any given attribute. 

The information gain IG(S,A) of an attribute A, from the collection S, can be defined as

![Information Gain Formula]({{site.baseurl}}/images/IG.png)

<span style = "color: #dfdfdf; font-size:0.6em">Image courtesy: Abhyast</span>

where `i` spans through the entire set of all possible values for attribute A, and S <sub>i</sub> is the portion of S for which attribute A has the value _i_. The first term is the entropy of the entire collection S. One way to think about IG is that, the value of IG is the number of bits saved when encoding a target value of an arbitrary member of the collection. 

Whilst building the decision tree, the information gain metric is used by the ID3 algorithm to select the best attribute - the attribute the provides the "best split" - at each level. 

# Complete example of decision tree learning

Let's take the example of a dataset. This dataset assesses the risk of tumour in a patient. We will be generating a decision tree using the ID3 algorithm.

<table >
	<tbody>
		<tr>
			<td> HEADACHE </td>
			<td> DIZZYNESS </td>
			<td> BLOOD PRESSURE </td>
			<td> RISK </td>
		</tr>
		<tr>
			<td> YES </td>
			<td> NO </td>
			<td> HIGH </td>
			<td> YES </td>
		</tr>
		<tr>
			<td> YES </td>
			<td> YES </td>
			<td> HIGH </td>
			<td> YES </td>
		</tr>
		<tr>
			<td> NO  </td>
			<td> NO </td>
			<td> NORMAL </td>
			<td> NO </td>
		</tr>
		<tr>
			<td> YES  </td>
			<td> YES </td>
			<td> NORMAL </td>
			<td> YES </td>
		</tr>
		<tr>
			<td> YES </td>
			<td> NO </td>
			<td> NORMAL </td>
			<td> NO </td>
		</tr>
		<tr>
			<td> NO </td>
			<td> YES </td>
			<td> NORMAL </td>
			<td> YES </td>
		</tr>
	</tbody>
</table> 

First, let's find the entropy of the entire collection: 

__E(S) = -p<sub>yes</sub> log<sub>2</sub> p<sub>yes</sub> - p<sub>no</sub> log<sub>2</sub> p<sub>no</sub>__

From the dataset: _p<sub>yes</sub>_ = 4/6 and _p<sub>no</sub>_ = 2/6

So, E(S) = - { (4/6) log<sub>2</sub>(4/6) } - { (2/6) log<sub>2</sub>(2/6) }

This gives us **E(S) = 0.9182**

Now, the information gain, let's consider the attribute `HEADACHE`. This attribute has two values `YES` and `NO`.  Now, the proportion of `YES` in the attribute: 4/6 and the proportion of `YES` in the attribute: 2/6

Hence, the split:

S<sub>YES</sub> - [3+, 1-] ( 3 positive and 1 negative classification when HEADACHE has the value YES)

S<sub>NO</sub> - [0+, 2-] ( 2 negative classifications when HEADACHE has the value NO)

Therefore,

IG(S, HEADACHE) = E(S) - (4/6) * E(S<sub>YES</sub>) - (2/6) * E(S<sub>NO</sub>).  
After Calculation: **IG(S, HEADACHE) = 0.37734**.  
Similarly, **IG(S, DIZZYNESS) = 0.4590** and **IG(S, BP) = 0.5848**

After we have calculated the information for these attributes, we choose the attribute with the highest information gain as the splitting attribute for the node. This process goes on top-down until we are left we just leaves - the classification. From above, it is clear that the attribute `BLOOD PRESSURE` will be our attribute of choice at the root node. 



<br /><br />