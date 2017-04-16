---
layout: post
title:  "Implementing K Nearest Neighbours from Scratch - in Python"
date:   2017-01-13 17:34:56 +0530
description:   k -Nearest Neighbors algorithm (or k-NN  for short) is a  non-parametric method used for  classification and  regression. K Nearest Neighbours is one of the most commonly implemented Machine Learning classification algorithms. In this post I will implement the algorithm from scratch in Python.
categories: Machine-Learning

---


K Nearest Neighbours is one of the most commonly implemented Machine Learning classification algorithms. The kNN is more widely used in classification problems than for regression problems, although it can be applied for both classification and regression problems. 

The kNN algorithm is easy to understand and to implement. It works well in a large number of cases and is a powerful tool to have in the closet.

This is what Wikipedia says about k-NN:

 > **k -Nearest Neighbors algorithm** (or **k-NN**  for short) is a  [non-parametric](https://en.wikipedia.org/wiki/Non-parametric_statistics) method used for  [classification](https://en.wikipedia.org/wiki/Statistical_classification) and  [regression](https://en.wikipedia.org/wiki/Regression_analysis).

The k-NN algorithm derives much of its power from the fact that it's _non-parametric_; this means that the algorithm has **no prior bias or a functional form**. It does not make any assumptions about the dataset in hand.


### How k-NN works:

k-NN algorithm uses the entire training dataset as its model. For a new prediction to come out of k-NN, the algorithm will scour the entire training data to find '_k_' points nearest to the incoming data. Then, depending on the class of these '_k_' points, the probable class of the incoming data is predicted.

The computation only happens on the arrival of new data. Each time a new data-point is fed into k-NN, it will again search through the entire dataset to find those '_k_' points. The algorithm does not build any model until the time a classification-based prediction is required. This technique of performing all the computation whilst the classification(ie. in the end) is referred to as **[lazy learning](https://en.wikipedia.org/wiki/Lazy_learning)** or **[instance based-learning](https://en.wikipedia.org/wiki/Instance-based_learning)**.

### Predict the presence of Chronic Kidney disease:

I've used the "Chronic Kidney Diseases" dataset from the UCI ML repository. We will be predicting the presence of chronic kidney disease based on many input parameters. The _predict class_ is binary: **"chronic"** or **"not chronic"**.

The dataset will be divided into _'test'_ and _'training'_ samples for **[cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))**. The training set will be used to 'teach' the algorithm about the dataset, ie. to build a model; which, in the case of k-NN algorithm happens during active runtime during prediction. The test set will be used for evaluation of the results.

## Implementing k-NN:

The implementation can be divided into the following:

1. Handle Data: Clean the file, normalize the parameters, given numeric values to non-numeric attributes. Read data from the file and split the data for cross validation.
2. Distance Calculation: Finding the distance between two data points. This distance metric is used to find the '_k_' points.
3. Prediction: Find the '_k_' points and make predictions on the incoming data.
4. Testing/Evaluation: Test the data using the testing set. Find the accuracy.

### Setting up the class:

Before we move forward, let's create a class for the algorithm.

    class CustomKNN:

        #constructor

        def __init__(self):

                self.accurate_predictions = 0

                self.total_predictions = 0

                self.accuracy = 0.0

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



## Distance Calculation:

### Normalizing Dataset:

Before calculating distance, it is very important to **Normalize** the dataset - to perform **[feature scaling](https://en.wikipedia.org/wiki/Feature_scaling)**. Since the distance measure is directly dependent on the _magnitude_ of the parameters, the features with higher average values will get more preference whilst decision making; for example, in the dataset in our case, the feature '_age_' might get more preference since its values are higher than that of other features. Not normalizing the data prior to distance calculation may reduce the accuracy.

I will be using sci-kit learn's `preprocessing` to scale the data.

	from sklearn import preprocessing

        #Normalize the data

        x = df.values #returns a numpy array

        min_max_scaler = preprocessing.MinMaxScaler()

        x_scaled = min_max_scaler.fit_transform(x)

        df = pd.DataFrame(x_scaled) #Replace df with normalized values

### Distance Metric:

The k-NN algorithm relies heavy on the idea of _similarity_ of data points. This similarity is computed is using the **distance metric**. Now, the decision regarding the decision measure is _very, very imperative_ in k-NN. A given incoming point can be predicted by the algorithm to belong one class or many depending on the distance metric used. From the previous sentence, it should be apparent that different distance measures may result in different answers.

There is no sure-shot way of choosing a distance metric, the results mainly depend on the dataset itself. The only way of surely knowing the right distance metric is to apply different distance measures to the same dataset and choose the one which is most accurate.

In this case, I will be using the **[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)** as the distance metric (through there are other options such as the **[Manhattan Distance](https://en.wiktionary.org/wiki/Manhattan_distance), [Minkowski Distance](https://en.wikipedia.org/wiki/Minkowski_distance)** ). The Euclidean distance is straight line distance between two data points, that is, the distance between the points if they were represented in an _n-dimensional Cartesian plane_, more specifically, if they were present in the _Euclidean space_.

From Wikipedia:

>In  [Cartesian coordinates](https://en.wikipedia.org/wiki/Cartesian_coordinates), if  **p**  = (p1, p2,..., pn) and  **q**  = (q1, q2,..., qn) are two points in  [Euclidean n-space](https://en.wikipedia.org/wiki/Euclidean_space), then the distance (d) from  **p**  to  **q** , or from  **q**  to  **p**  is given by:

> ![](data:image/*;base64,iVBORw0KGgoAAAANSUhEUgAABAsAAADsCAIAAACdVrjJAAAAAXNSR0IArs4c6QAAUMFJREFUeF7tnXfUXcV1tyMcY8A4IQZCWQgMBjtgwECooks0mWYEiGqKkUwiIxbFpthCS6aYgIPBiLYQRWA6ohpkTOgGhEgCmIDpnRXAYAdsipMY9D1hf9lMzr33nDnl3nvuub/7x7ve975T9n5mzpzZM3v2jJg3b96f6SMCIiACIiACIiACIiACIiACHxOYTxxEQAREQAREQAREQAREQAREwAnIQlBnEAEREAEREAEREAEREAER+ISALAT1BhEQAREQAREQAREQAREQAVkI6gMiIAIiIAIiIAIiIAIiIALtCGgPQf1CBERABERABERABERABERAewjqAyIgAiIgAiIgAiIgAiIgAtpDUB8QAREQAREQAREQAREQARFIJyAvI/UQERABERABERABERABERCBTwjIQlBvEAEREAEREAEREAEREAERkIWgPiACIiACIiACIiACIiACItCOgPYQ1C9EQAREQAREQAREQAREQAS0h6A+IAIiIAIiIAIiIAIiIAIioD0E9QEREAEREAEREAEREAEREIF0AvIyUg8RAREQAREQAREQAREQARH4hIAsBPUGERABERABERABERABERABWQjqAyIgAiIgAiIgAiIgAiIgAu0IaA9B/UIEREAEREAEREAEREAEREB7COoDIiACIiACIiACIiACIiAC2kNQHxABERABERABERABERABEUgnIC8j9RAREAEREAEREAEREAEREIFPCMhCUG8QAREQAREQAREQAREQARGQhaA+IAIiIAIiIAIiIAIiIAIi0I6A9hDUL0RABERABERABERABERABLSHoD4gAiIgAiIgAiIgAiIgAiKgPQT1AREQAREQAREQAREQAREQgXQC8jJSDxEBERABERABERABERABEfiEgCwE9QYREAEREAEREAEREAEREIFPCIyYN2+eeIiACIhAGQIffPDBQgstVKYE5RUBERABERCBwSXQvOm09hAGtzdKchGoC4ELL7ywLqJIDhEQAREQAREQgdIEtIdQGqEKEIGhJzBq1Kg5c+aceuqphxxyyNDDEAAREAEREAERGHgC2kMY+CaUAiLQXwKvvvrqs88+u9xyy2222Wb9lUS1i4AIiIAIiIAIVEJAFkIlGFWICAwvgblz57755pvzzTffGmusMbwUpLkIiIAIiIAINIiALIQGNaZUEYF+ELjooouodrfddutH5apTBERABERABESgegKyEKpnqhJFYHgI/O53v/vZz362+OKLb7fddsOjtTQVAREQAREQgWYTkIXQ7PaVdiLQXQL33XcfFeBltNZaa3W3JpUuAiIgAiIgAiLQKwKyEHpFWvWIQBMJXHHFFah10EEHLbjggk3UTzqJgAiIgAiIwDASULTTYWx16SwClRDwi9LuvffeDTfcsJIyVYgIiIAIiIAIiEDfCchC6HsTSAARGFQCuBhttNFGSP/b3/7285///KCq0US5CUGL2faHP/yBKLRrr722WqeJjVyNTo888sijjz76n//5n6usssrKK6+srlINVpXSLALhiMqTsswyyzRLv/bayEIYhlaWjiLQFQKTJ08+44wztt9++xtvvLErFajQ/ATY2Dn44INnzZr19ttvW+4ll1xyjz32OOGEE+QJlh9nk3Mw6Zk4ceItt9wSKklosn322afJaks3EchDwEbU8847zzMNz4gqCyFPT1FaERCB/yXgLkaYBxgJAlMTAvvtt5/Fn913333vvPPOl19+2QTbaaedrr322poIKTH6ToDnd/XVV+euw+WXX36DDTa47LLLTCTikn3ve9/T5eh9byAJUBMCWNFmHiRG1GFYGtNJ5Zp0QokhAgNG4KGHHmIywWqKTiDUp+XwGLnhhhuQZ8aMGTNnznzyySfPPPNMEw+nozvuuKM+okqS/hKYPXs25gEyYDdeeumlOAoyAeJP4pKddNJJbC/0VzzVLgJ1IPD000+zH9t2RH3ggQcaP6LKQqhDJ5QMIjB4BG666SYmE9yjLMfl+jQeDuXmXPSDH/yARWLcivbff38Tj8a67rrr6iOqJOkvAXcuOvDAA5GEp9j3DV5//fXGT336C1+1DwoBzAAfUbn8JzGiXnzxxYOiSDE5m2YhMK6tsMIKK620Em1ZjEi/crFmM2LECIQfwsWbs846C93HjRvXL/iF673mmmuQfIcddmA2VriQAc145ZVXInmMfxFdmkdy5MiRrHAPqLJ1EHuTTTahs/GwpAjzV3/1V/bfT33qU6wK8wuvNG+j+o8tgzuA16GHmAyMRbxH6Crps3yOsFt67AH7BWvftfj9739fH40kCW162GGHZT7+ApWLgI2oxx9/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rfDq8b333rs2okkQERABERABERCBnhKQhdBT3KpMBGpLIIx5yqnlHXbYobaiSjAREAEREAEREIGuEpCF0FW8KlwEBonALrvsYuJyalkuRoPUcpJVBERABERABColIAuhUpwqTAQGmcDo0aMt5ulOO+00yHpIdhEQAREQAREQgVIEZCGUwqfMItAkAuwbLLzwwhtssMH48eObpJd0EQEREAEREAERyEVgxLx583JlUGIREAEREAEREAEREAEREIEGE9AeQoMbV6qJgAiIgAiIgAiIgAiIQG4CshByI1MGERABERABERABERABEWgwAVkIDW5cqSYCIiACIiACIiACIiACuQnIQsiNTBlEQAREQAREQAREQAREoMEEZCE0uHGlmgiIgAiIgAiIgAiIgAjkJiALITcyZRABERABERABERABERCBBhOQhdDgxpVqIiACIiACIiACIiACIpCbgCyE3MiUQQREQAREQAREQAREQAQaTEAWQoMbV6qJgAiIgAiIgAiIgAiIQG4CshByI1MGERABERABERABERABEWgwAVkIDW5cqSYCIiACIiACIiACIiACuQnIQsiNTBlEQAREQAREQAREQAREoMEEZCE0uHGlmgiIgAiIgAiIgAiIgAjkJiALITcyZRABERABERABERABERCBBhOQhdDgxpVqIiACIiACIiACIiACIpCbgCyE3MiUQQREQAREQAREQAREQAQaTEAWQoMbV6qJgAiIgAiIgAiIgAiIQG4CshByI1MGERABERABERABERABEWgwAVkIDW5cqSYCIiACIiACIiACIiACuQnIQsiNTBlEQAREQAREQAREQAREoMEEZCE0uHGlmgiIgAiIgAiIgAiIgAjkJiALITcyZRABERABERABERABERCBBhOQhdDgxpVqIiACIiACIiACIiACIpCbgCyE3MiUQQREQAREQAREQAREQAQaTEAWQoMbV6qJgAiIgAiIgAiIgAiIQG4CshByI1MGERABERABERABERABEWgwAVkIDW5cqSYCIiACIiACIiACIiACuQnIQsiNTBlEQAREQAREQAREQAREoMEEZCE0uHGlmgiIgAiIgAiIgAiIgAjkJiALITcyZRABERABERABERABERCBBhOQhdDgxpVqIiACIiACIiACIiACIpCbgCyE3MiUQQREQAREQAREQAREQAQaTEAWQoMbV6qJgAiIgAiIgAiIgAiIQG4CshByI1MGERABERABERABERABEWgwAVkIDW5cqSYCIiACIiACIiACIiACuQnIQsiNTBlEQAREQAREQAREQAREoMEEZCE0uHGlmgiIgAiIgAiIgAiIgAjkJjBi3rx5uTMpgwiIwPARePbZZ8eOHcvP4VNdGneFgN4+XcGqQkVABESgCgLaQ6iCosoQgSEgcNJJJ8k8GIJ2looiIAIiIAIi8GfaQ1AnEAERiCIwceLE8847b8yYMZMmTUrJwD7DggsuGFWiEomACIiACIiACNSSgCyEWjaLhBKB+hEwC2HChAkzZsyon3SSSAREQAREQAREoDIC8jKqDKUKEgEREAEREAEREAEREIEGEJCF0IBGlAoiIAIiIAIiIAIiIAIiUBkBWQiVoVRBIiACIiACIiACIiACItAAArIQGtCIUkEEREAEREAEREAEREAEKiMgC6EylCpIBERABERABERABERABBpAQBZCAxpRKoiACIiACIiACIiACIhAZQRkIVSGUgWJgAiIgAiIgAiIgAiIQAMIyEJoQCNKBREYeAKnnXbaiBEjVlhhhbPOOuuDDz6YOnUqf/IZNWoU/+KbgddQCoiACIiACIjA4BCQhTA4bSVJRaChBC6++OJDDz0U5V544YVp06YttNBCxx133LLLLrvyyivPmTPnhz/8IT8bqrrUEgEREAEREIE6EpCFUMdWkUwiMFQEsArQ99577+Xnm2++yU8shCeffJJvllxySb6ZMmXKUAGRsiIgAiIgAiLQXwKyEPrLX7WLwLATePrpp9k62H777Z977jljse+++2ISLPjx5/XXX+cb7SEMey+R/iIgAiIgAr0lIAuht7xVmwiIwP8lsNhii02YMOGEE074xS9+wX+WW265Qw45xJI89dRT/Mkve+65p7CJgAiIgAiIgAj0jIAshJ6hVkUiIAJtCHz+85+fMWPGaqutdtlll/Hvl156aY011rB0N910E3/yy7hx48ROBERABERABESgZwRkIfQMtSoSARHoSABfo9btggsuuIAMHEXYfPPNX331VUU0UgcSAREQAREQgd4QkIXQG86qRQREII3AAw88YNsFW2+9taV75JFHPvroI35ZZ511nnjiiZEjR37/+98XRBEQAREQAREQgR4QkIXQA8iqQgREIIOAH0JYf/31Leldd91lNsPEiRO5JIFf2EkQRxEQAREQAREQgR4QkIXQA8iqQgSaQODFF19Ejddee60byvghhC996UtW/vzzz8/PFVdc8ac//Sn/5ZctttiiG1WrTBEQAREQAREQgQQBWQjqEiIgAlEEttxyS9JtsskmUanzJ+K8wTHHHOP5dt999+WXX/6tt966+uqrMQ/uvPNOgp/mL1U5REAEREAEREAEchOQhZAbmTKIwHAS4KpjFLeflX8efvjhSy+99Nhjj/WSiXH0/PPPE86Ie9OeeeaZZZZZpvJKVaAIiIAIiIAIiEBbArIQ1DFEQAT6T4AIp6NHj26VY8OPP/2XTxKIgAiIgAiIwDARkIUwTK0tXUVABERABERABERABEQgi4AshCxC+r8IiIAIiIAIiIAIiIAIDBMBWQjD1NrSVQREQAREQAREQAREQASyCMhCyCKk/4uACIiACIiACIiACIjAMBEYMW/evGHSV7qKgAgUJHDGGWdMnjx5+vTpBx10UHoRF198MTegvf/++wVrKpHtpJNO8hsVShSjrCIgAiIgAiIw1ARkIQx180t5EYgnEG8hrLTSSs8++2x8yRWm5EaFMGRqhSWrKBEQAREQAREYHgKyEIanraWpCJQiEG8h7LfffhdddFFrZVyKvPDCC+cV4vrrr/csXJ327rvvvv766ymF/Pa3v+Uuhby1KH2FBD744IOHHnqIOy4oky2dtddeWy1SIV4VJQIiIAI9ICALoQeQVYUINIFAvIXwu9/9jknhCy+8EKrNlcnf/va3p0yZUobFq6++ivPSY4899sgjj+DIRBVvvvlmokCMk3322adMLcpbhgBNM27cuETrX3755VySXaZY5RUBERABEeglAZ1U7iVt1SUCQ0GABWNmhIsvvnioLQv/p59++ty5c8sg4GZl1qSZgOJKRFFPPvkk9gAbC5gfXuy0adPKVKG8ZQhgwq255pqYB9tss014XmXChAlXXHFFmZKVVwREQAREoJcEZCH0krbqEoFhIbDeeut973vfCyfuaM56//jx49lhqIoCpgjbBY8++igHlL1MXFzuuOOOqqpQObkInHvuuaRffvnlZ8yYwaF2PL4wFfjmvffe22OPPWiaXKUpsQiIgAiIQL8IyELoF3nVKwINJ3DggQeuscYaCSVffvlllpOr1XzBBRfETnjllVc22GADSmaz4rTT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yMxeFvGggElvrBvGMnoprwiIgAgMNoHC7rDKKAIiMHAEOICLJ314LpnffQjjlGqKRt04qRzeUMYFat3j6fF2sBMwxlorCk8q+7lecu2xxx7OBxcjy2i3qj388MN+vVr3JC9WMncnu9jEWfJCsATM/uGn35VmthME/DRI5qV1JHaLK9cp7YQ6iThIiRPVxXRXLhEQAREQgfIEtIcw2AaepBeBXATw+L/yyitvuukmzxU6GtkFaj374FiCi4tF5MRumTZtWveqRnFWqSkf3xgOJbdWhNuV38zwk5/8BKd8vHHGjRvHqWJPbFc0IDYyM+1+7733bM3b7mKr1SeMVnTDDTfgu4U63M33wx/+kF0CzAPOLh9++OHIzHYBTHAuYrJu3ketp7FbVcNCMC8j7IRcp7QTRWFdhCGkynvc1aoVJIwIiIAIDDCB8kaGShABERhcAqziu6MRE8eU9eBq9xBsN8OGTvYxMhetyxO224X5cL1X29LCDQ2cZ8KZK7mAw6aBZWR+DCgzDPieuKLlxauwBGRLvJNQx5f8bfcgBG7q+K1nhEvKFMYTc8dcZuL0BN4NEKzt9k7J8pVdBERABESgAAHtIQywdSfRRaA8AaIGLbroolYOq8sciihfZmYJLL3vtttudjYaR52rrroqjC6amb1Ygu22285soZtvvrltCaDABsA/B2cbThfgrM/0d+WVV3Y4dgiBj4UGstPVSI4HfzGRupQrPITADdBohDosz2MkMLPHnpkyZUoIHHX4k2unTZ4tttgiUzBjiBHloZAys3RKwKYW7m0YLQiWiLlUuExlFAEREAERKElAFkJJgMouAgNPgMt9XYfeOBqdcsopDzzwgFWKP0/lsU3bNskaa6wx33z/M+LhS4PLTds0OMwce+yxzz//vC23/PjHP/bZNocQwlm1e2px+/Ls2bPHjx+febq3Zx0lvAmB6T4amTrPPPMMt5u1nYXfdtttJh7mxJw5c4jAmxJcyKMhceS9vHUEVcwMjBaZBz3rIapIBERABDIJyELIRKQEItBwAsTrdI+aHkQ0Ov744wmhY97wt99+e2/MA2tCO+rA5sC1114b06hMhd0FKxERyAIc8V/Cnp588slXX3114tBtTPldSpNyE0KnGjmowL/oBhy94LgCV6qlqHP++edbNCTMiR5s/nSJkooVAREQARFIISALQd1DBIadAHN0txCYuHf1CjBim5p5QKhNJppdim369Mef1tCZOBpZ9CF8WmKW/O+//34inJIeP34igYYdxc7ULrHEEmjEzQPcOdBLUyely6KXHf7mk7gJISWXXZ6wwAILcBiD34ni2mlFn/JxCiIxvlh+z92wP0LSXwREQAQaR0AWQuOaVAqJQH4CoaMR8enzFxCVw64+sN0DAun4EdWozNGJsA3GjBnDmYHWA7s4EWEbWLhPtjI6FckkGDOJfQZsGEuDHz/WAt+41WFRUF944QWCGmF1nHTSSdECdishsiEh1x34DXrcJM03nVyqQjksOuq7777LpQSoc9lll3WS8sgjj6R8GBLxSRsI3WpLlSsCIiACfSdQ4HSzsoiACDSMACvHvo3AWnjbiEYlYxl5pKAwEn83MLL+zbhKbNNOhZtlwiYGPk5t06QsvV9zzTWWBUTMqu3sL6p1Q5G8ZXLXQdsXSgoKrwJ1sJ1MHQ/Z1CoAxOBGLTDMK57Si4AIiIAIDBAB7SH03UaTACLQfwKhoxFHVCt3NAqvPvBI/N1Qm/hI//Zv/0bJLO13Kn/mzJksk3OLM2nansftdLMvuVZddVUr1s4029nfmvgXPffccwmVcY5affXVuXQiEzXqcFbY1OFId9v0sNp///3hhn8RAYgyy1QCERABERCBwSUgC2Fw206Si0CVBLgzy4ur1tGICfekSZP8ZjQml13yTsGL6YADDmAKy8wYd6ZOdJgNYwIx3WfKSxSdVnsAz5zWZR5O7pKrzvF2ODCd2Px5/PHHOUJNRKbyHQVKm2yyCY0It3vuuadLLVheTpUgAiIgAiJQCQFZCJVgVCEiMPAEOMXrjkb4mcQc5I3ROXH1wVlnndWNySW1cK7ADjkg1TrrrJN+0S+r/kz3cb/505/+1GnHIKEdYtdkr6ATdiRMaM2fVZk02B4cusANqWfRaWN6l9KIgAiIgAh0iYAshC6BVbEiMGAE8C0hlI0JXaGjkV99wNpzNyaXzO/ZOiDQEKcCzDxgHn/IIYdk0ifZ3Llzie6abktkljMkCcBlNyrU3EwakuaQmiIgAiLQbQKyELpNWOWLwMAQ+OY3v+myVuJoFF59cOGFF1Y4uSRgEUcOJk+evNBCC+21114WrNPNmw022GBgoEtQERABERABEagfgREsC9VPKkkkAiLQBwKExfza175ml2HxwbEkXF/n8mBuXCaiUaTxEMY23XbbbQvHNiUE5+9//3vk+dWvfvXOO+/gAcVdBAQzfeqpp9oyYjOBM8R9wKcqRUAEREAERKApBGQhNKUlpYcIVEFghRVWwN3cSrrxxhvDaX0uC8HNgyqEylEGRylmz57dKRpPjoKUVAREQAREQASGmIC8jIa48aW6CLQQqMTRyMJi2qmAHn9Gjhwp86DHzFWdCIiACIhA8wjIQmhem0ojEShOIIxohKN/sYhG48ePt9imvf+kXIPQe2FUowiIgAiIgAgMKAF5GQ1ow0lsEegWgU6ORvFeRnfffbddW5b4cKUAwXDyyr3SSitxHDkyF9chKzZRJCslEwEREAEREIFOBGQhqG+IgAj8HwIEIOKwr33FOQROI9jv8RaCgIqACIiACIiACAw0AXkZDXTzSXgRqJ5AJY5G1YulEkVABERABERABHpFQBZCr0irHhEYEALh1WmIzDVnAyK4xBQBERABERABEaiGgCyEajiqFBFoEoFKIho1CYh0EQEREAEREIGhIiALYaiaW8qKQBQBHI2WW245S1o4olHbmj744IMoCaITEVm18jKjK1dCERABERABEWgmAVkIzWxXaSUCZQjgaDTffJ8MDpU4GhE4db311iMq0TXXXFNGNs+LbXDYYYdxAQI/KylQhYiACIiACIiACBgBWQjqCSIgAm0IVO5odNZZZz344IPUdOWVV5Yhjrmy0047jRgxAtvg1FNPpag//elPZQpUXhEQAREQAREQgQQBWQjqEiIgAm0IVO5oNGrUqKWWWoqaDj/88DLE33jjjeuvv96doMoUpbwiIAIiIAIiIAJtCchCUMcQARFoQ6ByR6PRo0ezh8ClafgalSG+2WabPfXUU7feeutFF11UphzlFQEREAEREAER6ERAFoL6hgiIQHsCe++9t//jiiuuKI9pmWWWWXDBBUuWw5XJX/r4s/DCC5csStlFQAREQAREQAS0h6A+IAIikIPAuHHjFl98cctw2WWXvfPOOzkyB0nnzp27ww474BfEbc2cVy5WiHKJgAiIgAiIgAj0jMCIefPm9awyVSQCIjBYBJjWv/zyy6HMEyZMmDFjRrwWd9xxx5gxY0i/yCKLvP3227gG7bPPPpb96aefji+HIEhsQYTpr7322p133plv8ooUX6lSioAIiIAIiMBwEpCX0XC2u7QWgSgCe+65Z1S6zomOPvpo/nn55ZdjHvDLmWeeaWkJSfTlPJ8NNthA+w8l20LZRUAEREAERCCSgPYQIkEpmQgMI4FHHnlkq622evPNN135XAv2ZF9zzTU33njjE088kRCllHPQQQdNnz6d0rjmbNq0afGT/pVWWumII44I20B7CMPYI6WzCIiACIhATwjIQugJZlUiAgNLIOFolNdC+MEPfnDUUUddcsklZ5xxxpJLLjlr1qwNN9ywEhiyECrBqEJEQAREQAREoJWAvIzUK0RABNIIlHE0ImTqddddx/I/5gF1zD///GuttZZwi4AIiIAIiIAI1JyA9hBq3kASTwT6TCDhaJRrD8FEv/jii/fdd19+Oe644yZNmsQvRCyVl1Gf21XVi4AIiIAIiEBnArIQ1DtEQATSCDCVJ46QpyhgIXCb8pw5c/BWuvfee8ePH//6668///zzdkQhHv2KK65I1FRMC88iL6N4ekopAiIgAiIgArkIyMsoFy4lFoGhI8AdZxwvLqz2q6++agedl1566bfeegtTgdMI/IkDElcjx3/uvPPO0DwoLI8yioAIiIAIiIAIZBLQHkImIiUQgWEnQGRSi0QEiLx7CEQr+upXv4qdgKMRs3xuV7j99ttHjx5dmClFcfqZ7I8++uiDDz7IL9yTsPbaay+22GJ77LFHmZILi6SMIiACIiACItAwArIQGtagUkcEqicQOhrltRCQZurUqeecc44ZGBxFmDJlShkRuWeNexTalkAc1TLbHWWkUl4REAEREAERaBIBWQhNak3pIgLdIjB58mSLR1TAQsDAeOihh37961+PHTs2cS9yAXEp7ZVXXmmbkW0EeSIVQKosIiACIiACIpAgIAtBXUIERCCbgDsaFbAQsktXChEQAREQAREQgToR0EnlOrWGZBGBuhLgHoPwZuW6iim5REAEREAEREAEKiDwqWnTplVQjIoQARFoNIFPf/rTiyyyCHGNDjjggOWXX77Ruko5ERABERABERh2AvIyGvYeIP1FQAREQAREQAREQAREICQgLyP1BxEQAREQAREQAREQAREQgU8IyEJQbxABERABERABERABERABEZCFoD4gAiIgAiIgAiIgAiIgAiLQjoD2ENQvREAEREAEREAEREAEREAEtIegPiACIiACIiACIiACIiACIqA9BPUBERABERABERABERABERCBdALyMlIPEQEREAEREAEREAEREAER+ITA/wMMW0V1X8DqAAAAAABJRU5ErkJggg==)

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



## Prediction:

After figuring out the distances between the points, we will use the distances to find the '_k_' nearest neighbours of the given point and then, based on the classes of these 'neighbours', make the prediction on the class of the incoming data. 

This is quite straight-forward: First calculate the distance between the incoming point and all the points in the training set. Then select a subset of size _k_ from those points and find the probability of the incoming point being in each class. The class with the most probability will be selected as the predicted class.

We get to choose the value '_k_'. There are many rules of thumb to do this, but most often the value of '_k_' is chosen after trail and error. In this case, I am setting the default value of _k_ to 3.

In `CustomKNN` class:

	def predict(self, training_data, to_predict, k = 3):

                if len(training_data) >= k:

                        print("K cannot be smaller than the total voting groups(ie. number of training data points)")

                        return

                distributions = []

                for group in training_data:

                        for features in training_data[group]:

                                euclidean_distance = np.linalg.norm(np.array(features)- np.array(to_predict))

                                distributions.append([euclidean_distance, group])

                        results = [i[1] for i in sorted(distributions)[:k]]

                        result = Counter(results).most_common(1)[0][0]

                        confidence = Counter(results).most_common(1)[0][1]/k

                return result, confidence

## Testing/Evaluation :

Now that we have finished up the algorithm, it's time to test how well it performs. Since, for this dataset, k-NN is a bindary classifier, I will be using [classification accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification) to evaluate the algorithm. 

>The accuracy is the proportion of true results (both true positives and true negatives) among the total number of cases examined.

In the class `CustomKNN`:

	def test(self, test_set, training_set):
		for group in test_set:
			for data in test_set[group]:
				
				#Making the predictions

				predicted_class,confidence = self.predict(training_set, data, k =3)
				
				if predicted_class == group: #Got it right
					self.accurate_predictions += 1
				else:
					print("Wrong classification with confidence " + str(confidence * 100) + " and class " + str(predicted_class))
				
				self.total_predictions += 1

		self.accuracy = 100*(self.accurate_predictions/self.total_predictions)
		print("	nAcurracy :", str(self.accuracy) + "%")

Along with the classification accuracy, above function will also print out the wrongly classified elements with the probablities.

After a few runs, the best value for accuracy that I got:

	>>>Accuracy: 88.75%

## Comparing accuracy of Custom k-NN with sci-kit k-NN:

Now, let's compare our implmentation with the sci-kit learn implementation. This is just for demonstration purposes only. If I was using k-NN algorithm in a production environment, I would _definitely_ use the library function and so should you. 

Here is the code with the sci-kit learn k-NN for the same dataset:

	from sklearn import preprocessing, cross_validation, neighbors
	import pandas as pd
	import numpy as np


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



	df = pd.read_csv(r".\data\chronic_kidney_disease.csv") #Reading from the data file
	mod_data(df)

	X = np.array(df.drop(['class'], 1))
	y = np.array(df['class'])

	#Use 25% of the data as test
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25)

	clf = neighbors.KNeighborsClassifier()
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)

	print("Accuracy: " + str(accuracy*100) + "%") 

After many runs, the best accuracy that came out of the library algorithm:
	
	>>>Accuracy: 86.75%

However, in most runs, the accuracy hovered around the value of `81.25%`. I was quite surprised at the result myself. I am not fully sure as to why the custom implementation  _**slightly**_ out-performed the sci-kit learn implementation. This probably has something to do with the fact that I have used sci-kit k-NN as it is - without any customization whatsoever. The _k_ value and _distance metric_ themselves play an important role in accuracy. It is also possible that sci-kit implementation refrains from going through the entire dataset to improve the running time.

You can find the entire code on GitHub, [here](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/K%20Nearest%20Neighbours.py).

That's it for now. If you have any comments, please leave them below.

<br /><br />