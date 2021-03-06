---
layout: post
title:  "Implementing K Nearest Neighbours in Parallel from scratch"
date:   2017-02-10 03:34:56 +0530
description:   One of the prime drawbacks of k-NN is its efficiency. The brute force version of k-NN that was written previously is highly parallelizable. The computation of distances between the attributes is independent of one another. Also, the classification of incoming data points is independent of one another and can be easily accomplished in parallel. In this post I will implement the algorithm from scratch in Python in parallel.
categories: Machine-Learning Parallel-Processing

---


K Nearest Neighbours is one of the most commonly implemented Machine Learning classification algorithms. In my previous blog post, [I had implemented the algorithm from scratch in Python](/articles/machine-learning/2017/01/13/implementing-k-nearest-neighbours-from-scratch-in-python.html). If you are not very familiar with the algorithm or it's implementation, do check my previous post.

One of the prime drawbacks of the k-NN algorithm is it's efficiency. Being a supervised **[lazy learning](https://en.wikipedia.org/wiki/Lazy_learning)** algorithm, the k-NN waits till the end to compute. On top of this, due to its [non-parametric](https://en.wikipedia.org/wiki/Non-parametric_statistics) 'nature', the k-NN considers the entire dataset as it's model.

So, the algorithms works on the _entire_ dataset at the _very end_ for _each prediction_. This considerably slows down the performace of k-NN and for larger datasets, it is excruciatingly difficult to apply k-NN due to its inability to scale.

Now, let's see if we can speed up our [previous serial implementation](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/K%20Nearest%20Neighbours.py) by applying the concepts of parallel programming.

<script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<!-- Image AD -->
<ins class="adsbygoogle"
     style="display:inline-block;width:728px;height:90px"
     data-ad-client="ca-pub-3120660330925914"
     data-ad-slot="4462066103"></ins>
<script>
(adsbygoogle = window.adsbygoogle || []).push({});
</script>

## Proposal

The brute force version of k-NN that was written previously is [highly parallelizable](http://web.cs.ucdavis.edu/~amenta/pubs/bfknn.pdf). This is due to the fact the computation of the distances between the data points is completely _independent_ of one another. Furthermore, if there are _n_ points in the test set, all of the computation regarding the classification of these _n_ points is independent of one another and can be easily accomplished in parallel. This allows for partitioning the computation work with least synchronization effort. The distance computations can be calculated seperately and then brought together or the dataset itself can be split up into multiple factions to be run in parallel.

That is, the brute force k-NN has high potential to work faster under [data parallelism](https://en.wikipedia.org/wiki/Data_parallelism):

> Data parallelism is a form of parallelization across multiple processors in parallel computing environments. It focuses on distributing the data across different nodes, which operate on the data in parallel.

The idea is to split the data amongst different processors and then combine them later for procuring final results. The ideal scenario is the case where the processors do not  have to interact with each other, this is the case with brute-force k-NN.

## Implementation

### Parallel processing in Python

Parallel programming in Python isn't as straight foward as it is in mainstream languages such as Java or C/C++. This is due to the fact that the default python interpreter(Cpython) was designed with simplicity in mind and with the notion that multithreading is [tricky and dangerous](http://www.softpanorama.org/People/Ousterhout/Threads/index.shtml).  The python interpreter has a thread-safe mechanism, the **Global interpreter lock**.

>Global interpreter lock (GIL) is a mechanism used in computer language interpreters to synchronize the execution of threads so that only one native thread can execute at a time. An interpreter that uses GIL always allows exactly one thread to execute at a time, even if run on a multi-core processor.

Python is restricted to a single OS thread; therefore, it cannot make use of the multiple cores and processors available on modern hardware. Hence, using threads for parallel processing will _not_ work.

As a result, I am using the invaluable **[multiprocessing](http://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#multiprocessing)** module in Python for parallel processing. [I have previously written about working with the multiprocessing library](/articles/parallel-processing/2017/01/25/parallel-programming-in-python-with-ease.html), do have a look if you are unsure on the working of the module.

### Parallelizable regions

As stated before, there are two options that could be implemented whilst parallelizing brute force k-NN. The first is to parallelize the distance finding part within _each_ incoming datapoint, the second is to divide the test data and process on it in parallel. I am going to go ahead and implement the latter for the following reasons:

1. The distance finding function will be called the most number of times. From as theorotical standpoint, parallelizing this should yield the maximum benefit of parallel prcessing. However, for all practical purposes, we cannot ignore the overheads. The overheads of *process* creation for *each and every distance calculation* will surpass any benefit of parallelization, definately slowing down the program. There are less overheads when the data itself is divided and fed into different sub-processes.

2. The code is much easier to write and is less cluttered for data parallelism.

The implementation revolves around applying data parallelism to the distance finding part of the algorithm. In the parallelizable part, if there are _n_ data points on whom the distance algorithm is to be applied, we will divide the data intp _p_ datasets of size _n/p_ and then let each processor work _independently_ on a data of size _n/p_. In the serial part of the algorithm, we will be dividing the dataset, setting up the code to run in parallel, collect the output from the paralleized region and then continue with the k-NN algorithm.

		for group in test_set:
			for data in test_set[group]:
				predicted_class,confidence = self.predict(training_set, data, k =3)
				if predicted_class == group:
					self.accurate_predictions += 1
				else:
					print("Wrong classification with confidence " + str(confidence * 100) + " and class " + str(predicted_class))
				self.total_predictions += 1

The above for loop is the bottleneck of the k-NN algorithm. We need to parallelize the above for loop. Since we are going to be applying data parallelism, we needn't worry about the actual functions used; we will uilize the same functions again. Applying data parallelism will not affect the actual results in any way.


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


### Parallelizing the code

The parallel version retains *nearly all* of the serial regions of the serial implementation of the algorithm. The changes that lead to parallelism are made to the `test` function that accepts new data points and makes predictions on their class.

Now, let's parallelize the main loop using `multiprocessing pools`:

		pool = mp.Pool(processes= 8)
		arr = {}

		for group in test_set:
			arr[group] =  pool.starmap(self.predict, zip(repeat(training_set), test_set[group], repeat(3)))

The incoming data points will be split and fed into 8 sub-processes that can run in parallel.


In the parallel code, I will calculate the accuracy of the algorithm in the function `test` seperately, in order to avoid race conditions and sharing of variables amongst sub-processes.

First I will change the predict function a bit so that it includes the incoming data point in the output. This is essential since without it, there's no way of figuring out which prediction corresponds to which data point. Due to parallel execution, the order of the output is non-deterministic.

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

		return result, to_predict


Now let's write the accuracy calculation part of the code based on the new output provided by the predict function:

		#Calculating Accuracy
		for group in test_set:
			for data in test_set[group]:
				for i in arr[group]:
					if data == i[1]:
						self.total_predictions += 1
						if group == i[0]:
							self.accurate_predictions+=1

		self.accuracy = 100*(self.accurate_predictions/self.total_predictions)
		print("\nAcurracy :", str(self.accuracy) + "%")


The complete `test` function after parallelization:

	def test(self, test_set, training_set):
		pool = mp.Pool(processes= 8)

		arr = {}
		s = time.clock()
		for group in test_set:
			arr[group] =  pool.starmap(self.predict, zip(repeat(training_set), test_set[group], repeat(3)))
		e = time.clock()

		#Calculating Accuracy
		for group in test_set:
			for data in test_set[group]:
				for i in arr[group]:
					if data == i[1]:
						self.total_predictions += 1
						if group == i[0]:
							self.accurate_predictions+=1

		self.accuracy = 100*(self.accurate_predictions/self.total_predictions)
		print("\nAcurracy :", str(self.accuracy) + "%")

## "Correctness"

Now that we've parallelized the program, we have to check if the parallel program produces the same required result as it's serial counter part.

After numerous parallel runs of the algorithm, the best value of accuracy that it produced:

	>>> Accuracy: 90.75

compared to 91.25 from serial: __close enough.__

The average parallel accuracy is also very close to the average serial accuracy with values of 88.4 and 88.6 respectively.

After running the code on the same dataset *without* shuffling the data, both the programs produce the same results indicating that the parallel program is equivalent to the serial implementation.


<div class = "announcement" id = "announcement">
	<span>Still have questions? Find me on <a href='https://www.codementor.io/madhugnadig' target ="_blank" > Codementor </a></span>
</div>

## Speedup

Once the parallelization of a task is complete, it is important to evaluate the speed and efficiency of the new program, for parallelism is pointless without faster execution.

> Speedup (Sp) is defined as the ratio of runtime for a sequential algorithm (T1) to runtime for a parallel algorithm with p processors (Tp). That is, Sp = T1 / Tp. Ideal speedup results when Sp = p. Speedup is formally derived from Amdahl’s law, which considers the portion of a program that is serial vs. the portion that is parallel when calculating speedup.

The size of the input corresponds to the number of data points in the input file. Each data point is represented in *m* dimensional space, where *m* is the number of attributes in each data points. So, for *N* data points, the acual size of the input is N * m.


Here are the results for serial and parallel after many runs:

	| Number of Data points | Serial      | Parallel    | Speedup|
	|-----------------------|-------------|-------------|--------|
	| 400   	        | 1.250104    | 2.7561666   | 0.453  |
	| 800  	               | 3.664904    | 2.808934   | 1.304  |
	| 1600 	               | 15.434006   | 6.263597   | 2.464  |
	| 3200	               | 66.626987   | 18.958429  | 3.5143 |
	| 6400   	        | 244.1179921 | 64.78382    | 3.768  |


The advantages of parallel processing are apparent just as the data size increases a little bit to 800 data points; with a speed up of 1.3 (30% faster exec time). The speed up of __3.768__ is perhaps the best that we achieve since the program ran on a quad-core processor, for which the upper limit for speed up is 4 (ignoring the overheads).

The graph representing the speedup:


<style>

.axis path,
.axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

.bar {
  fill: #E34B48;
}

.bar:hover {
  fill: #270738 ;
}

.x.axis path {
  display: none;
}

.d3-tip {
  line-height: 1;
  font-weight: bold;
  padding: 12px;
  background: rgba(0, 0, 0, 0.8);
  color: #fff;
  border-radius: 2px;
}

/* Creates a small triangle extender for the tooltip */
.d3-tip:after {
  box-sizing: border-box;
  display: inline;
  font-size: 10px;
  width: 100%;
  line-height: 1;
  color: rgba(0, 0, 0, 0.8);
  content: "\25BC";
  position: absolute;
  text-align: center;
}

/* Style northward tooltips differently */
.d3-tip.n:after {
  margin: -1px 0 0 0;
  top: 100%;
  left: 0;
}
</style>


<div id = "graph" class = "graph">

</div>


<script src="http://d3js.org/d3.v3.min.js"></script>
<script src="http://labratrevenge.com/d3-tip/javascripts/d3.tip.v0.6.3.js"></script>
<script>

data = [
	{"letter":"400","frequency":0.453},
	{"letter":"800","frequency":1.304},
	{"letter":"1600","frequency":2.646},
	{"letter":"3200","frequency":3.5143},
	{"letter":"6400","frequency":3.768}
];

ww = document.getElementById("graph").offsetWidth;
hh = document.body.clientHeight/1.333;

console.log(ww);

var margin = {top: 40, right: 20, bottom: 30, left: 40},
    width = ww - margin.left - margin.right,
    height = hh - margin.top - margin.bottom;


var x = d3.scale.ordinal()
    .rangeRoundBands([0, width], .1);

var y = d3.scale.linear()
    .range([height, 0]);

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left");

var tip = d3.tip()
  .attr('class', 'd3-tip')
  .offset([-10, 0])
  .html(function(d) {
    return "<strong>Speedup:</strong> <span style='color:red'>" + d.frequency + "</span>";
  });

var svg = d3.select("#graph").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

svg.call(tip);


  x.domain(data.map(function(d) { return d.letter; }));
  y.domain([0, d3.max(data, function(d) { return d.frequency; })]);

  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Speedup");

  svg.selectAll(".bar")
      .data(data)
    .enter().append("rect")
      .attr("class", "bar")
      .attr("x", function(d) { return x(d.letter); })
      .attr("width", x.rangeBand())
      .attr("y", function(d) { return y(d.frequency); })
      .attr("height", function(d) { return height - y(d.frequency); })
      .on('mouseover', tip.show)
      .on('mouseout', tip.hide);



function type(d) {
  d.frequency = +d.frequency;
  return d;
}

</script>


You can find the entire code related to the parallel implementation, [here](https://github.com/madhug-nadig/Parallel-Processing-Nadig/blob/master/K%20Nearest%20Neighbours%20-%20In%20Parallel.py). Serial implementation can be found [here](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/K%20Nearest%20Neighbours.py).

That's it for now, if you have any comments, please leave them below.

<br /><br />
