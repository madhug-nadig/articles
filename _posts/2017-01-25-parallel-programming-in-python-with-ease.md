---
layout: post
title:  "Parallel Programming in Python with ease"
date:   2017-01-25 12:34:56 +0530
description:   Parallel programming in Python is a bit tricky as compared to languages such as C/C++ and Java. Python is restricted to a single OS thread; therefore, it cannot make use of the multiple cores and processors available on modern hardware. In this post I will use the `multiprocessing` library to easily create and coordinate multiple Python processes and run code in parallel.
categories: Parallel-Processing

---

<style>

#accouncement{
	width:80%;
	border:5px solid #882d2b;
	margin:5px;
	padding:5px;
	text-align:center;
}

#announcement span{
	color: #3398c7;
	text-align:center;
}

#announcement span a{
	text-decoration:none;
	background-image: linear-gradient(to top,#3398c7,#c0e4e4);
	color:#fff;
	font-weight: 700;
	border-radius:25%;
	padding: 15px;
}

#announcement a:hover{
	background-color:#000;
}

</style>

Due to the recent [slowdown/possible demise of Moore's law](https://www.technologyreview.com/s/601102/intel-puts-the-brakes-on-moores-law/), parallel programming has gained widespread prominence as the paradigm of the future. Since more than a decade, due to the anticipation of the end of the Moore's law, CPUs with multiple cores have become the norm. Multicore CPU's have also found their way into _smartphones_ too, with [LG Optimus 2X](https://en.wikipedia.org/wiki/LG_Optimus_2X) being the first phone to have multiple cores, way back in 2010. Just switching to the new processor _may no longer guarantee_ faster performance. With multicore/multiprocessor architectures, it is imperative to write software in way that they could be run in parallel. Most computer programs simply cannot take advantage of performance increases offered by GPUs or multi-core CPUs unless those programs are adquately modified. It is time for developers to take a more active role in improving performance by taking the computer architecture into consideration. 

In this post, I will write about parallel programming in `python`.

Parallel programming in Python is a bit _tricky_ as compared to languages such as C/C++ and Java, where one can write parallel programs by executing multiple threads. Python interpreter was designed with simplicity in mind and with the notion that multithreading is [tricky and dangerous](http://www.softpanorama.org/People/Ousterhout/Threads/index.shtml).  The default python(CPython) interpreter has a thread-safe mechanism, the **Global interpreter lock**. 

>Global interpreter lock (GIL) is a mechanism used in computer language interpreters to synchronize the execution of threads so that only one native thread can execute at a time. An interpreter that uses GIL always allows exactly one thread to execute at a time, even if run on a multi-core processor. 

**Python is restricted to a single OS thread**; therefore, it cannot make use of the multiple cores and processors available on modern hardware. So _throwing some threads_ into the program will not result in faster performance, since the threads essentially run in serial. The actual overhead of thread creation, synchronization and termination will actually slow down the program. This problem exists only in CPython, not in Jython or IronPython.

If you interested to know more about Cpython's GIL, you should read this [article](https://bhargav2017.wordpress.com/2017/04/03/the-python-gil/).


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

 
## The **[multiprocessing](http://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#multiprocessing)** library

Instead of threads, Python programmers should use the `multiprocessing` library to easily create and coordinate multiple Python processes. Each one is scheduled independently on the CPU by the OS. Multiprocessing is an invaluable library for writing parallel programs in python. 

>The multiprocessing module aims at providing a simple API for the use of parallelism
based on processes

>The approach that is based on processes is very popular within the Python users' community as it is an alternative to answering questions on the use of CPU-Bound threads and GIL
present in Python.

From what I gather from the internet, this programming model is easier than parallelism with threads and by far, the most popular soultion to parallel programming in python. Multiprocessing is ideal for **CPU bound tasks**.

The multiprocessing module has been in the Python Standard Library since Python 2.6. 
It is important to understand that in multiprocessing, it's the process level abstraction, not thread level. 

> The multiprocessing package offers both local and remote concurrency, effectively side-stepping the Global Interpreter Lock by using subprocesses instead of threads.

The multiprocessing module only allows message passing paradigm for inter-process.

>The message passing paradigm is based on the lack of synchronizing mechanisms as copies
of data are exchanged among processes.

### Process

In multiprocessing, new processes are spawned by creating a `Process` object and then calling its `start()` method. The processes can be terminated by using the `join()` method.

In a simple example, I will parallel compute a rudimentary calculation. I will compute the square root of the cube of the first 7 positive integers **parallely** using the `Process` class. A new process is spawned for each argument(_definately_ not scalable, but a good enough example), so this case, we will have 7 subprocesses in parallel.
    
    import multiprocessing as mp
    import math
    
    
    def cubes_and_sqare_root(a, order,output):
    	output.put((int(order), math.sqrt(a**3)))
    
    def main():
    	#Using the queue as the message passing paradigm 
    	output = mp.Queue()
    	processes = [mp.Process(target=cubes_and_sqare_root, args=(x, x,output)) for x in range(1,8)]
    
    	for process in processes:
    		process.start()
    
    	for process in processes:
    		process.join()
    
    	results = [output.get() for process in processes]
    
    	print(results)

    if __name__ == '__main__':
    	main()

Typically, the order of output cannot be predicted as one subprocess may take longer time than another.

The output:

`>>>[(2, 2.8284271247461903)(4, 8.0)(1, 1.0)(3, 5.196152422706632)(6, 14.696938456699069)(5, 11.180339887498949)(7, 18.520259177452136)]`

Let's print out the details of all the subprocesses involved in the above computation:

    import os
    
	def process_info():
		print('Module:', __name__, '\n')
        print('Parent Process id:', os.getppid(), '\n')
        print('Process id:', os.getpid(), '\n\n')

The output in exact order:

    Module:__mp_main__
    Parent Process id:23524
    Process id:22928
    
    Module:__mp_main__
    Parent Process id:23524
    Process id:24604
    
    Module:__mp_main__
    Parent Process id:23524
    Process id:11584
    
    Module:__mp_main__
    Parent Process id:23524
    Process id:23472
    
    Module:__mp_main__
    Parent Process id:23524
    Process id:9068
    
    Module:__mp_main__
    Parent Process id:23524
    Process id:24636
    
    Module:__mp_main__
    Parent Process id:23524
    Process id:23964

From the above output, it is apparent that 7 subprocesses were spawned by the same parent process, through multiprocessing. You can find the entire sample code related to the above program, [here](https://github.com/madhug-nadig/Parallel-Processing-Nadig/blob/master/Python%20multiprocessing%20example-%20Process.py)

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


### Pool

>The Pool class represents a pool of worker processes. It has methods which allows tasks to be offloaded to the worker processes in a few different ways.

Pools are easier to manage than Processes with Queues and in many cases Processes with feeding queues are just an overkill.

We can create a pool by:

	pool = mp.Pool( number_of_subprograms ) 

The argument passed is the number of subprocesses to be spawned. I will set it to the number of CPUs. 
	
	>>>mp.cpu_count()
	>>>8

To run the function in parallel, we have to use one of the methods in pool: 

-	`apply(func[, args[, kwds]])`: 

>Call func with arguments args and keyword arguments kwds. It blocks until the result is ready. Given this blocks, apply_async() is better suited for performing work in parallel. Additionally, func is only executed in one of the workers of the pool.

-	`apply_async(func[, args[, kwds[, callback[, error_callback]]]])`:

>A variant of the apply() method which returns a result object.

>If callback is specified then it should be a callable which accepts a single argument. When the result becomes ready callback is applied to it, that is unless the call failed, in which case the error_callback is applied instead.


-	`map(func, iterable[, chunksize])`

>A parallel equivalent of the map() built-in function (it supports only one iterable argument though). It blocks until the result is ready.

-	`map_async(func, iterable[, chunksize[, callback[, error_callback]]])`

>A variant of the map() method which returns a result object.

>If callback is specified then it should be a callable which accepts a single argument. When the result becomes ready callback is applied to it, that is unless the call failed, in which case the error_callback is applied instead.

The pool also has other methods, but the above 4 are the most prominent and used ones.

Here is the code to perform same computation that I did with `Process`, this time with `Pool`:

    import multiprocessing as mp
    import math
    import os
    
    
    def process_info():
    	print('Module:' + str(__name__) + '\n')
    	print('Parent Process id:' + str(os.getppid())+ '\n' )
    	print('Process id:' + str(os.getpid())+ '\n\n' )
    
    def cubes_and_sqare_root(a):
    	process_info()
    	return (int(a), math.sqrt(a**3))
    
    def main():
    	pool = mp.Pool(processes= mp.cpu_count())
    	results = [ pool.map(cubes_and_sqare_root, (x for x in range(1,8))) ]
    	print(results)
    
    if __name__ == '__main__':
    	main()
    
As seen above, `Pool` is much simpler than `Process.` You can find the entire sample code related to the above program, [here](https://github.com/madhug-nadig/Parallel-Processing-Nadig/blob/master/Python%20multiprocessing%20example-%20Pools.py)

## Speedup

Once the parallelization of a task is complete, it is important to evaluate the speed and efficiency of the new program. 

>Speedup (Sp) is defined as the ratio of runtime for a sequential algorithm (T1) to runtime for a parallel algorithm with p processors (Tp). That is, Sp = T1 / Tp. Ideal speedup results when Sp = p. Speedup is formally derived from [Amdahl's law](http://en.wikipedia.org/wiki/Amdahl's_law), which considers the portion of a program that is serial vs. the portion that is parallel when calculating speedup.

To calculate the Speedup, let's write the same computation in serial:
    
    import math
    import time
    
    def cubes_and_sqare_root(a):
    	return (int(a), math.sqrt(a**3))
    
    def main():
    	
    	s = time.clock()
    	results = list(map(cubes_and_sqare_root, (x for x in range(1,10000000))))
    	e = time.clock()
    	print(e-s)
    
    if __name__ == '__main__':
    	main()

<div class = "announcement" id = "announcement">
	<span>Still have questions? Find me on <a> Codementor </a></span>
</div>

Here are the results for serial and parallel after many runs:

    | Size of Input | Serial      | Parallel   | Speedup|
    |---------------|-------------|-------------|--------|
    | 100   	| 0.00009639  | 0.1197657   | 0.008  |
    | 1000  	| 0.00107654  | 0.1155729   | 0.009  |
    | 10000 	| 0.010722370 | 0.12766538  | 0.083  |
    | 100000	| 0.102903703 | 0.168076249 | 0.6122 |
    | 1000000   	| 1.035941925 | 0.601318320 | 1.72   |
    | 10000000  	| 10.85937320 | 6.245669530 | 1.73   |

Since the computation is relatively rudimentary, the advantages of parallel processing does not show until the data is large enough. This is due to the fact that the calculation itself isn't CPU intensive. Hence, for lower sizes of input data, the overheads of subprocess level management overtake the advantages of parallel processing. 

Once the size of data increases, the speedup rises rapidly, and then plateaus at a value around 1.72 - 1.73. With this, we can conclude that parallelization works well for CPU intensive tasks with multiprocessing.

Below is a graph depicting the Speedup.




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
<body>


<div id = "graph" class = "graph">

</div>


<script src="http://d3js.org/d3.v3.min.js"></script>
<script src="http://labratrevenge.com/d3-tip/javascripts/d3.tip.v0.6.3.js"></script>
<script>

data = [
	{"letter":"100","frequency":0.008},
	{"letter":"1000","frequency":0.009},
	{"letter":"10e4","frequency":0.083},
	{"letter":"10e5","frequency":0.6122},
	{"letter":"10e6","frequency":1.72},
	{"letter":"10e7","frequency":1.73}
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



That's it for now, if you have any comments, please leave them below.

<br />
<br />