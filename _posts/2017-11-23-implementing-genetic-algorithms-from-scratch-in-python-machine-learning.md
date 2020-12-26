---
layout: post
title: "Implementing Genetic Algorithms from scratch in Python"
date: 2017-11-23 21:15:00 +0530
description: Biologically inspired genetic algorithms find a lot applications in optimization and scheduling software. Genetic algorithms perform optimized search of the solution space by utilizing evolutionary concepts and biological operations such as cross overs, selection and mutation. In this post, I will implement a genetic algorithm from scratch in Python.   
categories: Machine-Learning
---

## What are Genetic Algorithms?

Genetic algorithms are a class of machine learning algorithms which approximate the process of natural selection seen in nature. Genetic algorithms belong to a larger set of [Evolutionary algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm), which take Charles Darwin's evolution theory as the center piece of inspiration. Genetic algorithms are widely used in solving variety of optimization algorithms in many different domains. Biological processes such as mutation, crossover and selection are heavily relied upon as a source of inspiration in the implementation of generic algorithms.  


[Full code here](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch)

In this article, I will implement the algorithm from scratch in python and apply it on an example problem.  

![Genetic algorithms application - evolution]({{site.baseurl}}/images/genetic_evolution.gif)

<span style = "color: #dfdfdf; font-size:0.6em">Image courtesy:Wikipedia</span>  


> Objects in a simulated environment were allowed to evolve into learning swimming - an experiment by Karl Sims at Thinking Machines in the last 1980s.    
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


### How Genetic Algorithms work

In the [previous post, I have explained the concepts of genetic algorithms and how they work with an example](http://madhugnadig.com/articles/machine-learning/2017/11/04/understanding-genetic-algorithms-and-with-an-example-machine-learning.html), it is recommended for the reader to go through the post before and getting familiar with the concept before hands-on implementation.  

# Implementing Genetic Algorithm From Scratch

### Steps in a Genetic algorithm:

1. Define the solution space a sequence of encoded characters.    
2. Define the [fitness function](http://madhugnadig.com/articles/machine-learning/2017/11/04/understanding-genetic-algorithms-and-with-an-example-machine-learning.html#fitness) and [termination condition](http://madhugnadig.com/articles/machine-learning/2017/11/04/understanding-genetic-algorithms-and-with-an-example-machine-learning.html#termination).  
3. Initialize the population.  
4. Select most fit members of the population based on their fitness for reproduction.   
5. Perform reproduction using crossover of 'genes'.  
6. Perform mutation on the members of the new generation.  
7. Repeat steps 4 through 6 until termination condition is met ie. population has converged into an acceptable solution.  

### Pseudocode:

```
START
Encode problem
Define Fitness function and Termination condition
Generate initial population
REPEAT
    Fitness computation
    Selection
    Crossover
    Mutation
UNTIL Selection Criteria is met
STOP
```

## The Problem: Solving Scheduling using Genetic Algorithm:

[Full code here](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch)

This implementation is based on [a notebook from Joo Korstanje](https://jooskorstanje.com/Genetic-Algorithm-from-scratch.html), it is recommended to check that note book out too.

For this example, let's solve a time tabling problem - let's say that we have an IaaS company with a services that runs [Cron](https://en.wikipedia.org/wiki/Cron) jobs at specific times. Let's saw we have around 20 server racks with 12 servers each. The servers all have equal computation power and the rack operates as an individual unit.  

We have the schedule of when the jobs have to run over a period of 5 days and how much compute power they take. A server cannot run more than one job at a time. We will find an optimal schedule for running these tasks by distributing load on our servers.   s

### Handling Data and Encoding:

We have a data set that has a list of scheduled jobs along with the required computation capacity (value of 2 implies we need 2 servers, for example) for each job. The [data set](https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/data/cron_jobs_schedule.csv) has the schema `<job_id, hour, day, capaity>`, We will use `pandas` to read the csv file, in our main:  

```
    # Reading from the data file
    df = pd.read_csv("./data/cron_jobs_schedule.csv")

    dataset = df.astype(int).values.tolist()
```

#### Encoding:

Before we can apply genetic algorithm to our problem, we first have to encode it in a way that can be solve using it. Generally we have to encode the problem, the goal and the data into some form of discrete numeric representation. In this example, I will aggregate the jobs and will create an array of which contains the capacity needed on a per-hour basis for the 5 days. The structure will look something like this:   

```
CRON SCHEDULE DATA STRUCTURE
[
  [0, 2, 2, 2, 4, 4, 0, 8, 6, 6, 2, 2, 4, 6, 8, 4, 10, 6, 6, 0, 8, 4, 8, 6],
  [0, 6, 8, 2, 8, 4, 0, 8, 6, 0, 2, 2, 4, 6, 8, 4, 6, 8, 6, 0, 2, 4, 8, 4],
  [0, 0, 4, 2, 8, 6, 0, 4, 10, 0, 4, 2, 4, 6, 8, 4, 6, 6, 6, 0, 4, 4, 6, 4],
  [6, 0, 8, 6, 8, 4, 0, 8, 6, 0, 4, 6, 6, 8, 8, 4, 6, 6, 6, 0, 8, 4, 8, 4],
  [2, 2, 0, 2, 4, 4, 0, 8, 6, 0, 2, 2, 4, 8, 8, 8, 4, 2, 6, 0, 4, 4, 8, 8]
]

```  

In this above 2D array, the 5 rows represent 5 days and the 24 elements within each row represent 24 hours in a day. So, the value at 1st row and 13th index of `6` implies that, for the hour 13 of the first day, we need a capacity of at least 6 servers in order to be able to run the crob job.  

I will convert out data set into a structure shown above, so in main function:
```
    required_hourly_cron_capacity = [
        [0 for _ in range(24)] for _ in range(5)]

    for record in dataset:
        required_hourly_cron_capacity[record[1]][record[2]] += record[3]
```

We will represent the schedule of the server with a structure shown below:  

```
  RACK SCHEDULE DATA STRUCTURE
      [
        [
          [ 2,  4,  1],
          [ 0,  7, 11],
          [ 6, 16,  7],
          [ 3, 13,  2],
          [10, 12,  7],
          [ 2, 18, 11],
          [12, 22,  7],
          [ 1, 11,  6],
          [ 8,  7,  3],
          [ 5,  7,  9],
          [14, 20,  3],
          [ 6,  7,  4],
          [ 7,  4,  7],
          [13, 22,  9],
          [11,  5,  3],
          [17, 16,  6],
          [15,  1,  8],
          [16, 13,  4],
          [18, 16, 10],
          [ 9,  1, 10],
          [19, 14,  7],
          [20, 16,  2],
          [22, 14, 10],
          [21, 20, 10]
        ],
        ...
        ...
```

The above data structure represents the racks schedule for each day, the 20 elements in first depth of the array represent the 20 racks, the 3 elements per eack represent : `rack_id`, `time when the rack start executing` and `servers used for execution` (max 12 servers)

### Setting up the Class:


Before we move forward, let's create a class for the algorithm.

      class CustomGeneticAlgorithms:
          def __init__(self):
            pass


We have the `CustomGeneticAlgorithm`, that will be our main class for the algorithm.

### Defining Functions:

Now, let's define the functions that go inside the `CustomGeneticAlgorithm` class. For the simplicity of this tutorial, we will not delve into Advanced Genetic Algorithm topics.

### Utility functions:

To support the encoding we performed in the previous step, let's write a handy utility function that converts the rack schedule data structure to that of the cron schedule:  

```
    def server_present(self, server, time):
      server_start_time = server[1]
      server_duration = server[2]
      server_end_time = server_start_time + server_duration
      if (time >= server_start_time) and (time < server_end_time):
          return True
      return False

    def deployed_to_hourlyplanning(self, deployed_hourly_cron_capacity):

      deployed_hourly_cron_capacity_week = []
      for day in deployed_hourly_cron_capacity:

          deployed_hourly_cron_capacity_day = []
          for server in day:

              server_present_hour = []
              for time in range(0, 24):

                  server_present_hour.append(
                      self.server_present(server, time))

              deployed_hourly_cron_capacity_day.append(server_present_hour)

          deployed_hourly_cron_capacity_week.append(
              deployed_hourly_cron_capacity_day)

      deployed_hourly_cron_capacity_week = np.array(
          deployed_hourly_cron_capacity_week).sum(axis=1)
      return deployed_hourly_cron_capacity_week
```

### Core Functions:

For our simple implementation, we have the following core functions for the algorithm - `generate_initial_population`, `calculate_fitness`, `crossover`, `mutate_gen`, `select_best` and `run`.  

### Generating Initial Population:

Population Initialization is the first step in the Genetic Algorithm Process. Population is a subset of solutions in the current generation. Population P can also be defined as a set of chromosomes. The initial population P(0), which is the first generation is usually created randomly. There are many ways to generate the initial population and having heuristics to nudge the algorithm in finding a solution faster can lead to efficiency gains. For our implementation, we will generate the population randomly.  

First, a function that outputs a random solution in the form of the rack schedule data structure:  

```
  def generate_random_plan(self, n_days, n_racks):
      period_planning = []
      for _ in range(n_days):
          day_planning = []
          for server_id in range(n_racks):
              start_time = np.random.randint(0, 23)
              machines = np.random.randint(0, 12)
              server = [server_id, start_time, machines]
              day_planning.append(server)

          period_planning.append(day_planning)

      return period_planning
```

Now, we can iteratively call the above function to generate population of any size:

```
  def generate_initial_population(self, population_size, n_days=7, n_racks=11):
          population = []
          for _ in range(population_size):
              member = self.generate_random_plan(
                  n_days=n_days, n_racks=n_racks)
              population.append(member)
          return population
```

### Fitness Function:


Creating a heuristic to find the fitness of any solution is crucial to any implementation of genetic algorithms. In this example, any solution that is most cost efficient would be the most fit solution. So, cost efficiency can be used as a core heuristic whilst developing our fitness function. The cost can be calculated as per how much the solution deviates from the required hourly cron capacity. If there are more servers than required then we are running in over capacity and this will incur costs; on the other hand if we are running less servers than required, we are in under capacity and the job may not complete.

Deviation can be easily calculated by finding the difference between the required capacity with the capacity created by the solution. If the deviation is positive, then we have over capacity, if negative we have under capacity.  

```
    def calculate_fitness(self, deployed_hourly_cron_capacity, required_hourly_cron_capacity):
        deviation = deployed_hourly_cron_capacity - required_hourly_cron_capacity
        overcapacity = abs(deviation[deviation > 0].sum())
        undercapacity = abs(deviation[deviation < 0].sum())
```

We need to associate a quantitative or weighted cost to _both_ over and under capacity - if we only penalize over capacity, then no jobs running at all would be an optimal solution; on the other hand, if we penalize only under capacity, we might end up running all the servers all the time which wastes resources. For this case, since we assumed to be running an IaaS service, we can set the cost of under capacity much higher than over capacity. For this case, we can add the weight of `0.5` for over capacity and `2` for under capacity, under capacity is 4 time more expensive than over capacity - so the algorithm is more incentivized to ensure that all the jobs are able to run.  

```
        overcapacity_cost = 1
        undercapacity_cost = 5

        fitness = overcapacity_cost * overcapacity + undercapacity_cost * undercapacity
        return fitnesss
```


### Biological Operator: Crossover:

Next, we look into the biological operator - Cross over. Here, we try to abstract the process of reproduction - by creating a new generation of population based of the most fit members of the previous generation.

The Crossover function takes in the population and the an attribute for the number of required offspring based on the current population:

```
  def crossover(self, population, n_offspring):
      n_population = len(population)

      offspring = []

```

First, we choose 2 random parents for reproduction:

```
      random_one = population[np.random.randint(
          low=0, high=n_population - 1)]
      random_two = population[np.random.randint(
          low=0, high=n_population - 1)]
```

Then, we choose random bits from each of the parent in such a way that there is no over lap. For this, a binary mask can be created:

```
      dad_mask = np.random.randint(0, 2, size=np.array(random_one).shape)
      # Make mom_mask exclusive of the elements chosen by dad_mask
      mom_mask = np.logical_not(dad_mask)
```

Using this binary mark, we merge these attribute of the two parents to create an offspring:

```
      child = np.add(np.multiply(random_one, dad_mask), np.multiply(random_two, mom_mask))
```

The full function:

```

    def crossover(self, population, n_offspring):
        n_population = len(population)

        offspring = []

        for _ in range(n_offspring):
            random_one = population[np.random.randint(
                low=0, high=n_population - 1)]
            random_two = population[np.random.randint(
                low=0, high=n_population - 1)]

            dad_mask = np.random.randint(0, 2, size=np.array(random_one).shape)
            mom_mask = np.logical_not(dad_mask)

            child = np.add(np.multiply(random_one, dad_mask),
                           np.multiply(random_two, mom_mask))

            offspring.append(child)
        return offspring
```

### Biological Operator: Mutation:

With mutation, there is an effort to add some randomness to expand the solution space. Since with crossover, the existing attributes are recombined each time, an agent of randomness would expand the solution space to beyond what is captured by the previous population.  

For this implementation of mutation, a simple implementation that adds random values at random elements can be implemented.

A function to perform mutation to one solution (member of the population):

```
    def mutate_parent(self, parent, n_mutations):
        size1 = parent.shape[0]
        size2 = parent.shape[1]

        for _ in range(n_mutations):
            rand1 = np.random.randint(0, size1)
            rand2 = np.random.randint(0, size2)
            rand3 = np.random.randint(0, 2)
            parent[rand1, rand2, rand3] = np.random.randint(0, 12)
        return parent
```

`n_mutations` represents the number of mutations to be performed. In general, more randomness and diversity will help avoid the problems with premature convergence, although this comes at a cost of efficiency.

A function that iteratively calls `mutate_parent` on the population:

```
    def mutate_gen(self, population, n_mutations):
        mutated_population = []
        for parent in population:
            mutated_population.append(self.mutate_parent(parent, n_mutations))
        return mutated_population
```

### Constraints

In real world problems, there usually are constraints on the validity of the solution. In this example, an artificial constraint that a rack only has 12 servers has been added. We have to take this into account while evaluating the solutions.

Any solution that assigns more than 12 servers to a rack is invalid:

```
  def is_acceptable(self, parent):
      return np.logical_not((np.array(parent)[:, :, 2:] > 12).any())

  def select_acceptable(self, population):
      population = [
          parent for parent in population if self.is_acceptable(parent)]
      return population
```

### Selection

In each iteration of a genetic algorithm, only the sub-set of the population is selected to carry forward features for the future generations. The fitness metric is utilized to make that choice on which members of the population.  

Let's implement the selection function. The selection function takes in `population`, `required_hourly_cron_capacity` and `n_best` (How many memebers of the population are to be selected).

```
    def select_best(self, population, required_hourly_cron_capacity, n_best):
        fitness = []
```

Next, we iteratively go over each member of the population and find the fitness of the member based on the values from the `calculate_fitness` function. We only choose the most fit `n_best` members of the population for the next generation.

```
      for idx, deployed_hourly_cron_capacity in enumerate(population):

          deployed_hourly_cron_capacity = self.deployed_to_hourlyplanning(
              deployed_hourly_cron_capacity)
          parent_fitness = self.calculate_fitness(deployed_hourly_cron_capacity,
                                                  required_hourly_cron_capacity)
          fitness.append([idx, parent_fitness])

      print('Current generation\'s optimal schedule has cost: {}'.format(
          pd.DataFrame(fitness)[1].min()))

      fitness_tmp = pd.DataFrame(fitness).sort_values(
          by=1, ascending=True).reset_index(drop=True)
      selected_parents_idx = list(fitness_tmp.iloc[:n_best, 0])
      selected_parents = [parent for idx, parent in enumerate(
          population) if idx in selected_parents_idx]

      return selected_parents
```



### Bringing it all together

Now that the core functions are implemented, the `run` function can be implemented that will orchestrate the entire algorithm. The `run` function will also be the entry point of running our genetic algorithm. In this function, we essentially implement the pseudocode added in one of the sections above.

#### Termination Condition:

Before implementing the `run` function, the termination condition has to be set that represents the point at which we are satisfied with the result. In this case, we have a few options options:  

1. Terminate once the cost goes below a certain value.  
2. Terminate once a pre-defined number of generations have been created.  
3. Terminate when the last pre-defined number of generations have the same fitness.  

In this implementation, I will use option 2, choosing the terminate the algorithm once we iterate the pre-defined number of times, regardless of how optimal the result at that point is.   

#### run Function

Now, let's wrap up the run function:

```
def run(self, required_hourly_cron_capacity, n_iterations, n_population_size=500):
    # Generate Initial Population
    population = self.generate_initial_population(population_size=n_population_size, n_days=5, n_racks=24)

    # Generate n_iterations number of generations
    for _ in range(n_iterations):

        #Check if solution is within constraints
        population = self.select_acceptable(population)

        # Selection - Survival of the fittest
        population = self.select_best(population, required_hourly_cron_capacity, n_best=100)

        # Crossover - to form new generation from the best members of the current generation
        population = self.crossover(population, n_offspring=n_population_size)

        # Mutation - Add randomness and increase the solution space
        population = self.mutate_gen(population, n_mutations=1)

    # Return the solution after n_iterations generations are created.
    best_child = self.select_best(population, required_hourly_cron_capacity, n_best=1)
    return best_child

```

## Wrapping up

Now that we have the implementation completed, let's run the algorithm and check out the results.

The complement main function:

```
  def main():

      # Reading from the data file
      df = pd.read_csv("./data/cron_jobs_schedule.csv")

      dataset = df.astype(int).values.tolist()

      required_hourly_cron_capacity = [
          [0 for _ in range(24)] for _ in range(5)]

      for record in dataset:
          required_hourly_cron_capacity[record[1]][record[2]] += record[3]

      genetic_algorithm = CustomGeneticAlgorithm()
      optimal_schedule = genetic_algorithm.run(
          required_hourly_cron_capacity, n_iterations=200)
      print('\nOptimal Server Schedule: \n', optimal_schedule)

```

Output:

```
Current generation's optimal schedule has cost: 384.5
Current generation's optimal schedule has cost: 360.5
Current generation's optimal schedule has cost: 325.5
Current generation's optimal schedule has cost: 322.5
Current generation's optimal schedule has cost: 299.5
Current generation's optimal schedule has cost: 283.5
Current generation's optimal schedule has cost: 269.5
Current generation's optimal schedule has cost: 247.5
Current generation's optimal schedule has cost: 245.0
Current generation's optimal schedule has cost: 238.5
Current generation's optimal schedule has cost: 233.5
Current generation's optimal schedule has cost: 212.5
Current generation's optimal schedule has cost: 205.0
Current generation's optimal schedule has cost: 211.0
Current generation's optimal schedule has cost: 191.0
Current generation's optimal schedule has cost: 204.5
Current generation's optimal schedule has cost: 193.5
Current generation's optimal schedule has cost: 194.0
Current generation's optimal schedule has cost: 187.5
Current generation's optimal schedule has cost: 188.0
Current generation's optimal schedule has cost: 192.5
Current generation's optimal schedule has cost: 194.5
Current generation's optimal schedule has cost: 185.5
Current generation's optimal schedule has cost: 183.5
Current generation's optimal schedule has cost: 174.0
Current generation's optimal schedule has cost: 178.5
Current generation's optimal schedule has cost: 175.0
Current generation's optimal schedule has cost: 176.0
Current generation's optimal schedule has cost: 168.5
Current generation's optimal schedule has cost: 178.5
Current generation's optimal schedule has cost: 174.0
Current generation's optimal schedule has cost: 171.0
Current generation's optimal schedule has cost: 163.0
Current generation's optimal schedule has cost: 165.5
Current generation's optimal schedule has cost: 159.5
Current generation's optimal schedule has cost: 164.5
Current generation's optimal schedule has cost: 163.0
Current generation's optimal schedule has cost: 162.0
Current generation's optimal schedule has cost: 154.5
Current generation's optimal schedule has cost: 152.5
Current generation's optimal schedule has cost: 156.0
Current generation's optimal schedule has cost: 153.5
Current generation's optimal schedule has cost: 152.0
Current generation's optimal schedule has cost: 151.5
Current generation's optimal schedule has cost: 148.5
Current generation's optimal schedule has cost: 152.0
Current generation's optimal schedule has cost: 153.0
Current generation's optimal schedule has cost: 142.0
Current generation's optimal schedule has cost: 142.0
Current generation's optimal schedule has cost: 142.5
Current generation's optimal schedule has cost: 141.5
Current generation's optimal schedule has cost: 141.5
Current generation's optimal schedule has cost: 138.5
Current generation's optimal schedule has cost: 141.0
Current generation's optimal schedule has cost: 135.5
Current generation's optimal schedule has cost: 134.5
Current generation's optimal schedule has cost: 132.0
Current generation's optimal schedule has cost: 131.0
Current generation's optimal schedule has cost: 126.5
Current generation's optimal schedule has cost: 132.0
Current generation's optimal schedule has cost: 127.0
Current generation's optimal schedule has cost: 126.0
Current generation's optimal schedule has cost: 126.5
Current generation's optimal schedule has cost: 115.0
Current generation's optimal schedule has cost: 112.0
Current generation's optimal schedule has cost: 119.5
Current generation's optimal schedule has cost: 118.5
Current generation's optimal schedule has cost: 113.5
Current generation's optimal schedule has cost: 111.0
Current generation's optimal schedule has cost: 102.5
Current generation's optimal schedule has cost: 106.0
Current generation's optimal schedule has cost: 106.5
Current generation's optimal schedule has cost: 101.0
Current generation's optimal schedule has cost: 98.5
Current generation's optimal schedule has cost: 95.5
Current generation's optimal schedule has cost: 93.5
Current generation's optimal schedule has cost: 93.5
Current generation's optimal schedule has cost: 93.0
Current generation's optimal schedule has cost: 90.5
Current generation's optimal schedule has cost: 91.0
Current generation's optimal schedule has cost: 88.0
Current generation's optimal schedule has cost: 87.5
Current generation's optimal schedule has cost: 86.0
Current generation's optimal schedule has cost: 84.0
Current generation's optimal schedule has cost: 83.5
Current generation's optimal schedule has cost: 83.0
Current generation's optimal schedule has cost: 81.5
Current generation's optimal schedule has cost: 80.0
Current generation's optimal schedule has cost: 77.0
Current generation's optimal schedule has cost: 74.5
Current generation's optimal schedule has cost: 71.0
Current generation's optimal schedule has cost: 70.0
Current generation's optimal schedule has cost: 67.5
Current generation's optimal schedule has cost: 68.5
Current generation's optimal schedule has cost: 63.5
Current generation's optimal schedule has cost: 62.0
Current generation's optimal schedule has cost: 62.0
Current generation's optimal schedule has cost: 61.0
Current generation's optimal schedule has cost: 61.0
Current generation's optimal schedule has cost: 61.0

```

The algorithm converges at the cost of `61`. 

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
