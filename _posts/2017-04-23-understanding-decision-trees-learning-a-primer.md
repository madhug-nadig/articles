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


## Representating a Decision Tree

Decision trees perform classification after sorting the instances in a top-down approach - from the root to the leaf. Each non-leaf node _splits_ the set of instances based on a test of an attribute. Each branch emanting from a node corresponds to one of the possible values of the said attribute in the node. The leaves of the decision tree specifies the label or the class in which a given instance belongs to. 

Here's an example:

  <style>
	
	.node {
		cursor: pointer;
	}

	.node circle {
	  fill: #fff;
	  stroke: steelblue;
	  stroke-width: 3px;
	}

	.node text {
	  font: 12px sans-serif;
	}

	.link {
	  fill: none;
	  stroke: #ccc;
	  stroke-width: 2px;
	}
	
    </style>

<!-- load the d3.js library -->	
<script src="http://d3js.org/d3.v3.min.js"></script>
	
<script>
var width = 400,
    height = 300;

var tree = d3.layout.tree()
    .size([height, width - 160]);

var diagonal = d3.svg.diagonal()
    .projection(function (d) {
        return [d.y, d.x];
    });

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .attr("transform", "translate(40,0)");

var root = getData(),
    nodes = tree.nodes(root),
    links = tree.links(nodes);

var link = svg.selectAll(".link")
    .data(links)
    .enter()
    .append("g")
    .attr("class", "link");

link.append("path")
    .attr("fill", "none")
    .attr("stroke", "#ff8888")
    .attr("stroke-width", "1.5px")
    .attr("d", diagonal);

link.append("text")
    .attr("font-family", "Arial, Helvetica, sans-serif")
    .attr("fill", "Black")
    .style("font", "normal 12px Arial")
    .attr("transform", function(d) {
        return "translate(" +
            ((d.source.y + d.target.y)/2) + "," + 
            ((d.source.x + d.target.x)/2) + ")";
    })   
    .attr("dy", ".35em")
    .attr("text-anchor", "middle")
    .text(function(d) {
        console.log(d.target.rule);
         return d.target.rule;
    });

var node = svg.selectAll(".node")
    .data(nodes)
    .enter()
    .append("g")
    .attr("class", "node")
    .attr("transform", function (d) {
        return "translate(" + d.y + "," + d.x + ")";
    });

node.append("circle")
    .attr("r", 4.5);

node.append("text")
    .attr("dx", function (d) {
        return d.children ? -8 : 8;
    })
    .attr("dy", 3)
    .style("text-anchor", function (d) {
        return d.children ? "end" : "start";
    })
    .text(function (d) {
        return d.name;
    });

function getData() {
    return {
        "name": "0",
        "rule": "null",
            "children": [{
            "name": "2",
            "rule": "sunny",
                "children": [{
                "name": "no(3/100%)",
                "rule": "high"
            }, {
                "name": "yes(2/100%)",
                "rule": "normal"
            }]
        }, {
            "name": "yes(4/100%)",
            "rule": "overcast"
        }, {
            "name": "3",
            "rule": "rainy",
                "children": [{
                "name": "no(2/100%)",
                "rule": "TRUE"
            }, {
                "name": "yes(3/100%)",
                "rule": "FALSE"
            }]
        }]
    };
};
</script>

<br /><br />