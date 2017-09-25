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
var json = 
{
    "name": "Sex",
    "children": [
        {
            "name": "Male",
            "children": [
                {
                    "name": "Is Age Greater than 9.5",
                    "children": [
                        {"name": "Child 1"},
                        {"name": "Child 2"},
                        {"name": "Child 3"}
                    ]
                },
                {
                    "name": "Section 2",
                    "children": [
                        {"name": "Child 1"},
                        {"name": "Child 2"},
                        {"name": "Child 3"}
                    ]
                }
            ]
        },
        {
            "name": "Female",
            "children": [
                { "name": "Survived" }
                    
            ]
        }
    ]
};

var width = 700;
var height = 650;
var maxLabel = 150;
var duration = 500;
var radius = 5;
    
var i = 0;
var root;

var tree = d3.layout.tree()
    .size([height, width]);

var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
        .append("g")
        .attr("transform", "translate(" + maxLabel + ",0)");

root = json;
root.x0 = height / 2;
root.y0 = 0;

root.children.forEach(collapse);

function update(source) 
{
    // Compute the new tree layout.
    var nodes = tree.nodes(root).reverse();
    var links = tree.links(nodes);

    // Normalize for fixed-depth.
    nodes.forEach(function(d) { d.y = d.depth * maxLabel; });

    // Update the nodes…
    var node = svg.selectAll("g.node")
        .data(nodes, function(d){ 
            return d.id || (d.id = ++i); 
        });

    // Enter any new nodes at the parent's previous position.
    var nodeEnter = node.enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", function(d){ return "translate(" + source.y0 + "," + source.x0 + ")"; })
        .on("click", click);

    nodeEnter.append("circle")
        .attr("r", 0)
        .style("fill", function(d){ 
            return d._children ? "lightsteelblue" : "white"; 
        });

    nodeEnter.append("text")
        .attr("x", function(d){ 
            var spacing = computeRadius(d) + 5;
            return d.children || d._children ? -spacing : spacing; 
        })
        .attr("dy", "3")
        .attr("text-anchor", function(d){ return d.children || d._children ? "end" : "start"; })
        .text(function(d){ return d.name; })
        .style("fill-opacity", 0);

    // Transition nodes to their new position.
    var nodeUpdate = node.transition()
        .duration(duration)
        .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

    nodeUpdate.select("circle")
        .attr("r", function(d){ return computeRadius(d); })
        .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

    nodeUpdate.select("text").style("fill-opacity", 1);

    // Transition exiting nodes to the parent's new position.
    var nodeExit = node.exit().transition()
        .duration(duration)
        .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
        .remove();

    nodeExit.select("circle").attr("r", 0);
    nodeExit.select("text").style("fill-opacity", 0);

    // Update the links…
    var link = svg.selectAll("path.link")
        .data(links, function(d){ return d.target.id; });

    // Enter any new links at the parent's previous position.
    link.enter().insert("path", "g")
        .attr("class", "link")
        .attr("d", function(d){
            var o = {x: source.x0, y: source.y0};
            return diagonal({source: o, target: o});
        });

    // Transition links to their new position.
    link.transition()
        .duration(duration)
        .attr("d", diagonal);

    // Transition exiting nodes to the parent's new position.
    link.exit().transition()
        .duration(duration)
        .attr("d", function(d){
            var o = {x: source.x, y: source.y};
            return diagonal({source: o, target: o});
        })
        .remove();

    // Stash the old positions for transition.
    nodes.forEach(function(d){
        d.x0 = d.x;
        d.y0 = d.y;
    });
}

function computeRadius(d)
{
    if(d.children || d._children) return radius + (radius * nbEndNodes(d) / 10);
    else return radius;
}

function nbEndNodes(n)
{
    nb = 0;    
    if(n.children){
        n.children.forEach(function(c){ 
            nb += nbEndNodes(c); 
        });
    }
    else if(n._children){
        n._children.forEach(function(c){ 
            nb += nbEndNodes(c); 
        });
    }
    else nb++;
    
    return nb;
}

function click(d)
{
    if (d.children){
        d._children = d.children;
        d.children = null;
    } 
    else{
        d.children = d._children;
        d._children = null;
    }
    update(d);
}

function collapse(d){
    if (d.children){
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
    }
}

update(root);

</script>

<br /><br />