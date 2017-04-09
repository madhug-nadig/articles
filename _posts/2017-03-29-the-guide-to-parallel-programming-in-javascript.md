---
layout: post
title:  "Parallel Computing in JavaScript : The Guide"
date:   2017-03-29 18:34:56 +0530
description:   
categories: Parallel-Processing

---

Back when JavaScript was developed as _Mocha_ at Netscape Communications Corporation by Brendan Eich in 1995, the language was meant for adding _programmability_ on the web - making the web more interactive for the user. Over the years, JavaScript has gained massive prominence and has become one of the most important languages of the day. The rise of the web has taken JavaScript places it was never conceived to be. 

Anyone who has written considerable amount of code in JavaScript will not be surprised to know that JavaScript [was written in 10 days](https://www.computer.org/csdl/mags/co/2012/02/mco2012020007.pdf). As lovable as it may be, JavaScript has subtle bug friendly features and [lot of other things that programmers would rather not have](https://whydoesitsuck.com/why-does-javascript-suck/). But it is one of the essential programming(scripting?) languages in today's web driven world. Hence, it is imperative to implement the best of software programming techniques and technologies in JavaScript.

As JavaScript grew out of its bounds, lots and lots of features were added on the go which weren't thought of initially. The addition of all these unanticipated features, from functional programming to OOP, were pretty much work arounds of existing language features.
One of the cool features added to JS in the recent years was the ability to parallel compute JavaScript and this will the focus of the following article.

# Why parallel processing in JS

## More and More JS on the browser

You might ask why parallelize JavaScript, after all you use on the browser for small scale scripting. Sure, if you using JavaScript for an `onclick` handler or for a simple animation effect, you do _not_ need parallelization. However, there are many heavyweight JS applications on the web that do a lot of procesing in JavaScript:

 > Client-side image processing (in Facebook and Lightroom) is written in JS; in-browser office packages such as Google Docs are written in JS; and components of Firefox, such as the built-in PDF viewer, pdf.js, and the language classifier, are written in JS. In fact, some of these applications are in the form of asm.js, a simple JS subset, that is a popular target language for C++ compilers; game engines originally written in C++ are being recompiled to JS to run on the web as asm.js programs.

The amount JavaScript in a webpage is steadily increasing over the years and the reason to parallelize JS is same as the reason to parallelize any other langauge: Moore's law is dying and multi-core architecture is taking over the world.

## JavaScript is everywhere

JavaScript is no longer confined to the browser. It runs everywhere and on anything, from servers to IoT devices. Many of these programs are heavy-weight and might tremendously benefit from the fruits of parallel computing. The campaign to [_javascript everything_](https://www.browseemall.com/Blog/index.php/2015/11/05/should-we-use-javascript-for-everything-now/) has succeded in a [way](https://medium.com/@tracend/javascript-everything-91c20a23930) and finding ways to parallel compute JavaScript in non-browser environments is now important.

# The JavaScript event Loop

That's it for now; if you have any comments, please leave them below.


<br /><br />