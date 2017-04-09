---
layout: post
title:  "Parallel Computing in JavaScript : The Guide"
date:   2017-03-29 18:34:56 +0530
description: Parallel programming in JavaScript is not as straight-forward as it is in languages such as C/C++ and Java or even Python due to its event based paradigm. JavaScript, while traditionally being used for performing small computations, is being increasingly used for heavy-wight applications. In this post I will focussing on parallel computation in JavaScript through Web Workers and the `parallel.js` library.
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

Perhaps the most solid obstacle to parallel programming in JS was it's event based paradigm. 

>JavaScript has a concurrency model based on an "event loop" where almost all I/O is non-blocking. When the operation has been completed, a message is enqueued along with the provided callback function. At some point in the future, the message is dequeued and the callback fired.

The independence of the caller from the response allows for the JS runtime to do other things while waiting for the asynchronous operation to complete and their callbacks to trigger.

>JavaScript runtimes contain a message queue which stores a list of messages to be processed and their associated callback functions. These messages are queued in response to external events (such as a mouse being clicked or receiving the response to an HTTP request) given a callback function has been provided.

The runtime has a single thread listening to _events_, when one of the events fire the main thread _calls_ the requisite callback function and runs it in the background. Once the event is handled, it returns back to the main thread. This way, the single _event loop_ can handle any number of events because the main event loop is _non-blocking_. This same model is implemented regardless of where JavaScript runs, in the **client side** or the **server side**(or practically anywhere else).

Now, if we were to parallelize JavaScript by having multiple threads, then by definition, each thread would be listening to the event. On the occurance of the said event, it is not possible to determine which thread or how many threads will handle the event. It is very possible that all the threads handle the same event and the single event gets handled _n_ number of times, where _n_ is the number of threads. So, it suffices to say that the normal paradigm of parallel computing wouldn't work. JavaScript needs something more "JavaScripty" - another work around in the long array of JavaScript work arounds.

> Normally in order to achieve any sort of parallel computation using JavaScript you would need to break your jobs up into tiny chunks and split their execution apart using timers. This is both slow and unimpressive.

Thanks to web workers, we now have a better way to achieve this.

# Enter Web Workers

Parallel computing in JavaScript can be achieved through web workers, which were introduced with the HTML5 specification. 

> A web worker, as defined by the World Wide Web Consortium (W3C) and the Web Hypertext Application Technology Working Group (WHATWG), is a JavaScript script executed from an HTML page that runs in the background, independently of other user-interface scripts that may also have been executed from the same HTML page. Web workers are often able to utilize multi-core CPUs more effectively. Web workers are relatively heavy-weight. They are expected to be long-lived, have a high start-up performance cost, and a high per-instance memory cost.

Web workers allow the user to run JavaScript in parallel without interfering the user interface. A _worker_ script will be loaded and run in the backgroud, completely independent of the user interface scripts. This means that the workers do not have any access to the user interface elements, such as the DOM and common JS functions like `getElementById`(You can still make AJAX calls using Web Workers). The prime use case of the web worker API is to perform computationally expensive task in the background, without interrupting or being interrupted by user interaction. 

With web workers, it is now possbile to have multiple JS threads running in parallel. It allows the browser to have a normal operation with the event loop based execution of the single main thread on the user side, while making room for multiple threads in the background. Each web worker has a separate message queue, event loop, and memory space independent from the original thread that instantiated it. 

Web workers communicate with the main document/ the main thread via message passing technique. The message passing is done using the [postMessage]() API.

![K Means Math]({{site.baseurl}}/images/web-workers.png)


That's it for now; if you have any comments, please leave them below.


<br /><br />