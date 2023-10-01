# Quick Iterations, Small Payload

## The most important needs of successful debugging

There are just 2 needs:

1. The debug cycles have to be quick
2. The data being debugged has to be small

### 1. Quick debug cycles

If you have to wait 10 minutes to get to the point where the problem happens after you restarted the program and you have to repeat that 30 times, you will not only tire of waiting and likely to give up, but you will have a very difficult time correlating your attempts to figure out what's going on with the response of the program. While you wait for the program to get to the point you want to scrutinize you are likely to start thinking about new ideas and meanwhile get confused of which of them is really being tested.

Therefore, it's critical to try to make the time between the relaunch and the critical point as fast as possible. **Ideally it should take a few seconds**. Granted - it's not always possible, in which case any shortening of the cycle will improve the chance the debugging will be successful.

Which takes us naturally to the second requirement.

### 2. Small payload

Most of the time programs deal with a lot of data, it can be as large as many GBs, as is the case, for example, with huge models in machine learning.

Shrinking the amount of data you work with to a small or even better tiny data is crucial for the following reasons:

1. Fast restart time (see the previous need of quick debug cycles) - tiny data helps with faster restart time.
2. It's much easier to remember, compare and do mental math on this:

```
tensor([[0.1, 0.2],
        [0.3, 0.4]])
```

```
tensor([[0.8860, 0.7966, 0.0274,  ..., 0.4142, 0.7156, 0.3564],
        [0.3885, 0.1056, 0.3069,  ..., 0.8970, 0.8329, 0.0012],
        [0.1999, 0.8671, 0.8428,  ..., 0.4565, 0.4934, 0.3453],
        ...,
        [0.0877, 0.0928, 0.4091,  ..., 0.6417, 0.8201, 0.3010],
        [0.6187, 0.0016, 0.5060,  ..., 0.1208, 0.4379, 0.5473],
        [0.7241, 0.7567, 0.7357,  ..., 0.6959, 0.1976, 0.7924]])
```

I hope the data can speak for itself and require no further commentary to why the first one is preferable to the second one.



## Real data vs. random data vs. synthetic data

When debugging a program that keeps on crashing you don't need real data, any random or synthetic data will do.

Only when the program works, but it's not producing the expected quality, you need the actual real data.

Depending on a situation you'd start with either random or synthetic data and then progress to real data.

Sometimes it's good enough to just call `rand` and create a small payload like:

```
tensor([[0.0595, 0.3011],
        [0.6725, 0.3576]])
```

But if you want to keep track of numbers and if you need to perform some transformations surely it's easier to do that with a payload like:

```
tensor([[0.1, 0.2],
        [0.3, 0.4]])
```

If you ever debugged an assembly code that is operates on hex code (`[0-9A-F]`) often someone would inject hex  sequences like `DEADBEEF` because it really stands out in a hex code like: `A0E92827A99DEADBEEF183E`. So, often, you can be creative and create synthetic data which will really make it easy to see what the problem is. At other times any **small** random data will do.

Even if you have to eventually use real data, still try to use the smallest possible real data.

For example, if you are debugging a machine learning project, that is training a 175B parameter model, but try to use a 1B or even a 125M parameter model, for quality needs. And a 10K parameter model for crashing debugging.



## Single process, single cpu, single gpu

Big data, big resources, big overheads

Large data often doesn't fit onto a single CPU or GPU and requires some parallelization method that would require multiple processes makes debugging very difficult. Most debuggers won't be able to handle such tasks and `print` will be your only friend and savior.



## race conditions

gdb, pdb, whatever other debugger you use, often changes the timing of the process and a deadlock or a race condition can't be reproduced.

print and strace


## async vs sync

often turning async behavior off reveals the problem very easily.
