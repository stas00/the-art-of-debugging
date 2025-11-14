# Incoming Fodder

Please ignore this file. This is just a dump of ideas that I'd like to expand on in the future

## resources

https://missing.csail.mit.edu/2020/debugging-profiling/


## misc topics

- randomization - or rather determinism


## strace

Say you get a: "No space left on device" traceback, but it doesn't tell you which file it failed to write to.

You could try to go and figure out where the file handle was opened in the code, but often it can be far from trivial.

strace comes to help:


```
687222 openat(AT_FDCWD, "/tmp/tmp5_gxu46v", O_RDONLY|O_CLOEXEC <unfinished ...>
```


## blackbox approach to complex systems


## when to use print vs CLI debugger vs IDE debugger

Usually one has three major approaches to debugging the variables and states inside software: manually injected `print`,

For a quick

IDE Debuggers

Tools like [PyCharm](https://www.jetbrains.com/pycharm/) and
[VSCode](https://code.visualstudio.com/) have awesome visual python debuggers.




## Debugging Unix Tools:

bash -x, bash -eo pipe
make -n


## Users wishlist

Users asked to cover the following topics:

Heikki Arponen - One addition suggestion though: remote ssh development. Works great on vs code (can be a pain with pycharm). Debugging DDP can be a bit unstable, but sometimes even that is OK
