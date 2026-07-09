# Debugging Python Programs

## Print-based techniques

### q

It can be difficult to use `print` calls to debug a Python program that already prints a lot of messages to `stdout`, `stderr` and logger streams, which could make it difficult to see your custom debug print messages. The `q` package comes to rescue in this situation.

This package, that can be installed from https://github.com/zestyping/q or `pip install q`, is designed for a quick code and function tracing. By default it sends its output into the `/tmp/q` file, so in one console start watching this file:

```bash
tail -F /tmp/q
```

Then in your code wrap code statements you want traced with `q()` and decorate functions with `@q`. Here is an example:

```python
# q1.py
import q

# trace any code and its outputs
q(1+2)

# to trace function arguments and returns add @q before each function definition
@q
def add(a,b): return a+b
add(1, 2)
```
Let's run it:
```bash
python q1.py
```

Now you can ignore the normal std/logger streams and just watch `/tmp/q`:

```bash
$ cat /tmp/q

 0.0s <module>: 1+2=3
 0.0s add(1, 2)
 0.0s -> 3
```

It first showed the statement `1+2` and its outcome `3`, followed by a trace of a function with its arguments `1, 2` and the return value `3`.

It gave us even a rough execution time information in the first column, but secs is too low of a resolution for this particular code example.

notes:
- on MacOS before using this package you need to add:
```bash
export TMPDIR=/tmp/
```
in the console you run the python program from.


### Printing object variables

If you have an object to debug you can't just dump its contents easily. Let's create a simple class with 2 variable - one class and one instance variable, create an object and try to print its contents:

```python
# print_object_1.py
class A():
    foo = 1
    def __init__(self): self.bar = 2

a = A()
print(a)
```
```bash
$ python print_object_1.py
<__main__.A object at 0x1012b9b50>
```
This is not very useful as the object appears opaque and `print` can't look inside.

Python has a special `__repr__` method that helps to make object's data print-friendly:

```python
# print_object_2.py
class A():
    foo = 1
    def __init__(self): self.bar = 2
    def __repr__(self): return f"{self.foo=}, {self.bar=}"

a = A()
print(a)
```
```bash
$ python print_object_2.py
self.foo=1, self.bar=2
```
So this is great, but we have to hope the creator of the class provided this helper method.

For example, if the class is wrapped with `@dataclasses.dataclass`, it already comes with a `__repr__` method:
```python
# print_object_3.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class A():
    foo: Optional = 1
    def __init__(self): self.bar = 2
a = A()
print(a)
```
```bash
$ python print_object_3.py
A(foo=1)
```
As you can see it only prints the class variables (`foo=1`), the instance variables are missed by its `__repr__` - so this is not helpful.

There is also built-in `vars`, which also has the issue that it shows only the instance member and not the class one:

```python
# print_object_4.py
class A():
    foo = 1
    def __init__(self): self.bar = 2

a = A()
from pprint import pprint
pprint(vars(a))
```
```bash
$ python print_object_4.py
{'bar': 2}
```


Since we don't always have classes under our control and while we can monkey patch a class, it's at times easier to just use some smart util that can do it automatically for you. For example, `rich` `inspect()` can do it:

```python
# print_object_5.py
import rich;
class A():
    foo = 1
    def __init__(self): self.bar = 2

a = A()
rich.inspect(a)
```
```bash
$ pip install rich
$ python print_object_5.py
╭───────── <class '__main__.A'> ─────────╮
│ ╭────────────────────────────────────╮ │
│ │ <__main__.A object at 0x1017c7860> │ │
│ ╰────────────────────────────────────╯ │
│                                        │
│ bar = 2                                │
│ foo = 1                                │
╰────────────────────────────────────────╯
```

footnote: I'm looking for a more user-friendly replacement for `rich.inspect()` since it is too opinionated wrt putting borders around all its outputs and this is really a problem for post processing, that's why I avoid using this otherwise handy module.

You can control what attributes are dumped via various args to [`rich.inspect()`](https://rich.readthedocs.io/en/stable/reference/init.html#rich.inspect).

The closest Python built-in way I found is `inspect.getmembers()`, but alas it won't let me filter the variable attributes (but it does let you filter by other types of attributes via a [`predicate` argument](https://docs.python.org/3.13/library/inspect.html#inspect.getmembers)).

```python
# print_object_6.py
import inspect
class A():
    foo = 1
    def __init__(self): self.bar = 2

a = A()
from pprint import pprint as pp
pp(inspect.getmembers(a))
```
```bash
$ python print_object_6.py
[('__class__', <class '__main__.A'>),
 ('__delattr__', <method-wrapper '__delattr__' of A object at 0x104953b90>),
 ('__dict__', {}),
 [...] trimmed output
 ('__weakref__', None),
 ('bar', 2),
 ('foo', 1)]
```
As you can see it prints the variable attributes, but also a lot of dunder attributes we don't want. Surely we could post process its output to skip `r"__"` matches but that's too much code to write - we want something that is fast to write unless you supplement each code base you debug with your own debug library, which at times is a great option, if you put it on pypi so that it's quick to install.

You may have noticed I used `pprint` in a few code samples - which is another handy built-in library for pretty printing nested lists, dictionaries and other large structures. If I were to just `print` - there would have been one long line of output.


### Auto-print what's being observed

Say, you want to see the output of summation:

```bash
$ python -c 'x=5; y=6; print(f"{x+y}")'
11
```
But then you don't know what's being summed in the printout, so you have to write then:

```bash
$ python -c 'x=5; y=6; print(f"x+y={x+y}")'
x+y=11
```
But this is both long and error-prone, because the 2 parts aren't atomic - you may choose to modify the expression in the second part `{x+y+1}`, but forget to update the first part and end up with wrong conclusions.

Since python-3.8 there is an atomic operand auto-description feature. Let's rewrite the last one liner to remove the description of what's being printed and append a magical `=` to the expression inside `{}`:
```bash
$ python -c 'x=5; y=6; print(f"{x+y=}")'
x+y=11
```
Now what's being evaluated is automatically printed. All you need to do is to add `=` before the closing `}`.

Here is another example:
```bash
$ python -c 'x=5; y=6; print(f"{x=}, {y=}")'
x=5, y=6
```

So once again you can see that atomic operations are ideal for fruitful debugging.

### Who is calling?

Everybody knows that when a Python program crashes a stack trace (traceback) is printed and that's how we know where to fix things.

But how do you go about discovering whether what you're working on gets actually called and don't get a false impression that everything works, when in reality it's just hasn't been called. A common use case of this is when dealing with multiple copies of the same code - e.g. multiple virtual environments or git repo clones. I use a quick and dirty solution. I add `die` to the code:

```bash
$ cat << EOT > test.py
def a():
    print("a was called")
    die
def b(): a()
def c(): a()
b()
EOT

$ python test.py
```

gives us:

```
a was called
Traceback (most recent call last):
  File "/test.py", line 6, in <module>
    b()
  File "/test.py", line 4, in b
    def b(): a()
             ^^^
  File "/test.py", line 3, in a
    die
NameError: name 'die' is not defined. Did you mean: 'dir'?
```

So here we can immediately see that it was `b()` that called `a()` and the right `a()` was called (the one I was editing) and not in some other file or virtual environment.

The `die` trick - why any non-Python word works as the "break", and where the name comes from - is explained in [Am I editing the right file and the right class?](../methodology/README.md#am-i-editing-the-right-file-and-the-right-class).

`traceback.print_stack()` is another way to check the right code path was chosen or to discover the callers, since in complex code bases the same function can be called by very different code branches.


```bash
$ cat << EOT > test.py
import traceback
def a():
    traceback.print_stack()
    print("a was called")
def b(): a()
def c(): a()
b()
EOT

$ python test.py
```

gives us:
```bash
python test.py
  File "/test.py", line 7, in <module>
    b()
  File "/test.py", line 5, in b
    def b(): a()
  File "/test.py", line 3, in a
    traceback.print_stack()
a was called
```
So again, you can see that it was `b()` that called `a()`. But here the program continued running, which often is undesirable since if there are many logs it might be tricky to find the traceback's output. That's why my personal preference is the `die` solution above.

## Running the code you think you are running

### Ensuring the Python package you edit is the one that is run

If you're modifying a git repository of a Python package and then installing into a virtual environment this could be a very error-prone process since you are never sure if you reinstalled the modified files or not when you test things. And that's why instead of using `pip install .` it's much better to use `pip install -e .` which instead of installing the Python files into the virtual environment, tells the latter to access the files from the git clone (or whatever other way the source code is made available). For example, if you're working on HF transformers, here is how to do it better:
```bash
git clone https://github.com/huggingface/transformers/
cd transformers
pip install -e .
python my-program.py
```
now you can tweak the files inside the git clone and they will be automatically used at run time.

Yet another approach is with the help of `PYTHONPATH`. In this approach you don't install the package into the virtual environment, but instead you tell `python` to load it directly from its source. Repeating the last example, it would change into:
```bash
git clone https://github.com/huggingface/transformers/
cd transformers
PYTHONPATH=src python my-program.py
```
If you compare before and after:
1. there is no `pip install`
2. a new environment variable `PYTHONPATH=src` is added before the program is run and it's active only for the duration of the program.

we used `src` because that's where source files of HF `transformers` package reside, that is they are under `[src/](https://github.com/huggingface/transformers/tree/main/src/). Other git repos could use a different path where they place their modules, and actually it's even more common to not have any prefix at all. If there is no prefix then use `PYTHONPATH=.` which means the current directory is where Python can find the packages at.

Why would you want to do that? If you're like me, I have multiple clones of the same repo, e.g.:

```
transformers-pr-18998
transformers-project-a
transformers-problem-b
```
And I work in parallel on multiple features, where each clone uses a different branch. And most of the time I use the same conda environment for all of them. Thus when I develop things I might use `pip install -e .` but if I have to go back and force between different projects frequently I purposefully uninstall `pip uninstall transformers` to ensure that some unrelated version gets loaded and instead I use `PYTHONPATH` at run time. For example, I start with:
```bash
cd transformers-pr-18998
PYTHONPATH=`pwd`/src python myprogram
```

and possibly in another terminal I'll run:

```bash
cd transformers-project-a
PYTHONPATH=`pwd`/src python myotherprogram
```
and each program will see only the files from the right branch.

If `PYTHONPATH` was somehow already non-empty (rare), like any other `PATH`-type environment variables you can prefix to it like so:
```bash
PYTHONPATH=`pwd`/src:$PYTHONPATH python ...

```
The order is critical - the earlier paths will have a higher precedence than later ones.

Since it's easy to use an invalid path at `PYTHONPATH` and there will be no error as long as Python will find a version of the files you try to load, there are 2 ways to ensure that the correct libraries are loaded:

1. You can dump `sys.path` which `PYTHONPATH` updates:
```bash
$ cd transformers-a
$ PYTHONPATH=`pwd`/src:$PYTHONPATH python -c 'import sys; print("\n".join(sys.path))'
/code/huggingface/transformers-a/src
/code/huggingface/transformers-a
/home/stas/anaconda3/envs/py39-pt21/lib/python39.zip
/home/stas/anaconda3/envs/py39-pt21/lib/python3.9
/home/stas/anaconda3/envs/py39-pt21/lib/python3.9/lib-dynload
/home/stas/anaconda3/envs/py39-pt21/lib/python3.9/site-packages
```
in this example I'm continuing the situation with HF `transformers` where `src` is the base sub-directory inside the git clone, but other projects might have a different subdir or none. Note how `/code/huggingface/transformers-a/src` appears first in the list.

2. You can use the [purposefully break the script approach](../methodology/README.md#am-i-editing-the-right-file-and-the-right-class) to validate that the files you're editing are the files that actually get loaded.

### Setting up your test suite to always use the git repo's Python packages

If your test suite relies on the package it tests being preinstalled you are likely to be testing the wrong files. This is usually less of a problem when the git package places the Python packages at the root directory of the repo, but when a project is structured like HF `transformers` where Python packages are placed under `src` or another top-level subdir Python will not find these packages. Also if you launch the tests not from the repo's root directory it'll always fail to find your repo's packages.

There is an easy solution to that. You can see how I did it in [ipyexperiments](https://github.com/stas00/ipyexperiments). I created [tests/conftest.py](https://github.com/stas00/ipyexperiments/blob/39c9b454e89e53b74c74dcb579c12ecf3d2161b9/tests/conftest.py#L7-L12) which contains:
```python
import sys
from os.path import abspath, dirname

# make sure we test against the checked out git version of the repo and
# not the pre-installed version. With 'pip install -e .' it's not
# needed, but it's better to be safe and ensure the git path comes
# second in sys.path (the first path is the test dir path)
git_repo_path = abspath(dirname(dirname(__file__)))
sys.path.insert(1, git_repo_path)
```
if you prefer `pathlib` the same code would look like:
```python
from pathlib import Path
# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(1, git_repo_path)
```

That's it, now your tests will always have the repo path appear second in `sys.path` (after the test dir path).

footnote: you can choose to insert into position `0` instead of `1` as well, but I like to keep Python's current directory as the first one to look for files.

If your project uses a sub-directory like HF `transformers`' `src`, simply add it to the `git_repo_path`, e.g. you can see how this is done [here](https://github.com/huggingface/transformers/blob/6cbc1369a330860c128a1ba365f246751382c9e5/conftest.py#L30-L31)

As you are figuring it out simply dump:
```python
print("\n".join(sys.path))
```
and check that the paths are correct.

## Diagnosing hangs

### py-spy

`py-spy` is a great tool for diagnosing processes that either hang or spin out of control or there is an issue with a network connection, blocked IO, etc. It's similar to getting a traceback on exception, except the process is still running.

It's covered in depth, with worked examples and multi-process/multi-node recipes, in the PyTorch chapter - see [py-spy](../pytorch/README.md#py-spy). While the examples there use PyTorch, the tool and the techniques apply to any Python program.

## Profilers

When a program runs slower than desired, or if the compute is expensive, often it's possible to rewrite parts of the program to make it faster. For example, if some function is run in a loop, repeating hundreds of thousands of times speeding it up just a little bit can make the total runtime much faster. Very often 20% of the code contributes to 80% of execution time, so finding that code and optimizing it should improve the performance. It also helps to understand where it's worthwhile to invest the optimization efforts.

Let's look at some of the useful Python profilers.

### cProfile

cProfile is a built-in profiler that is quite easy to use. The full documentation is [here](https://docs.python.org/3/library/profile.html).

#### Command line

The simplest way of trying it out is to run your existing program with `-m cProfile`. Assuming the program is:

```python
# sleep.py
import time
def run():
    for _ in range(3):
        time.sleep(0.1)
run()
```

We can now run:
```bash
$ python -m cProfile sleep.py
         7 function calls in 0.300 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.300    0.300 {built-in method builtins.exec}
        1    0.000    0.000    0.300    0.300 sleep.py:1(<module>)
        1    0.000    0.000    0.300    0.300 sleep.py:2(run)
        3    0.300    0.100    0.300    0.100 {built-in method time.sleep}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

What can we learn from the output of the profiler:
- `time.sleep` was called 3 times, for a total of 0.3sec, each call being 0.1sec
- `run` was called once, and close to zero time was spent in the function itself (`tottime`) but cumulatively 0.3sec passed

When running a real program it's often important to sort the output. The default sorting is by cumulative time but you can sort by any column name from the output above with the help of the `-s` argument. For example, `-s ncalls` will sort by `ncalls` (the number of calls). Let's validate:

```bash
$ python -m cProfile -s ncalls sleep.py
         7 function calls in 0.300 seconds

   Ordered by: call count

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        3    0.300    0.100    0.300    0.100 {built-in method time.sleep}
        1    0.000    0.000    0.300    0.300 {built-in method builtins.exec}
        1    0.000    0.000    0.300    0.300 sleep.py:2(run)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.300    0.300 sleep.py:1(<module>)
```

And it is so.

#### cProfile Context manager

Profiling a whole big program is often ineffective as you may have too many things to sort through. If you already know which code segment you want to profile, it's best to profile just it. Let's see how we can use `cProfile.Profile()` as a context manager:

```python
# sleep2.py
import time
import cProfile
from pstats import SortKey
with cProfile.Profile() as pr:
    for _ in range(3):
        time.sleep(0.1)
pr.print_stats(SortKey.CUMULATIVE)
```

Now we run the program normally, without passing `-m cProfile` at the command line:
```bash
$ python sleep2.py
         5 function calls in 0.301 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        3    0.301    0.100    0.301    0.100 {built-in method time.sleep}
        1    0.000    0.000    0.000    0.000 cProfile.py:119(__exit__)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

The only argument to `pr.print_stats` is how you want the output to be sorted, the example above uses `SortKey.CUMULATIVE` - cumulative time, `SortKey` is an `enum` variable, with other commonly useful values are `SortKey.TIME` (total time) and `SortKey.CALLS` (number of calls). To dump all the available values run `[c.name for c in SortKey]` or read the [doc](https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats).

The previous ways will dump all function calls and if your program has hundreds of those it might be too many, so here is how to control the length of the output:

```python
# sleep3.py
import time
import cProfile
from pstats import Stats
def run():
    for _ in range(3):
        time.sleep(0.1)
with cProfile.Profile() as pr:
    run()
stats = Stats(pr)
stats.sort_stats('tottime').print_stats(2)
stats.sort_stats('cumulative').print_stats(4)
```

Let's run:

```bash
$ python sleep3.py
         6 function calls in 0.300 seconds

   Ordered by: internal time
   List reduced from 4 to 2 due to restriction <2>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        3    0.300    0.100    0.300    0.100 {built-in method time.sleep}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


         6 function calls in 0.300 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.300    0.300 /code/users/stas/github/sf/arctic-verl/sleep3.py:5(run)
        3    0.300    0.100    0.300    0.100 {built-in method time.sleep}
        1    0.000    0.000    0.000    0.000 /home/yak/miniconda3/envs/dev/lib/python3.12/cProfile.py:119(__exit__)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

```

This example shows that you can print multiple reports from the same profiler context manager object. Here we print the 2 top lines sorted by total time and 4 top lines sorted by cumulative time.

Cumulative time is useful to understand where some of the large internal time overheads come from - because it shows you the stack of calls leading to the slow call. So `tottime` shows candidates to study and `cumulative` for finding context for those calls. So I often print some 10-20 lines sorted by total time and then immediately a bit longer report with cumulative sorting (I make it as long as needed to catch the function of interest).

The argument to `stats.sort_stats` could be a string name or the `SortKey` enum like in the previous example. The complete table of options is [here](https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats).

#### cProfile Runner

Instead of the context manager you can also use cProfile's runner. This is useful when you have a few versions of functions to compare. For example, let's compare the built-in `**` vs `math.pow`:

```python
# power.py
import cProfile
from pstats import SortKey
import math
def way1():
    for i in range(1000000): x = i**2
def way2():
    for i in range(1000000): x = math.pow(i, 2)
cProfile.run("way1()", sort=SortKey.TIME)
cProfile.run("way2()", sort=SortKey.TIME)
```
Now we run:
```bash
$ python power.py
         4 function calls in 0.026 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.026    0.026    0.026    0.026 power.py:4(way1)
        1    0.000    0.000    0.026    0.026 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.026    0.026 <string>:1(<module>)


         1000004 function calls in 0.169 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  1000000    0.093    0.000    0.093    0.000 {built-in method math.pow}
        1    0.076    0.076    0.169    0.169 power.py:6(way2)
        1    0.000    0.000    0.169    0.169 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.169    0.169 <string>:1(<module>)
```
We can see that the built-in `**` is several times faster than `math.pow` - at least for the given use-case.

If you want to save the output of each profiler to a file instead of printing to `stdout` pass the filename as the 2nd argument:
``` python
cProfile.run("way2()", "cprofile.results", sort=SortKey.TIME)
```
Now when executed the output will be dumped into `cprofile.results`.

Finally, if a function is local cProfile can't see it, so you need to switch to `cProfile.runctx` from `cProfile.run` and pass `locals()` as the third argument:
```python
cProfile.runctx('way1()', None, locals(), sort=-1)
```

The definition of `runctx` is:
```python
cProfile.runctx(command, globals, locals, filename=None, ...)
```
so you can pass `globals` as well.


#### Increasing timing resolution

As you can see the default time resolution is too low, often when working with very fast compute we need higher resolution. `cProfile` doesn't provide a way to change it, but one can hack around it. This monkey patch will increase the resolution to 6 decimals (i.e. report microsecond resolution)

```python
import pstats; pstats.f8 = lambda x: f"{x:3.6f}"
```

In context:

```python
# sleep4.py
import time
import cProfile
import pstats
pstats.f8 = lambda x: f"{x:3.6f}"
def run():
    for _ in range(3):
        time.sleep(0.1)
with cProfile.Profile() as pr:
    run()
pr.print_stats()
```

Run:

```bash
$ python sleep4.py
         6 function calls in 0.300 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1 0.000019 0.000019 0.000040 0.000040 cProfile.py:119(__exit__)
        1 0.000013 0.000013 0.300410 0.300410 sleep4.py:5(run)
        3 0.300397 0.100132 0.300397 0.100132 {built-in method time.sleep}
        1 0.000021 0.000021 0.000021 0.000021 {method 'disable' of '_lsprof.Profiler' objects}
```

In the first example we have seen this in the report, which told us nothing about the actual run time of `run`:

```
      ncalls  tottime  percall  cumtime  percall filename:lineno(function)
           1    0.000    0.000    0.300    0.300 sleep.py:2(run)
```
but now we see it took 13msecs:
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1 0.000013 0.000013 0.300410 0.300410 sleep4.py:5(run)
```

### line_profiler

`line_profiler` is a third-party Python profiler similar to cProfile, but instead of profiling function calls it profiles lines of code, which in some situations might be more practical.

First install it:
```bash
pip install line_profiler
```

Now let's port our `power.py` example from [the cProfile example](#cprofile-runner):

```python
# power2.py
import math
def way1():
    for i in range(1000000):
        x = i**2
def way2():
    for i in range(1000000):
        x = math.pow(i, 2)
profile(way1)()
profile(way2)()
```
Please note we aren't importing `profile` - the special launcher `kernprof` will take care of it:

```bash
$ kernprof -l -v power2.py
Wrote profile results to 'power2.py.lprof'
Timer unit: 1e-06 s

Total time: 0.334813 s
File: power2.py
Function: way1 at line 2

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     2                                           def way1():
     3   1000001     164213.1      0.2     49.0      for i in range(1000000):
     4   1000000     170599.7      0.2     51.0          x = i**2

Total time: 0.361192 s
File: power2.py
Function: way2 at line 5

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     5                                           def way2():
     6   1000001     166696.8      0.2     46.2      for i in range(1000000):
     7   1000000     194495.1      0.2     53.8          x = math.pow(i, 2)
```

As you can see it now reports timing per each line of code.

Contrary to the cProfile example this gives us an inconclusive result, probably because the profiling overhead was too big here. So let's get a 3rd opinion and use `timeit`:

```python
# power3.py
import timeit
import math
def way1():
    for i in range(1000000): x = i**2
def way2():
    for i in range(1000000): x = math.pow(i, 2)
print(f'way1={timeit.Timer("way1()", globals=globals()).timeit(number=1)}')
print(f'way2={timeit.Timer("way2()", globals=globals()).timeit(number=1)}')
```

```bash
$ python power3.py
way1=0.031605484895408154
way2=0.06159617658704519
```

Which gives us yet another outcome, here `math.pow` is only about 2x slower than `**`.

One thing to observe here is that 3 different measurement approaches give very different results. Since what usually matters is the relative performance of one approach versus another, the exact method probably doesn't matter as long as you consistently use the same one.
