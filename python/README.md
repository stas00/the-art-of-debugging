# Debugging Python Programs




## auto-print what's being observed

Say, you want to see the output of summation:

```
$ python -c 'x=5; y=6; print(f"{x+y}")'
11
```
But then you don't know what's being summed in the printout, so you have to write then:

```
$ python -c 'x=5; y=6; print(f"x+y={x+y}")'
x+y=11
```
But this is both long and error-prone, because the 2 parts aren't atomic - you may choose to modify the expression in the second part `{x+y+1}`, but forget to update the first part and end up with wrong conclusions.

Since python-3.8 there is an atomic operand auto-description feature. Let's rewrite the last one liner to remove the description of what's being printed and append a magical `=` to the expression inside `{}`:
```
$ python -c 'x=5; y=6; print(f"{x+y=}")'
x+y=11
```
Now what's being evaluated is automatically printed. All you need to do is to add `=` before the closing `}`.

Here is another example:
```
$ python -c 'x=5; y=6; print(f"{x=}, {y=}")'
x=5, y=6
```

So once again you can see that atomic operations are ideal for fruitful debugging.



## Debugging the right Python package

If you're modifying a git repository of a Python package and then installing into a virtual environment this could be a very error-prone process since you are never sure if you reinstalled the modified files or not when you test things. And that's why instead of using `pip install .` it's much better to use `pip install -e .` which instead of installing the Python files into the virtual environment, tells the latter to access the files from the git clone (or whatever other way the source code is made available). For example, if you're working on HF transformers, here is how to do it better:
```
git clone https://github.com/huggingface/transformers/
cd transformers
pip install -e .
python my-program.py
```
now you can tweak the files inside the git clone and they will be automatically used at run time.

Yet another approach is with the help of `PYTHONPATH`. In this approach you don't install the package into the virtual environment, but instead you tell `python` to load it directly from its source. Repeating the last example, it would change into:
```
git clone https://github.com/huggingface/transformers/
cd transformers
PYTHONPATH=src python my-program.py
```
If you compare before and after:
1. there is no `pip install`
2. a new environment variable `PYTHONPATH=src` is added before the program is run and it's active only for the duration of the program.

we used `src` because that's where source files of HF `transformers` package reside , that is they are under `[src/](https://github.com/huggingface/transformers/tree/main/src/). Other git repos could use a different path where they place their modules, and actually it's even more common to not have any prefix at all. If there is no profix then use `PYTHONPATH=.` which means the current directory is where Python can find the packages at.

Why would you want to do that? If you're like me, I have multiple clones of the same repo, e.g.:

```
transformers-pr-18998
transformers-project-a
transformers-problem-b
```
And I work in parallel on multiple features, where each clone uses a different branch. And most of the time I use the same conda environment for all of them. Thus when I develop things I might use `pip install -e .` but if I have to go back and force between different projects frequently I purposefully uninstall `pip uninstall transformers` to ensure that some unrelated version gets loaded and instead I use `PYTHONPATH` at run time. For example, I start with:
```
cd transformers-pr-18998
PYTHONPATH=`pwd`/src python myprogram
```

and possibly in another terminal I'll run:

```
cd transformers-project-a
PYTHONPATH=`pwd`/src python myotherprogram
```
and each program will see only the files from the right branch.

If `PYTHONPATH` was somehow already non-empty (rare), like any other `PATH`-type environment variables you can prefix to it like so:
```
PYTHONPATH=`pwd`/src:$PYTHONPATH python ...

```
The order is critical - the earlier paths will have a higher precedence than later ones.

Since it's easy to use an invalid path at `PYTHONPATH` and there will be no error as long as Python will find a version of the files you try to load, there are 2 ways to ensure that the correct libraries are loaded:

1. You can dump `sys.path` which `PYTHONPATH` updates:
```
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

2. You can use the [purposefully break the script approach](../methodology#am-i-editing-the-right-file-and-the-right-class) to validate that the files you're editing are the files that actually get loaded.


## Setting up your test suite to always use the git repo's Python packages

If your test suite relies on the package it tests being preinstalled you are likely to be testing the wrong files. This is usually less of a problem when the git package places the Python packages at the root directory of the repo, but when a project is structured like HF `transformers` where Python packages are placed under `src` or another top-level subdir Python will not find these packages. Also if you launch the tests not from the repo's root directory it'll always fail to find your repo's packages.

There is an easy solution to that. You can see how I did it in [ipyexperiments](https://github.com/stas00/ipyexperiments). I created [tests/conftest.py](https://github.com/stas00/ipyexperiments/blob/39c9b454e89e53b74c74dcb579c12ecf3d2161b9/tests/conftest.py#L7-L12) which contains:
```
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
```
from pathlib import Path
# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(1, git_repo_path)
```

That's it, now your tests will always have the repo path appear first in `sys.path`.

footnote: you can choose to insert into position `0` instead of `1` as well, but I like to keep Python's current directory as the first one to look for files.

If your project uses a sub-directory like HF `transformers`' `src`, simply add it to the `git_repo_path`, e.g. you can see how this is done [here](https://github.com/huggingface/transformers/blob/6cbc1369a330860c128a1ba365f246751382c9e5/conftest.py#L30-L31)

As you are figuring it out simply dump:
```
print("\n".join(sys.path))
```
and check that the paths are correct.


## py-spy

`py-spy` is a great tool for diagnosing processes that either hang or spin out of control or there is an issue with a network connection, blocked IO, etc. It's similar to getting a traceback on exception, except the process is still running.

First do `pip install py-spy`.

Now you can attach to each process with:

```
py-spy dump -n -p PID
```
and it will tell you where the process is

- `PID` is the process id of the hanging python process.
- `-n` is useful if you want to see strack traces from python extensions written in C, C++, etc., as the program may hang in one of the extensions
- you may need to add `sudo` before the command - for more details see [this note](https://github.com/benfred/py-spy#when-do-you-need-to-run-as-sudo).

If you have no `sudo` access your sysadmin might be able to perform this for you:
```
sudo echo 0 > /proc/sys/kernel/yama/ptrace_scope
```
which will allow you running `py-spy` (and `strace`) without needing `sudo`. Beware of the possible [security implications](https://wiki.ubuntu.com/SecurityTeam/Roadmap/KernelHardening#ptrace_Protection) - but typically if your compute node is inaccessible from the Internet it's less likely to be a risk.

To make this change permanent edit `/etc/sysctl.d/10-ptrace.conf` and set:
```
kernel.yama.ptrace_scope = 0
```

Here is an example of `py-spy dump` python stack trace:
```
Thread 835995 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:1191)
    _aggregate_total_loss (deepspeed/runtime/pipe/engine.py:540)
    train_batch (deepspeed/runtime/pipe/engine.py:330)
    train_step (megatron/training.py:436)
    train (megatron/training.py:851)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```
The very first line is where the program is stuck.

If the hanging happens inside a CPP extension, add `--native` `py-spy` and it'll show the non-python code if any.

If the process has multiple threads it'll show a stack trace of each thread. For example:

```
Thread 0x7F6D3C29D740 (idle): "MainThread"
    wait (threading.py:312)
    result (concurrent/futures/_base.py:435)
    main (slurmeventd.py:208)
    <module> (slurmeventd.py:217)
Thread 0x7F6CF5FFB700 (idle): "Thread-CallbackRequestDispatcher"
    wait (threading.py:312)
    get (queue.py:171)
    _get_many (pubsub_v1/subscriber/_protocol/helper_threads.py:56)
    __call__ (pubsub_v1/subscriber/_protocol/helper_threads.py:103)
    run (threading.py:892)
    _bootstrap_inner (threading.py:954)
    _bootstrap (threading.py:912)
```

`MainThread` is the main process.

To run it on multiple processes at once:

```
pgrep python | xargs -I {} py-spy dump --pid {}
```

If you want only subprocesses, e.g. to skip the launcher process:

```
pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
```

You can also read about how to run it on multiple nodes [here](https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-hanging-solutions.md#py-spy).
