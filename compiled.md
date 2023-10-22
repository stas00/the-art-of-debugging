# Debugging Compiled Programs

XXX: Please ignore this chapter for now - I'm working on writing it out


In order to successfully debug C/C++ programs you don't need to understand how to write or compile them.

I will briefly show how these are done, but you can safely skip over if you can't follow and just run the commands as is, as we need to build them to emulate problems you're likely to encounter when attempting to use programs written and compiled by others.

Also it's important to understand that when you use interpreter languages like Python, you're still likely to run into C/C++ issues if they use C/C++ extensions. For example, while PyTorch will give you a Python traceback most of the time, there will be situations where a CUDA kernel or some other C++ extension is run and that's when you need to know how to diagnose these issues.


## Super-fast introduction to gdb



run a pytest via gdb (when getting a segfault)
```
gdb -ex r --args python -m pytest -sv tests/test_failing.py
```

then when it segfaults hit `c`+Enter, then run `bt` and `c`+Enter

more info and tricks [here](https://wiki.python.org/moin/DebuggingWithGdb).

another way:
```
gdb python
> run /home/stas/anaconda3/envs/py38-pt18/bin/pytest tests/test_trainer.py
```
if needing to catch a throw and get a bt do:
```
> catch throw
> run ...
> bt
```
if the process is hanging then attach to it in another shell
```
sudo gdb --pid=107903
thread apply all bt
bt
```

use case:


## shared libraries, nm, symbols, ldd, LD_LIBRARY_PATH

## Super-fast introduction to shared libraries

 We will start our work in `dl1` subdir:
```
cd dl1
```

Let's build a simple shared library.
```
gcc -fPIC -c util.c
gcc -shared -o libmyutil.so util.o
```
Next build the executable against the shared library we have just built:
```
gcc dl1.c -L. -lmyutil -o dl1
```


Try running it:

```
$ ./dl1
./dl1: error while loading shared libraries: libmyutil.so: cannot open shared object file: No such file or directory
```

Let's introduce `ldd` - this is a tool that prints shared object dependencies

So let's check what dependencies are missing:
```
$ ldd dl1
        linux-vdso.so.1 (0x00007ffcebff8000)
        libmyutil.so => not found
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fca8a200000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fca8a5c7000)
```

Aha, as you can see it can't find `libmyutil.so`. even though it's found in the same folder as we have just built it.

Let's introduce a special environment variable `LD_LIBRARY_PATH` which contains a `:`-separated list of paths where shared libraries are searched. If you're already familiar with the environment variable `PATH`, this is exactly the same but instead of searching for executable files it'll search for libraries.

Let's deploy the following fix to set `LD_LIBRARY_PATH` to the current path, where `libmyutil.so` can be found:
```
$ LD_LIBRARY_PATH=. ./dl1
Inside main()
Inside util_a()
```

And now the program was able to run.

When using the approach of setting the environment variable or several of them with the command being executed:
```
ENV1=foo ENV2=bar ./my_program
```
only that program will see this exact setting. When that program exits other programs will see the value of these environment variables as they were before the last run.

An alternative solution is to `export` this environment variable instead. In which case all future programs executed from this shell will see this new environment variable value. Here is how you do it:

```
export LD_LIBRARY_PATH=.
```
Now you can just run:
```
./dl1
```

as `LD_LIBRARY_PATH` could already be non-empty, usually you might want to use this strategy instead - which extends the original value:

```
export LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH"
```

Depending on whether you prepend or append the additional path to search for libraries, it'll be searched first or last correspondingly.

To check the current value
```
echo $LD_LIBRARY_PATH
```
This understanding is crucial since you may have multiple versions of the same library installed on your system, and you need to know which one of them gets loaded.

To keep things tidy and not end up with having the same path added multiple times, here is a helper function, which will only prepend the path to `LD_LIBRARY_PATH` if it isn't already there

```
function add_to_LD_LIBRARY_PATH {
    case ":$LD_LIBRARY_PATH:" in
        *":$1:"*) :;; # already there
        *) LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH";; # or LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$1"
    esac
}

add_to_LD_LIBRARY_PATH .
```
This is useful for when you script things, as you never know what was called before your script was run.

(XXX: is it already exported?)



XXX: stopped here:
build a shared library

```
gcc -fPIC -c util2.c
gcc -shared -o libutil.so.2 util2.o
```


build the executable against the shared library

```
gcc dl2.c -L. -lutil -o dl2
```


XXX: LD_PRELOAD
LD_PRELOAD=./libutil.so ldd ./dl1


## nm

Use case: Let's follow my bug report [undefined symbol curandCreateGenerator for torch extensions](https://github.com/pytorch/pytorch/issues/69666)

The problem was:
```
 ImportError: ~/.cache/torch_extensions/py38_cu111/cpu_adam/cpu_adam.so:
 undefined symbol: curandCreateGenerator
```

the symbol is undefined with either build version, but I think it should get resolved at loading time with linked cuda libraries.

```
$ nm ~/.cache/torch_extensions/py38_cu113/cpu_adam/cpu_adam.so | grep curandCreateGenerator
                 U curandCreateGenerator
```

so this tells me some cuda library is missing.
but it needs to be:
```
                 U curandCreateGenerator@@libcurand.so.10
```
ldd output is the same with either package.
```
$ ldd ~/.cache/torch_extensions/py38_cu113/cpu_adam/cpu_adam.so
        linux-vdso.so.1 (0x00007ffe328dd000)
        libgtk3-nocsd.so.0 => /usr/lib/x86_64-linux-gnu/libgtk3-nocsd.so.0 (0x00007f4cb1daf000)
        libc10.so => not found
        libc10_cuda.so => not found
        libtorch_cpu.so => not found
        libtorch_python.so => not found
        libcudart.so.11.0 => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.0 (0x00007f4cb1b0d000)
        libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f4cb1929000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f4cb17da000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f4cb17bf000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f4cb15cd000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f4cb15c7000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f4cb15a4000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f4cb211e000)
        librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f4cb1597000)
```

Adding
```
def extra_ldflags(self):
    return ['-lcurand']
```

Solves the problem for JIT (ninja build). Check!

However if it's a prebuilding and not JIT, extra_ldflags is getting ignored.

Here is the g++ linker command line:
```
g++ -pthread -shared -B /home/stas/anaconda3/envs/py38-pt111/compiler_compat
-L/home/stas/anaconda3/envs/py38-pt111/lib -Wl,-rpath=/home/stas/anaconda3/envs/py38-pt111/lib
-Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.8/csrc/adam/cpu_adam.o
build/temp.linux-x86_64-3.8/csrc/common/custom_cuda_kernel.o
-L/home/stas/anaconda3/envs/py38-pt111/lib/python3.8/site-packages/torch/lib
-L/usr/local/cuda-11.5/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda
-ltorch_cuda_cu -ltorch_cuda_cpp -o
build/lib.linux-x86_64-3.8/deepspeed/ops/adam/cpu_adam_op.cpython-38-x86_64-linux-gnu.so
```

If I manually added -lcurand to it, then all gets resolved. but this has to happen automatically of course.


## ldd

## strace
