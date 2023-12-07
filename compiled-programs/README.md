# Debugging Compiled Programs

This chapter will empower you to deal with errors like:

```
Segmentation fault (core dumped)
```

```
ImportError: ~/.cache/torch_extensions/py38_cu111/cpu_adam/cpu_adam.so:
 undefined symbol: curandCreateGenerator
```

```
./dl1: error while loading shared libraries:
libmyutil.so: cannot open shared object file: No such file or directory
```

It's going to arm you with knowledge that will cover shared libraries,  unresolved symbols, `nm`, `ldd`, `LD_LIBRARY_PATH`, `LD_PRELOAD`, `gdb`, etc., using simple demonstrations that you can follow along and use cases from github Issues.

## Introduction

In order to successfully debug C/C++ programs you don't need to understand how to write or compile them.

I will briefly show how these are done, but you can safely skip over if you can't follow and just run the commands as is, as we need to build them to emulate problems you're likely to encounter when attempting to use programs written and compiled by others.

Also it's important to understand that when you use interpreter languages like Python, you're still likely to run into C/C++ issues if they use C/C++ extensions. For example, while PyTorch will give you a Python traceback most of the time, there will be situations where a CUDA kernel or some other C++ extension is run and that's when you need to know how to diagnose these issues.


## Following along

You can just read this chapter without doing anything, but if you want to get your hands dirty so to speak, here is what you need to do first to access the demo files:

```
git clone https://github.com/stas00/the-art-of-debugging
cd the-art-of-debugging
```

Each section will assume you're in the `the-art-of-debugging` directory when it tells you to `cd` somewhere.

## Segmentation fault, core files and gdb

First, we are going to create a program that leads to a segmentation fault and generates us a simple `core` file that we could use for the demonstration purposes.

We will use the program located at the sub-dir [`segfault`](./segfault):
```
cd compiled-programs/segfault
```

There could be dozens of reasons for why a C/C++ program may crash with `Segmentation fault`. We want something super-simpler, so the following C program [make-segfault](./segfault/make-segfault.c) dereferences an uninitialized pointer which leads to a Segmentation fault:
```
#include <stdio.h>

int main(void)
{
    int* ptr;
    printf("%d", *ptr); // dereference an uninitialized pointer
    return (0);
}
```

Let's first compile the program
```
gcc -g make-segfault.c -o make-segfault
```

`-g` is for enabling debug.

Now, let's run the program:
```
./make-segfault
Segmentation fault
```

This situation is impossible to diagnose since the core file hasn't been dumped.

You want the core file, since most of the it contains everything you need to understand why the program crashed.

### Get the core file dumped

By default your Bash shell might be configured not to dump core files beyond a certain size, so we need to tell it to allow any core file sizes with:
```
ulimit -c unlimited
```
This change is only effective in the shell you run it in.

footnote: for more details see: `man bash` and then search for `ulimit`

footnote: if you use a different shell, check its manpage for its specific way of doing the same - search for `core file` or something similar.

Now, let's run the program:
```
./make-segfault
Segmentation fault (core dumped)
```

So this time we can see the program did dump the core, but in my case `ls -l` shows no core file. This usually means that the system is configured to send the core file to some other subsystem (e.g. to `apport` if you're on Ubuntu) or a specific path.

So we need to get the control back and tell the kernel where to save the core files and which format to use.

Let's first check where the core files are currently sent:
```
sysctl kernel.core_pattern
kernel.core_pattern = |/usr/share/apport/apport %p %s %c %d %P %E
```
so indeed you can see that core files are sent to `apport`. I'm going to override this setting with:
```
sudo sysctl -w kernel.core_pattern=/tmp/core-%e.%p.%h.%t
```
will save the core files under `/tmp/` using the name of the program, followed by the process id, then the hostname and finally the time stamp of when it was run. So let's now re-run the program again:

```
$ ./make-segfault
Segmentation fault (core dumped)
$ ls -lt /tmp/core*
-rw------- 1 stas stas 304K Oct 22 20:10 core-make-segfault.150885.hope.1698030657
```
And voila we have the core file.

footnote: as `apport` is a security measure on Ubuntu you may want to restore the `core_pattern` value back to its original `apport` setting.


To see the current setting of `kernel.core_pattern` execute:
```
/sbin/sysctl kernel.core_pattern
```
If you aren't on Ubuntu, chances are that your system is set to just `core`:
```
$ /sbin/sysctl kernel.core_pattern
kernel.core_pattern = core
```
which means it'll just save the `core` file in the current directory without any useful tags.

If you want the useful tags and have it saved in the current directory, run:
```
sudo sysctl -w kernel.core_pattern=core-%e.%p.%h.%t
```

footnote: if you don't have `sudo` access lookup valgrind which sometimes helps with this situation or alternatively ask your sysadmin to provide you a way to get the core files dumped.



### Get the backtrace from the core file

To inspect the core file, with invoke `gdb` with the path to the core file and the path to the program that created it and when it loaded we run the following command: `bt`:
```
gdb -c /tmp/core-make-segfault.150885.hope.1698030657 make-segfault
[...]
Core was generated by `./make-segfault'.
Program terminated with signal SIGSEGV, Segmentation fault.
#0  0x000056433301e159 in main () at make-segfault.c:6
6           printf("%d", *ptr);
(gdb) bt
#0  0x000056433301e159 in main () at make-segfault.c:6
```

As you can see, indeed, the problem is at line 6, where the dereference `*ptr` is happening as `ptr` hasn't been initialized.

Actually, you can run `bt full` to get more information:
```
(gdb) bt full
#0  0x000056433301e159 in main () at make-segfault.c:6
        ptr = 0x0
```
so it's even more clear that `ptr` hasn't been set (it must not be `0x0`).

This was a super-simple program, but most modern programs will run multiple threads, so in order to see the stack trace of each thread at the moment of the Segmentation fault, run one of these commands:
```
(gdb) thread apply all bt
(gdb) thread apply all bt full
```

case study: [this issue](https://github.com/pytorch/pytorch/issues/59384#issuecomment-854953165) shows a huge traceback from pytorch crashing.


### Get the backtrace from the still running process

There is also a way to attach to an already running process:
```
sudo gdb --pid=107903
thread apply all bt
bt
```
Modify the process id to match the process that you want to debug.

This approach is very useful if the process is hanging or seems to be spinning consuming a lot of CPU power.

case study: see this [Issue](https://github.com/pytorch/pytorch/issues/60158#issuecomment-865142029) for an example of how this approach was used to diagnose a deadlock in pytorch. This is also a good example of seeing a backtrace for multiple threads in a real application.


### Abort the program while dumping a core file

And you can even force a core dump, by running either of these 2 commands (edit the pid):
```
gcore 107903
kill -ABRT 107903
```
this will also kill the program. This is again could be useful if the program is out of control and you want to make sure you saved the core file which you can analyse later, while freeing the resources.

footnote: `strace -p 107903` can be also useful for seeing where the process is stuck, but if it's some serious problem your `strace` could get stuck as well. For example, the latter problem can happen if a process tied to a GPU which stopped functioning and doesn't respond.

### Run the program under gdb

Since `gdb` is a debugger you can of course launch the program via `gdb` and step through it until you hit the Segmentation fault. Let's use our example program again:
```
$ gdb ./make-segfault
[...]
Reading symbols from ./make-segfault...
(gdb) run
Starting program: /the-art-of-debugging/compiled-programs/segfault/make-segfault
[...]
Program received signal SIGSEGV, Segmentation fault.
0x0000555555555159 in main () at make-segfault.c:6
6           printf("%d", *ptr); // dereference an uninitialized pointer
```

The advantage of this approach is that you don't need the core file. This is useful for example if you have no `sudo` access and the system isn't set up to allow core files dumped or it traps them by another program.

Like any other debugger you can also set breakpoints:
```
$ gdb ./make-segfault
[...]
(gdb) b make-segfault.c:6
Breakpoint 1 at 0x1155: file make-segfault.c, line 6.
(gdb) run
Starting program: /the-art-of-debugging/compiled-programs/segfault/make-segfault
[...]
Breakpoint 1, main () at make-segfault.c:6
6           printf("%d", *ptr); // dereference an uninitialized pointer
(gdb) n

Program received signal SIGSEGV, Segmentation fault.
0x0000555555555159 in main () at make-segfault.c:6
6           printf("%d", *ptr); // dereference an uninitialized pointer
```
Here we used:
1. `b make-segfault.c:6` to set a breakpoint at line 6, where our bug is.
2. `run` executed the program and stopped it just before the breakpoint
3. `n` executed the buggy line 6 and we got Segmentation fault

footnote: run `man gdb` for more information on `gdb`

case study: Sometimes I have one of the tests running under `pytest` segfault. In such a case I run `pytest` via gdb:
```
gdb -ex r --args python -m pytest -sv tests/test_failing.py
```
then when it segfaults hit `c`+Enter, then run `bt` and `c`+Enter.

In this use case, we launched the python program via `--args`, and `-ex` told `gdb` to run `r` which is the shortcut for `run`.

Here is another way to do the same:
```
gdb python
> run /home/stas/anaconda3/envs/py38-pt18/bin/pytest tests/test_failing.py
```
the difference here is that you need to pass the full path to the program.

`gdb` is super powerful and can do many marvellous things but most of these are out of the scope of this basic introduction. It is very likely that if you understood everything covered so far you should be able to diagnose 95% of most problems you are likely to encounter.

case study: [this issue](https://github.com/pytorch/pytorch/issues/46807#issuecomment-718452462) shows how gdb was used to get a backtrace on a crashing test.

footnote: more useful recipes for python + gdb can be found [here](https://wiki.python.org/moin/DebuggingWithGdb).



## shared libraries, ld.so.conf, nm, unresolved symbols, ldd, LD_LIBRARY_PATH, LD_PRELOAD,

### introduction to shared libraries

Operating systems use a modular approach to reusable components, which are called shared libraries. On Unix the majority of these libraries are found under `/usr/lib` and additionally at various other places.

The way the OS kernel finds these libraries is by configuring which paths it should search when an application requests a use of a specific library. Typically this is done either via a single file `/etc/ld.so.conf` or more recently via a folder `/etc/ld.so.conf.d/` which contains multiple `.conf` files. Let's see a few examples coming from a recent Ubuntu box:

```
$ cat /etc/ld.so.conf.d/libc.conf
# libc default configuration
/usr/local/lib

$ cat /etc/ld.so.conf.d/i386-linux-gnu.conf
# Multiarch support
/usr/local/lib/i386-linux-gnu
/lib/i386-linux-gnu
/usr/lib/i386-linux-gnu
/usr/local/lib/i686-linux-gnu
/lib/i686-linux-gnu
/usr/lib/i686-linux-gnu
```

So this tells us that `/usr/local/lib`, `/usr/local/lib/i386-linux-gnu`, etc., will be searched.

Whenever a new set of shared libraries is installed or these config files are updated one needs to run:

```
sudo ldconfig
```
which updates `/etc/ld.so.cache` with all the libraries that can be found.

The most commonly used library is typically `libc` and we can see various versions of it:
```
ls -l /lib/x86_64-linux-gnu/libc.*
-rw-r--r-- 1 root root 5.8M Sep 25 07:45 /lib/x86_64-linux-gnu/libc.a
-rw-r--r-- 1 root root  283 Sep 25 07:45 /lib/x86_64-linux-gnu/libc.so
-rwxr-xr-x 1 root root 2.2M Sep 25 07:45 /lib/x86_64-linux-gnu/libc.so.6*
```

Here:
- `libc.so.6` is the shared library from the 6th generation of `libc` (could be higher or lower)
- `libc.so` is a special kind of a linker script used during application building and in this case it's not a shared library
- `libc.a` is a static object that can be used to "bundle" the contents of `libc` with the application rather than loading it run time.

footnote: on a different system `libc.*` could be found at a different location, but its path will be in the `ld.so.conf` cache.

Any application can be linked statically, e.g. using `libc.a`, but it'll lead to a much larger size and waste of CPU memory. Instead, most application link to the shared `libc.so.6` library, so that it's loaded into the CPU memory once and then dozens of applications all enjoy sharing this single copy of the library in the CPU memory.

We will discuss `ldd` shortly but this is how you discover which shared libraries an application is linked to:
```
$ ldd /bin/ls
        linux-vdso.so.1 (0x00007fff24394000)
        libselinux.so.1 => /lib/x86_64-linux-gnu/libselinux.so.1 (0x00007f20c6da0000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f20c6b78000)
        libpcre2-8.so.0 => /lib/x86_64-linux-gnu/libpcre2-8.so.0 (0x00007f20c6ae1000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f20c6e2a000)
```

So you can see that `/bin/ls` is linked against `libc.so.6` and the library resolver found the library location at `/lib/x86_64-linux-gnu/libc.so.6`.

In the following sections you will discover other ways to add shared libraries dynamically at runtime.

Now we are going to build two versions of a simple shared library and use those to understand and solve various issues around shared libraries and their symbols.



### Build a first version of `libmyutil.so`

We will start our work in the `dl1` subdir:
```
cd compiled-programs/dl1
```

Here is the source of a simple shared library [util.c](./dl1/util.c):
```
#include <stdio.h>

int util_a()
{
    printf("Inside util_a()\n");
    return 0;
}
```
and a simple program to drive it [dl1.c](./dl1/dl1.c):
```
#include <stdio.h>

extern void util_a();
int main()
{
    printf("Inside main()\n");
    util_a();

    return 0;
}
```

Let's build a simple shared library `libmyutil.so`:
```
gcc -g -fPIC -c util.c
gcc -g -shared -o libmyutil.so util.o
```
Next build the program against the shared library we have just built:
```
gcc dl1.c -L. -lmyutil -o dl1
```

Note that
- the prefix `lib` of `libmyutil.so` is removed when setting `-lmyutil`, which explains why all shared libraries start with `lib`.
-`-L.` tells `gcc` to additionally search the current directory for shared libraries.

Let's try running this newly build application:
```
$ ./dl1
./dl1: error while loading shared libraries: libmyutil.so: cannot open shared object file: No such file or directory
```

That didn't work. We need help to understand what the problem is:

### ldd

Let's introduce `ldd` - this is a tool that prints shared object dependencies

So let's check what dependencies are missing:
```
$ ldd dl1
        linux-vdso.so.1 (0x00007ffcebff8000)
        libmyutil.so => not found
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fca8a200000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fca8a5c7000)
```

As you can see it can't find `libmyutil.so` even though it's located in the same folder as we have just built it:
```
$ ls -1
dl1
dl1.c
libmyutil.so
util.c
util.o
```

footnote: run `man ldd` for more information on `ldd`


### LD_LIBRARY_PATH

Let's introduce a special environment variable `LD_LIBRARY_PATH` which contains a `:`-separated list of paths where additional shared libraries are searched. If you're already familiar with the environment variable `PATH`, this is exactly the same, but instead of searching for executable files it'll search for shared libraries.

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
$ ./dl1
Inside main()
Inside util_a()
```

as `LD_LIBRARY_PATH` could already be non-empty, usually you might want to use the following strategy instead:

```
export LD_LIBRARY_PATH=".:$LD_LIBRARY_PATH"
```
As you can see it extends the original value of `LD_LIBRARY_PATH`.

Depending on whether you prepend or append the additional path to search for libraries, it'll be searched first or last correspondingly.

To check the current value of `LD_LIBRARY_PATH` do:
```
echo $LD_LIBRARY_PATH
```

The ordering understanding is crucial since you may have multiple versions of the same library installed on your system, and you need to know which one of them gets loaded.

To keep things tidy and not end up with having the same path added multiple times, here is a helper function, which will only prepend the path to `LD_LIBRARY_PATH` if it isn't already there:
```
function add_to_LD_LIBRARY_PATH {
    case ":$LD_LIBRARY_PATH:" in
        *":$1:"*) :;; # already there
        *) LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH";; # or LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$1"
    esac
}
```

Now to add `.` to it, just run:
```
add_to_LD_LIBRARY_PATH .
```
This is useful for when you script things, as you never know what was set before your script was run. I use this functionality in my `~/.bash_profile`.

footnote: run `man ld.so` for more information on `LD_LIBRARY_PATH`


### Build a second version of `libmyutil.so`

Next let's extend the original `libmyutil.so` library with a new function. To accomplish that we will start our work in `dl2` subdir which is very similar to `dl1` but we are going to extend our library to support a new function `util_b`:
```
cd compiled-programs/dl2
```

Here is the source of a simple shared library [util.c](./dl2/util.c):
```
#include <stdio.h>

int util_a()
{
    printf("Inside util_a()\n");
    return 0;
}

int util_b()
{
    printf("Inside util_b()\n");
    return 0;
}
```
and a simple program to drive it [dl2.c](./dl2/dl2.c):
```
#include <stdio.h>

extern void util_a();
extern void util_b();

int main()
{
    printf("Inside main()\n");
    util_a();
    util_b();
    return 0;
}
```

As you can see these 2 .c files are almost the same, except we added a simple `util_b` function and called it from the program.

Let's build the shared library:
```
gcc -fPIC -c util.c
gcc -shared -o libmyutil.so util.o
```

Now build the executable against the shared library we have just created:
```
gcc dl2.c -L. -lmyutil -o dl2
```

We already know that we need to tell `LD_LIBRARY_PATH` where to find this new shared library, so we call:
```
$ LD_LIBRARY_PATH=. ./dl2
Inside main()
Inside util_a()
Inside util_b()
```
And all is good.

As you remember we built an earlier version of `libmyutil.so` in `../dl1` - so let's see what happens if we try to run `dl2` while loading the older `libmyutil.so` from `../dl1`:
```
$ LD_LIBRARY_PATH=../dl1 ./dl2
./dl2: symbol lookup error: ./dl2: undefined symbol: util_b
```

As you recall `libmyutil.so` in `../dl1` only had `util_a` defined. And `util_b` was added in the 2nd version of `libmyutil.so` defined in `dl2`.

This is trivial to see when you have the source code and a tiny program, but let's say this is not the case.

### nm

Let's introduce `nm`, which lists symbols from object files.

We know our `dl2.c` has `main` in it, so let's check if it knows that symbol:
```
$ nm ./dl2 | grep main
                 U __libc_start_main@GLIBC_2.34
0000000000001189 T main
```
we can see that the `main` symbol is defined and `0000000000001189` is the address of its location in `./dl2`.

And you can also see `__libc_start_main@GLIBC_2.34` which is not defined. So let's see where we can find its definition. We see that it tells us that it's part of `libc`, so let's find the path of `libc`:
```
$ ldd ./dl2
        linux-vdso.so.1 (0x00007fffeaacd000)
        /the-art-of-debugging/c/dl2/libmyutil.so (0x00007fdd5a150000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fdd59ef0000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fdd5a15c000)

```
and since we can see it's at `/lib/x86_64-linux-gnu/libc.so.6` we can check its symbols:
```
$ nm /lib/x86_64-linux-gnu/libc.so.6
nm: /lib/x86_64-linux-gnu/libc.so.6: no symbols
```
Well, that didn't work. Let's make `nm` work harder and check the dynamic symbol table instead by adding `-D`:
```
$ nm -D /lib/x86_64-linux-gnu/libc.so.6 | grep __libc_start_main
0000000000029dc0 T __libc_start_main@@GLIBC_2.34
0000000000029dc0 T __libc_start_main@GLIBC_2.2.5
```
and voila, we can see `__libc_start_main@GLIBC_2.34` is indeed found in the `libc` library.

Since at runtime `./dl2` will load `/lib/x86_64-linux-gnu/libc.so.6` these symbols will get dynamically resolved.

Let's now go back to our issue with:
```
$ LD_LIBRARY_PATH=../dl1 ./dl2
./dl2: symbol lookup error: ./dl2: undefined symbol: util_b
```

Which symbols does `./dl2` refers to:
```
$ nm ./dl2 | grep util
                 U util_a
                 U util_b
```

So `nm` tells us that `dl2` contains references to `util_a` and `util_b` which aren't resolved.

We already know that we have 2 libraries with the same name `libmyutil.so` so we can check which symbols each of them has:

```
$ nm ../dl1/libmyutil.so | grep util
0000000000001119 T util_a
$ nm ../dl2/libmyutil.so | grep util
0000000000001119 T util_a
0000000000001134 T util_b
```
and then we can see that `../dl1/libmyutil.so` contains the actual location of the symbol `util_a` with its address, and `../dl2/libmyutil.so` contains both  `util_a` and `util_b` and their addresses.

So now you know why if the wrong shared library is loaded, some symbols might be missing, even though it's the same library name.

footnote: run `man nm` for more information on `nm`

Moreover, in the libraries distributed via package managers, each shared library is likely to have multiple versions, so that applications relying on different versions of the same library could all continue working by simply requiring multiple versions of the same library to be installed. Typically, the versioning is done by appending the major, minor and patch versions numbers in the shared library file names.

For example, I found I have 2 major versions of `libcrypto+.so` installed:
```
/usr/lib/x86_64-linux-gnu/libcrypto++.so.6
/usr/lib/x86_64-linux-gnu/libcrypto++.so.6.0.0
/usr/lib/x86_64-linux-gnu/libcrypto++.so.8
/usr/lib/x86_64-linux-gnu/libcrypto++.so.8.6.0
```

footnote: `8.6.0` version is deciphered as major=8, minor=6, patch=0

If I list those libraries:
```
ls - l/usr/lib/x86_64-linux-gnu/libcrypto++*
lrwxrwxrwx 1 root root   20 Mar 24  2020 /usr/lib/x86_64-linux-gnu/libcrypto++.so.6 -> libcrypto++.so.6.0.0
-rw-r--r-- 1 root root 3.8M Mar 24  2020 /usr/lib/x86_64-linux-gnu/libcrypto++.so.6.0.0
lrwxrwxrwx 1 root root   20 Dec 19  2021 /usr/lib/x86_64-linux-gnu/libcrypto++.so.8 -> libcrypto++.so.8.6.0
-rw-r--r-- 1 root root 4.1M Dec 19  2021 /usr/lib/x86_64-linux-gnu/libcrypto++.so.8.6.0
```
you can also see that each major `.so` version is a symlink to the full `major.minor.patch` version (e.g. `6` -> `6.0.0`).

Yet, at other times the same library with the same version can be found in various directories:
```
$ find /usr/local/cuda* | grep libnvvm\.so | sort
/usr/local/cuda-11.7/nvvm/lib64/libnvvm.so.4
/usr/local/cuda-11.7/nvvm/lib64/libnvvm.so.4.0.0
/usr/local/cuda-12.2/nvvm/lib64/libnvvm.so.4
/usr/local/cuda-12.2/nvvm/lib64/libnvvm.so.4.0.0
```

So depending on which of the prefix paths are set to search that version will be loaded. Having your dynamic loader find the wrong version of the library is what typically leads to missing symbols and/or segfaults. In this particular case the shared library `libnvvm.so` isn't the same in the 2 folders despite having identical names and version numbers:
```
$ diff /usr/local/cuda-11.7/nvvm/lib64/libnvvm.so.4.0.0 /usr/local/cuda-12.2/nvvm/lib64/libnvvm.so.4.0.0
Binary files /usr/local/cuda-11.7/nvvm/lib64/libnvvm.so.4.0.0 and /usr/local/cuda-12.2/nvvm/lib64/libnvvm.so.4.0.0 differ
```

Therefore it's crucial that the path `/usr/local/cuda-12.2/nvvm/lib64/` is in `LD_LIBRARY_PATH` when you want to load an application that relies on cuda-12.2. If you have `/usr/local/cuda-11.7/nvvm/lib64/` in there instead, it's very possible the application may crash or complain about some symbol is missing.

case study: in other situations a symbol could be missing because the program wasn't linked properly at build time. Here is
a bug report [undefined symbol curandCreateGenerator for torch extensions](https://github.com/pytorch/pytorch/issues/69666) that demonstrates this exact issue.

### LD_PRELOAD

Let's introduce the `LD_PRELOAD` environment variable, which has multiple purposes, but which can also be used to ensure that the exact desired shared library is loaded, typically when there are multiple libraries with the same name in the library search path. Most of the time this is used as a workaround, since properly packaged distributed shared libraries should already do the right thing.

Following the listing in the previous section, you can force the use of `libnvvm.so` version from `cuda-11.7` with:
```
LD_PRELOAD=/usr/local/cuda-11.7/nvvm/lib64/libnvvm.so.4.0.0 myprogram
```
or from `cuda-12.2` with:
```
LD_PRELOAD=/usr/local/cuda-12.2/nvvm/lib64/libnvvm.so.4.0.0 myprogram
```

footnote: the main use of this environment variable is to intentionally override various APIs. The APIs loaded via this environment variable take precedence over the same APIs loaded by other shared libraries.

If you need to pass multiple paths, it expects these to be `:`-separated, like `LD_LIBRARY_PATH`, as in:
```
LD_PRELOAD="/path/to/libfoo.so:/path/to/libbar.so" myprogram
```

footnote: run `man ld.so` for more information on `LD_PRELOAD`
