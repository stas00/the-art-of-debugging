# Debugging Compiled Programs

Please ignore this chapter for now - I'm working on writing it out

## gdb


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
## strace


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
