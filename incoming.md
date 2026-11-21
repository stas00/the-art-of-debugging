# Incoming Fodder

Please ignore this file. This is just a dump of ideas that I'd like to expand on in the future

## misc topics

- randomization - or rather determinism


## blackbox approach to complex systems

## profiling tools from the Pinterest/Intel gProfiler article

Source: [Pinterest and Intel Take Open-Source Profiling to Internet Scale](https://www.infoq.com/news/2026/11/pinterest-intel-profiling/) (InfoQ, 2026-11-10). Of these the book has only `py-spy` and flame graphs, in [python/README.md](python/README.md).

- [gProfiler](https://github.com/intel/gprofiler) - an open-source profiler that runs `perf`, `async-profiler` and an eBPF-based Python profiler together on a host and merges their stacks into one flame graph across languages, native libraries and the kernel, with no code changes. Its agent stays dormant until it is told to profile a host, service or cluster from a UI or API. Pinterest and Intel added wall-time profiling, PMC flame graphs and Intel PerfSpect telemetry. gProfiler Performance Studio aggregates the profiles from all hosts and compares them across hosts, services and time.
- `perf` - Linux's system-wide sampling profiler, for native code and the kernel.
- [async-profiler](https://github.com/async-profiler/async-profiler) - a sampling profiler for the JVM.
- eBPF-based profiling of Python - sampling the interpreter from the kernel.
- wall-time profiling - shows where threads wait on I/O, locks or RPCs, which a CPU profile doesn't, since a slow program can use little CPU. Compare with `py-spy --idle`.
- PMC flame graphs - flame graphs of hardware performance counter events, such as cache misses and branch mispredictions, by function.
- [Intel PerfSpect](https://github.com/intel/PerfSpect) - hardware telemetry: cache behavior, memory bandwidth, CPU stalls.
- Intel VTune - deeper processor and application analysis on one machine, rather than across a fleet.
- continuous-profiling alternatives: [Parca](https://www.parca.dev/) and [Grafana Pyroscope](https://grafana.com/oss/pyroscope/), both eBPF-based and fleet-wide, and Datadog's commercial continuous profiler.

## commercial debuggers

- [TotalView](https://www.perforce.com/products/totalview) - Perforce's commercial debugger for HPC. It debugs C, C++, Fortran and Python, including Python mixed with C and C++ in one session; multithreaded and multiprocess programs, MPI and OpenMP, from a desktop to many nodes; NVIDIA GPUs running CUDA or OpenACC and AMD GPUs running HIP; and memory - leaks and buffer overruns. Its parallel Python debugging shows and controls several Python processes of a distributed application at once. Compare with the free gdb and cuda-gdb the book already covers, and with Linaro DDT, its main commercial counterpart.
