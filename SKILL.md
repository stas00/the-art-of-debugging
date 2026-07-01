---
name: art-of-debugging
description: >-
  Systematic methodology and concrete tool recipes for debugging Unix, Python,
  and PyTorch programs - crashes, hangs, segfaults, wrong output, CUDA OOM, NaN/Inf,
  slowness, and multi-node/multi-GPU issues. Use when a program crashes, hangs,
  deadlocks, segfaults, runs out of memory (OOM), produces NaN/Inf or wrong
  numbers, runs too slowly, or when the user mentions gdb, strace, py-spy,
  core files, CUDA_LAUNCH_BLOCKING, ldd/nm/LD_PRELOAD, cProfile, or distributed
  training hangs. Distilled from "The Art of Debugging", the latest version of which can be found at https://github.com/stas00/the-art-of-debugging

  The latest SKILL.md version can be found at https://github.com/stas00/the-art-of-debugging/blob/master/SKILL.md
---

# The Art of Debugging

> Distilled from **The Art of Debugging** by Stas Bekman - source: https://github.com/stas00/the-art-of-debugging (CC BY-SA 4.0). This skill is a condensed index; each section links back to the full chapter for depth.

Actionable methodology + copy-paste recipes for debugging Unix / Python / PyTorch programs. Apply the general loop first; then jump to the domain cheatsheet for the failure at hand.

## The debugging loop

The single most important idea: **most of the effort is in *locating* the cause; once you truly understand it, the fix is usually easy.** Optimize everything for reaching understanding faster.

1. **Reproduce reliably.** Get one command that triggers the bug every time. If it's flaky, pin the nondeterminism (seeds, ordering, timing, network, uninitialized memory) first - you can't debug what you can't repeat.
2. **Shrink the payload.** Make the repro *fast*: fewer layers, tiny model/data, one process, one CPU/GPU, one node. A 2-second repro beats a 2-minute one - you'll run it hundreds of times. See [methodology](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md).
3. **Localize.** Confirm you're editing the code that actually runs (the `die` trick), then bisect the search space: which commit, which file/function, which line, which input, which rank.
4. **Get a usable signal.** Turn a cryptic failure into a precise one: a real traceback (sync mode), a stack dump (py-spy/gdb), a syscall trace (strace), a printed value at the boundary, or a min/max/NaN check on a tensor.
5. **Change one thing, re-run, verify.** Fix on the fast repro, confirm, then re-widen to the full payload. Revert anything that didn't help.

### Make the loop fast and reliable

- **Atomic debug cycles.** Make each iteration a single self-contained, repeatable command (setup + run in one shot) so you never hand-redo multi-step state. See [atomic debug cycles](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#atomic-debug-cycles).
- **Automate diagnostics, minimize typing.** Alias the repro and your most-used commands; one keystroke to re-run. See [alias frequently used commands](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#alias-frequently-used-commands) and [automate diagnostics](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#automate-diagnostics-minimize-or-avoid-typing).
- **One-liner programs.** `perl`/`awk`/`python -c` to slice logs, extract fields, transform data on the fly instead of writing throwaway scripts. See [the power of one-liner programs](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#the-power-of-one-liner-programs).
- **Juggle configs cleanly** when running many debug experiments so results don't get confused. See [juggling multiple sets of configs](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#juggling-multiple-sets-of-configs-for-different-debug-experiments).

### Localization techniques

- **Am I editing the right file/class?** Insert a guaranteed break where you *think* execution goes; if the program doesn't die, you're in the wrong file/class/env:
  ```python
  def suspect():
      die   # NameError -> proves this code runs; the traceback also names the caller
  ```
  `traceback.print_stack()` shows callers without stopping (useful when the same function is reached via many paths). See [am I editing the right file and the right class?](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#am-i-editing-the-right-file-and-the-right-class).
- **Bisect a regression.** `git bisect start / bad / good <rev>` walks commits automatically to the one that broke things - script the test for `git bisect run`. See [finding a breaking commit by bisecting](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#finding-a-breaking-commit-by-bisecting-revisions).
- **Small/synthetic payload first.** Use tiny or synthetic inputs; switch to real data only when the bug is data-dependent. See [real vs random vs synthetic data](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#real-data-vs-random-data-vs-synthetic-data).
- **Race conditions.** Reordering/timing bugs hide under async; forcing synchronous execution can expose (or mask) them - note which. See [avoiding race conditions](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#avoiding-race-conditions) and [async vs sync mode](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#async-vs-sync-mode).

### Reproducing resource & environment issues

- **Cap resources on purpose** to test failure paths: emulate a nearly-full disk, limited CPU RAM, or limited GPU memory. See [running out of resources](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#running-out-of-resources-disk-space-cpu-memory-gpu-memory).
- **Watch resources live.** `watch -n1 nvidia-smi` / `free -h` / `df -h` in a second visible terminal to correlate a hang/OOM with what the machine is doing. See [watching and reproducing resource issues](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#watching-and-reproducing-resource-issues).
- **Inject `sleep`** to freeze a program at the interesting moment so you can attach a debugger or snapshot state. See [uses for sleep](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#uses-for-sleep).
- **HPC/SLURM:** keep the allocation and re-run with `srun` instead of re-`sbatch`-ing to cut per-iteration overhead. See [SLURM salloc and srun fast debug combo](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#slurm-salloc-and-srun-fast-debug-combo).

## Unix / shell

Full chapter: [Unix Tools for Debugging](https://github.com/stas00/the-art-of-debugging/blob/master/unix/README.md).

- **Make shell scripts fail loudly and traceably:**
  ```bash
  set -e          # abort on first error
  set -o pipefail # a failing command anywhere in a pipe fails the whole pipe
  set -u          # abort on undefined variables (catches typos)
  set -x          # trace: print each command with expanded values as it runs
  set +x          # turn tracing back off around a noisy region
  ```
  Combine as `set -euo pipefail`. See [controlling script execution](https://github.com/stas00/the-art-of-debugging/blob/master/unix/README.md#controlling-script-execution).
- **`strace` - trace system calls** to see what a program actually does (files, network, why it's stuck):
  ```bash
  strace python -c "print('hi')"                 # trace from the start
  strace --pid PID                               # attach to a running/stuck process
  strace -o log.txt -f python -m torch.distributed.run ...  # -f follows forked children
  strace -e trace=open,openat,read python prog.py           # filter to specific syscalls
  strace -e trace=network -p PID                            # is it stuck on a socket?
  ```
  Classic use: a process at 100% CPU with no output, or hung on I/O/network. See [strace](https://github.com/stas00/the-art-of-debugging/blob/master/unix/README.md#strace).
- **`nohup` - survive logout/disconnect** (don't lose a long run to a dropped SSH):
  ```bash
  nohup ./long-running-command > log.txt &
  ```
  See [nohup](https://github.com/stas00/the-art-of-debugging/blob/master/unix/README.md#nohup).
- **`make`** - after editing compiled sources, rebuild before re-testing, or you'll debug a stale binary. See [make](https://github.com/stas00/the-art-of-debugging/blob/master/unix/README.md#make).
- **Terminal ergonomics:** search long scrollback and copy multi-line commands cleanly; keep an **informative prompt** (host, path, git branch, last exit code) so you always know where/what ran. See [shell environment](https://github.com/stas00/the-art-of-debugging/blob/master/unix/README.md#shell-environment).

## Compiled programs (C/C++, extensions, shared libraries)

Full chapter: [Debugging Compiled Programs](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md). Compile with `-g` for debug symbols.

- **Segfault -> backtrace from a core file:**
  ```bash
  ulimit -c unlimited                                       # allow core dumps in this shell
  sudo sysctl -w kernel.core_pattern=/tmp/core-%e.%p.%h.%t  # control where cores go
  ./program                                                 # crash -> core file written
  gdb ./program /tmp/core-...                               # or: gdb -c core ./program
  ```
  At the `(gdb)` prompt:
  ```
  bt                    # backtrace (read bottom-up: outermost caller -> crash site)
  bt full               # + local variable values at each frame
  thread apply all bt   # backtrace for every thread (essential for multithreaded crashes)
  ```
  See [segmentation fault, core files and gdb](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#segmentation-fault-core-files-and-gdb).
- **No core? Run it under gdb** and step to the crash:
  ```bash
  gdb ./program
  (gdb) run            # then: bt / break FILE:LINE / next / step / print VAR / continue
  ```
  See [run the program under gdb](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#run-the-program-under-gdb).
- **Inspect / snapshot a running process:**
  ```bash
  sudo gdb --pid=PID    # attach; then: thread apply all bt
  gcore PID             # force a core dump without killing (or: kill -ABRT PID)
  ```
  See [get the backtrace from the still running process](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#get-the-backtrace-from-the-still-running-process).
- **"symbol not found" / wrong library loaded:**
  ```bash
  ldd ./program                             # which shared libs resolve, and to what paths
  LD_LIBRARY_PATH=/path/to/libs ./program   # prepend a search dir
  nm -D libfoo.so | grep symbol             # is the symbol actually exported? (T=defined, U=undefined)
  LD_PRELOAD=/path/to/shim.so ./program     # force-load / override a library
  ```
  See [debugging shared libraries and symbol resolution](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#debugging-shared-libraries-and-symbol-resolution) ([ldd](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#ldd), [nm](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#nm)).

## Python

Full chapter: [Debugging Python Programs](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md).

- **Print effectively** instead of scattering bare `print`:
  - **auto-print name+value** so you never mislabel output. See [auto-print what's being observed](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#auto-print-whats-being-observed).
  - **dump all attributes** of an object to see its real state. See [printing object variables](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#printing-object-variables).
  - **trace calls/returns** with `q` (writes to `/tmp/q`, doesn't pollute stdout). See [q](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#q).
- **Run the code you think you're running.** Edits not taking effect? Wrong copy is imported:
  ```bash
  pip install -e .               # run from the source tree, not a copied install
  PYTHONPATH=src python prog.py  # or point Python straight at the source
  python -c "import pkg; print(pkg.__file__)"   # confirm which file is actually loaded
  ```
  See [ensuring the Python package you edit is the one that is run](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#ensuring-the-python-package-you-edit-is-the-one-that-is-run) and [make tests use the git repo's packages](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#setting-up-your-test-suite-to-always-use-the-git-repos-python-packages).
- **Who called this?** `traceback.print_stack()` or the `die` trick to reveal the caller in complex codebases. See [who is calling?](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#who-is-calling).
- **Diagnose a hang (process alive but stuck)** with `py-spy` - no code changes, attaches live:
  ```bash
  pip install py-spy
  py-spy dump -n -p PID          # -n also shows native (C/C++ extension) frames
  # all Python subprocesses at once (skip the launcher):
  pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
  ```
  No sudo? `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`. **The first line of each dump is where it's stuck.** See [py-spy](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#py-spy).
- **Slow code -> profile before optimizing** (measure, don't guess):
  ```bash
  python -m cProfile -s cumtime prog.py     # what dominates cumulative time
  kernprof -l -v prog.py                     # line_profiler: per-line timing of @profile funcs
  ```
  For sub-ms functions, bump `pstats` precision (e.g. `pstats.f8 = lambda x: f"{x:6.3f}"`) so timings aren't all `0.000`. See [profilers](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#profilers) and [cProfile](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#cprofile).

## PyTorch (incl. CUDA / multi-GPU / multi-node)

Full chapter: [Debugging PyTorch Programs](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md).

### Debug fast

Shrink the model, not the problem - make a full run finish in seconds:
- **Fewer layers** via (1) a local clone with config edits, (2) editing the config object on the fly, or (3) hacking the modeling code. See [reducing the number of layers](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#reducing-the-number-of-layers-for-large-models).
- **Tiny random model + tiny tokenizer + tiny dataset** for near-instant iterations; reproduce at full scale only for scale-only bugs. See [making a tiny model](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#making-a-tiny-model) and [faster debug with tiny models, tokenizers and datasets](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#faster-debug-and-development-with-tiny-models-tokenizers-and-datasets).

### Cryptic CUDA errors

CUDA is async, so the reported line is usually wrong. Force a real traceback:
```bash
CUDA_LAUNCH_BLOCKING=1 python prog.py   # sync CUDA -> accurate Python traceback
CUDA_VISIBLE_DEVICES="" python prog.py  # run on CPU (if feasible) for the clearest traceback
```
See [dealing with async CUDA bugs](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#dealing-with-async-cuda-bugs).

### CUDA / CPU OOM

- Distinguish OOM in **`forward`** (activations - batch/seq len) vs **`backward`** (gradients/optimizer state). See [debugging CUDA OOM in forward](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#debugging-cuda-oom-in-forward) / [backward](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#debugging-cuda-oom-in-backward).
- **Fragmentation** (free memory exists but not contiguous): tune `PYTORCH_CUDA_ALLOC_CONF` (e.g. `expandable_segments:True`, `max_split_size_mb:...`). See [overcoming CUDA OOM due to memory fragmentation](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#overcoming-cuda-oom-due-to-memory-fragmentation).
- **See who allocated what** with the memory profiler / allocation tracing; probe the ceiling with the allocatable-GBs test. See [PyTorch memory profiler](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#pytorch-memory-profiler), [strategic memory allocation tracing](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#strategic-memory-allocation-tracing), [discovering allocatable GBs before OOM](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#discovering-how-many-gbs-is-allocatable-before-oom-for-cpu-and-gpu).
- **CPU OOM / peak RAM:** see [CPU memory](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#cpu-memory).

### NaN/Inf & wrong numbers

```python
torch.autograd.set_detect_anomaly(True)   # pinpoint the op that first produced NaN/Inf in backward
```
Find where bad values first appear; watch fp underflow/overflow (especially fp16/bf16); expect small, benign cross-device numeric differences. Inspect tensors compactly (shape/device/dtype/stats) and use `lovely-tensors` for one-line summaries that surface bad tensors fast. See [detecting problematic tensor values](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#detecting-problematic-tensor-values), [underflow and overflow detection](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#underflow-and-overflow-detection), [floating point discrepancies across devices](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#floating-point-math-discrepancies-on-different-devices), [dumping tensor values](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#many-ways-to-dump-tensors-values), and [auto-dumping tensor attributes](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#auto-dumping-desired-tensor-attributes).

### Segfault in a PyTorch/NCCL extension

Same core-file + gdb flow as compiled programs, but **activate the exact python env that produced the core** or gdb can't unpack it:
```bash
conda activate my-env
gdb python core-python-...      # then: bt / thread apply all bt
```
See [segfaults and getting a backtrace from a core file](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#segfaults-and-getting-a-backtrace-from-a-core-file).

### Multi-GPU / multi-node hang or deadlock

1. **Verify comms first** with a minimal all-reduce test (`torch-distributed-gpu-test.py`); rule out network/NCCL before app code. See [getting nodes to talk to each other](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#getting-nodes-to-talk-to-each-other) and [InfiniBand connection](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#solving-the-infiniband-connection-between-multiple-nodes).
2. **Dump every rank's stack at once** with `py-spy` (recipes for python/deepspeed/accelerate, across nodes via `srun`/`pdsh`). Ranks stuck at *different* lines reveal the desync (a mismatched collective). See [diagnosing crashes, hangs and tracing execution](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#diagnosing-crashes-hangs-and-tracing-execution).
3. **Make distributed output legible:** prefix every log line with `node:rank`, and target `pdb` at one rank. See [prefixing logs](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#prefixing-logs-with-noderank-interleaved-asserts), [pdb on a specific rank](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#invoke-pdb-on-a-specific-rank-in-multi-node-training).
4. **Narrow further:** check for a [network-level hang](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#network-level-hanging), [isolate a bad GPU](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#isolate-problematic-gpus), or trace line-by-line with the [python `trace`](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#python-trace) module. On AMD, a slow/hung run may be [IOMMU-related](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#amdrocm-hangs-or-slow-with-iommu-enabled).

### Performance

- **Time regions precisely.** For GPU work use CUDA events (CPU timers lie because kernels are async):
  ```python
  s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  s.record(); run(); e.record(); torch.cuda.synchronize()
  ms = s.elapsed_time(e)
  ```
  See [measuring durations](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#measuring-durations).
- **Profile ops** with `torch.profiler` (CPU+GPU, op-level, small overhead); when it's not enough, drop to `cProfile` for pure-Python hot spots. See [performance and profiling](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#performance-and-profiling).

## Pick the tool by symptom

| Symptom | Reach for |
|---|---|
| Stuck / 100% CPU / no output | `py-spy dump` (Python), `strace --pid` (syscalls), `gdb --pid` (native) |
| Multi-GPU/node hang | minimal collective test -> `py-spy` across all ranks -> `node:rank` logs |
| Segfault / crash in C or extension | core file + `gdb` (`bt`, `bt full`, `thread apply all bt`) |
| Cryptic CUDA error / wrong line | `CUDA_LAUNCH_BLOCKING=1`, or run on CPU |
| CUDA/CPU OOM | forward vs backward; fragmentation (`PYTORCH_CUDA_ALLOC_CONF`); memory profiler |
| NaN/Inf / wrong numbers | `set_detect_anomaly`, under/overflow detection, per-tensor stats, `lovely-tensors` |
| "my edits do nothing" | the `die` trick; `pip install -e .` / `PYTHONPATH`; check `pkg.__file__` |
| Who calls this? | `traceback.print_stack()` / `die` |
| Wrong/missing shared lib | `ldd`, `nm -D`, `LD_LIBRARY_PATH`, `LD_PRELOAD` |
| Too slow (Python) | `cProfile -s cumtime`, `line_profiler` |
| Too slow (PyTorch/GPU) | CUDA events, `torch.profiler` |
| Regression appeared | `git bisect run` |
| Script fails silently | `set -euo pipefail`, `set -x` |
| Flaky / non-deterministic | pin seeds/order/timing; force sync; check race conditions |
| Long run dies on disconnect | `nohup ... > log &` (or tmux/screen) |

## Notes for AI agents

- **Observe before guessing:** obtain a stack dump / traceback / syscall trace / boundary value / tensor stat *before* proposing a cause; don't speculate from the error string alone.
- **Secure a fast, reliable repro first,** then optimize its speed - iteration count matters more than any single clever idea.
- **Change one variable at a time,** re-run the repro, and revert changes that don't move the needle.
- **Confirm you're running the code you edited** (`pkg.__file__`, the `die` trick) before deeper investigation - a huge share of "impossible" bugs are wrong-file/wrong-env.
- **Read the linked chapter section** before applying an unfamiliar recipe - each has worked examples, caveats, and copy-paste scripts.
- Prefer built-in, low-overhead tools (`py-spy`, `strace`, `gdb`, env vars) that need no source changes and work on already-running processes.
