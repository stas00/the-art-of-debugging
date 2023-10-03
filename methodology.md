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


## Atomic debug cycles

If you need to run from your shell just these 2 lines on every restart:

```
rm -r data
./launch.sh
```
and the `data` folder impacts how `launch.sh` behaves, you're very likely to forget to reset `data` at some point and run `launch.sh` thinking that you did and get erroneous outcomes which could derail the whole debug process completely since you might accidentally discard the only salvation idea that mistakenly didn't get tested, but you thought you did and the outcome erroneously showed that it wasn't it.

So the foolproof methodology is to change the above 2 command into a single compound one liner:
```
rm -r data; ./launch.sh
```
Now you just need to hit `arrow up` in most Unix shells like Bash to repeat this command again and again.

Certainly, if you have multiple commands to deal with the one liner approach might not work well, then put that commands sequence into a shell script and repetitively launch that script instead. Then you can even have a variety of different debug scripts with the variations that you need.

This idea in a way can be called an atomic operation, where the concept ensures that several sequential actions always happen in the exact same order and they happens a single operation.

This is also why notebook technologies like [Jupyter Notebook](./https://jupyter.org/) and alike, which allow you to go back and forth between different lines of the code and re-execute them selectively, are super-useful at quick prototyping, but are terrible to use for debug purposes because the execution order can't be enforced easily and it is tempting to re-run only parts of the notebook and not wait for a potentially slow full re-run.

A need for a file auto-save falls into this category as well. Since edit-try, edit-try, edit-try cycle is always repeated in the debug process, if the edited change isn't always saved before it's tried, the exact same problem of testing the wrong hypotheses occurs and the discovery of a working solution could be missed completely.



## Automate diagnostics, minimize or avoid typing

Where possible automate diagnostics so you don't need to type them up and waste time and make mistakes.

Here are some examples:

If you use debuggers like gdb or pdb, these usually comes with shortcuts - some premade, some can be added yourself.

For example, you can create `~/.pdbrc` with:

```
alias nl n;;l
alias sl s;;l
```

So now when you're inside `pdb` you don't need to type `next` and then `list`, you can just do `nl` - and `sl` for making an atomic step and list.

Now let's say you need to often dump a complex structure in a certain way. Here is an alias I discovered in someone's `~/.pdbrc`:

```
alias pd for k in sorted(%1.keys()): print "%s%-15s= %-80.80s" % ("%2",k,repr(%1[k]))
```
Now you can just print `pd` in the prompt to get the above command executed - priceless!

In general, the less you need to type, the more likely you will succeed to resolve the issue.

Sometimes it's even possible to completely automate the reporting w/o requiring any typing at all. When I try to debug memory leaks I use tools that report memory usage or deltas automatically. e.g. I developed [ipyexperiments](https://github.com/stas00/ipyexperiments) that auto-reports CPU and GPU memory usage and increments in Jupyter notebook environment. As mentioned earlier one has to be very careful using such environments for debugging so I have to always remember to re-run the whole notebook and not be tempted to re-run only parts of it. But the benefit here is that I can group the code into sections and get auto-reports at how each section of code consumed CPU and/or GPU memory. This is also a fantastic tool for diagnosing memory leaks, as I can re-run the same cell multiple times and see whether the memory usage grows or not.


XXX: link to useful gdb aliases - `compiled.md`?

## watch -n and multiple visible terminals

watch nvidia-smi output like top(1)

```
watch -n 1 nvidia-smi
```
I, of course, have it as an alias:
```
alias wn='watch -n 1 nvidia-smi'
alias wnm='nvidia-smi --query-gpu=timestamp,utilization.memory,memory.used --format=csv -l 1'
```



## sleep

Injecting `sleep` calls in the code being debugged can be very helpful during debug when you need to watch resource usages.

## Narrow down what you monitor

Say you want to watch the processes using `top`, which shows various statistics about processes - this is both confusing filter:


```
# generate ps auxc output for specific processes and show RSS in GBs
ps auxc | grep python | perl -plae '$F[5]=sprintf q[%0.3fGB],$F[5]/2**20; $_=qq[@F]' | column -t
# same but watch like top
watch -n 0.5 $'ps auxc | grep python | perl -plae \'$F[5]=sprintf q[%0.3fGB],$F[5]/2**20; $_=qq[@F]\' | column -t'
# same as above + alias
alias watch-python=$'watch -n 0.5 \'ps auxc | grep python | perl -plae "\$F[5]=sprintf q[%0.3fGB],\$F[5]/2**20; \$_=qq[@F]" | column -t | grep python\''
#
# same as above but with ps headers
(ps auxc | head -1; ps auxc | grep python | perl -plae '$F[5]=sprintf q[%0.3fGB],$F[5]/2**20; $_=qq[@F]') | column -t
# same as above + watch
watch -n 0.5 $'(ps auxc | head -1; ps auxc | grep python | perl -plae \'$F[5]=sprintf q[%0.3fGB],$F[5]/2**20; $_=qq[@F]\') | column -t'
# same as above + alias
alias watch-python=$'watch -n 0.5 \'(ps auxc | head -1; ps auxc | grep python | perl -plae "\$F[5]=sprintf q[%0.3fGB],\$F[5]/2**20; \$_=qq[@F]") | column -t\''
#
# same again, but in a more complicated way - over-complicated but works
ps auxc | perl -ae 'push @x,qq[@F] if !@x; $F[5]=sprintf q[%0.3fGB],$F[5]/2**20; push @x,qq[@F] if $F[10]=~/python/; END {$,=qq[\n]; print @x}' | column -t
# same as above and watch
watch -n 0.5 'ps auxc | perl -ae "push @x,qq[@F] if !@x;; \$F[5]=sprintf q[%0.3fGB],\$F[5]/2**20; push @x,qq[@F] if \$F[10]=~/python/; END {\$,=qq[\n]; print @x}" | column -t'
# same and make alias
alias watch-python=$'watch -n 0.5 \'ps auxc | perl -ae "push @x,qq[@F] if !@x; \$F[5]=sprintf q[%0.3fGB],\$F[5]/2**20; push @x,qq[@F] if \$F[10]=~/python/; END {\$,=qq[\n]; print @x}" | column -t\''
```

```
# top
# to filter out only a wanted program that may not be running yet
top, then o, then COMMAND=python
# to filter already running programs - it will not add new programs started after this command
top -p `pgrep -d "," java`

# htop - more flexible for filtering
# filter by python, sort by RSS, only show my procs
htop -F python -s M_RESIDENT -u `whoami`
# to get the full list of cols to sort by
htop --sort-key help
```


## Power of One-liners

big import failing - move it into a single import one liner.


## Dealing with running out of memory usage

CPU memory:

allocate 1gb on each step and print how many were allocated
perl -le '$|=1; $gbs=10; $b="A"; $gb = $b x 2**30; $y .= $gb and print $_  for 2..$gbs; sleep 20'

meanwhile to watch RSS grow by 1gb:
watch -n 0.5 $'(ps auxc | head -1; ps auxc | grep perl | perl -plae \'$F[5]=sprintf q[%0.3fGB],$F[5]/2**20; $_=qq[@F]\') | column -t'


I also created some helpful tools for pre-filling GPU memory - XXX: ipyexperiments


but sometimes a much better approach is to create a special shell where you limit the resources to a controlled amount

```
# https://www.freedesktop.org/software/systemd/man/systemd-run.html
#
# launch a new shell
systemd-run --user --scope -p MemoryHigh=100G -p MemoryMax=100G -p MemorySwapMax=50G --setenv="MEMLIMIT=100GB" bash
#
# Now any process launched from this shell will get killed if it consumes more than 100GB of RAM and no more than 50GB of SWAP memory
# it will throttle the process when reaching MemoryHigh and kill it when reaching MemoryMax
# so these are soft and hard limits correspondingly
# practically, especially when debugging leaks best to set both to the same value

# can check if it's a shell running systemd-run with `echo $SYSTEMD_EXEC_PID`
# and the custom set variable is useful to check if we are inside the shell with a given limit
echo $MEMLIMIT
100GB
# additional properties are here: https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html#
```

## Finding a breaking commit by bisecting revisions

The discussed next approach should work for any revision control system that supports bisecting. We will use `git bisect` in this discussion.

`git bisect` helps to quickly find the commit that caused a certain problem.

Use case: Say, you were using `transformers==4.33.0` and then you needed a more recent feature so you upgraded to the bleed-edge `transformers@main` and your code broke. there could have been hundreds of commits between the two versions and it'd be very difficult to find the right commit that lead to the breakage by going through all the commits. Here is how you can quickly find out which commit was the cause.

Solution: Bisecting all the commits between the known good and bad commits.

We are going to use 2 terminals A and B. Terminal A will be used for `git bisect` and terminal B for testing your software. There is no technical reason why you couldn't get away with a single terminal but it's easier with 2.

1. In terminal A fetch the git repo and install it in devel mode (`pip install -e .`) into your python environment.
```
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
Now the code of this clone will be used automatically when you run your application, instead of the version you previously installed from pypi or conda or elsewhere.

2. next we launch the bisecting - In terminal A, run:

```
git bisect start
```

3. Establish the last good and the first bad commits

`git bisect` needs just 2 data points to do its work. It needs to know one commit that is known to work (`good`) and one that is know to break (`bad`). So if you look at the sequence of commits on a given branch it'd have 2 know points and many commits around these that are of an unknown quality:

```
...... good ..... .... .... .... ..... bad ....
------------->---------------->----------------> time
```

So for example if you know that `transformers==4.33.0` was good and `transformers@main` (`HEAD`) is bad, find which commit is corresponding to the tag `4.33.0` by visiting [the releases page](https://github.com/huggingface/transformers/releases) and searching for `4.33.0` we find that it was commit `[5a4f340d](https://github.com/huggingface/transformers/commit/5a4f340df74b42b594aedf60199eea95cdb9bed0)`. (typically 8 first places is enough to have a unique identifier)


So now we specify which is the first known good commit:
```
git bisect good 5a4f340d
```

and as we said we will use `HEAD` (latest commit) as the bad one, so it's easy:
```
git bisect bad HEAD
```

If you know it broke in `4.34.0` you can find its latest commit and use that instead of `HEAD`.

We are now all set at finding out the commit that broke things for you.

4. Good or bad

Now in terminal B run your program.

If it fails, in terminal A run:
```
git bisect bad
```
if it succeeds:
```
git bisect good
```

and keep repeating this step until the problem is found.

Once you finished bisecting, `git bisect` will tell you which commit was responsible for breaking things. You can then go to `https://github.com/huggingface/transformers/commit/` and append the commit SHA to that url which will take you to the commit, and which will then link to the PR from which it originated. And then you can ask for help by following up in that PR.

If your program doesn't take too long to run even if there are thousands of commits to search, you are dealing with `2**n` bisecting so 1024 commits can be searched in 10 steps.

If it's very slow, try to reduce it to something small - ideally a small reproduction program that shows the problem really fast. Often commenting out huge chunks of code that you deem irrelevant can be all it takes.

If you want to see the progress, you can ask it to show the current range of remaining commits to check with:
```
git bisect visualize --oneline
```

5. Clean up

So now restore the git repo clone to the same state you started from (most likely `HEAD) with:
```
git bisect reset
```

Possible problems and their solutions:

a. skipping

If for some reason the current sha cannot be tested - it can be skipped and it will continue bisecting with the rest
```
git bisect skip
```
This is often helpful if some API has changed in the middle of the range and your program starts to fail for a totally different reason.

You might also try to make a variation of the program that adapts to the new API, and use it instead, but it's not always easy to do.

b. reversed order

Normally git expects `bad` to be after `good`. Now, if `bad` happens before `good` revision order-wise and you want to find the first revision that fixed a previously existing problem - you can reverse definition of `good` and `bad` - it'd be confusing to work with overloaded logic states, so it's recommended to use a new set of states instead `fixed` and `broken` - here is how you do that.

```
git bisect start --term-new=fixed --term-old=broken
git bisect fixed
git bisect broken 6c94774
```
and then use:
```
git fixed / git broken
```
instead of:
```
git good / git bad
```

c. complications

There are sometime other complications, like when different revisions' dependencies aren't the same and for example one revision may require `numpy=1.25` and the other `numpy=1.26`. If the dependency package versions are backward compatible installing the later version should do the trick. But that's not always the case. So sometimes one has to reinstall the right dependencies before retesting the program.

Sometimes, it helps when there is a range of commits that are actually broken in a different way, you can either find a range of `good...bad` commits that isn't including the other bad range, or you can try to `git bisect skip` the other bad commits as explained earlier.



## SLURM: salloc and srun fast debug combo

If you're in a SLURM environment, do not `sbatch` the job you debug again and again, as it'll add the overhead of the SLURM manager re-allocating the nodes again and again, which, depending on a situation can take up to a minute and even longer, adding a huge repetitive additional overhead to the overhead of starting the program.

Instead, run once, e.g.:

```
salloc --nodes=1 --ntasks-per-node=1 --exclusive --cpus-per-task=96 --gres=gpu:8 --time=6:00:00 bash
```
after editing the command to match your `sbatch`'s settings.

And now from the new shell it opens, run `srun` as many times as needed:
```
srun ... your program
```
when finished or if `salloc` timed out or got `scancel`ed - make sure to exit this bash shell (`Ctrl-d`), since otherwise it'll be populate with stale `SLURM_*` environment variables.

Important: when you exit the shell it'll also revoke the job allocation if it's still running.

note: if you don't specify `bash` at the end of `salloc` you will get auto-ssh'ed into the node that got allocated via `salloc` which you might prefer instead of remaining on the original node. But it can only work with a single node. This is somewhat similar to using the `--pty` flag.


## strace

Say you get a: "No space left on device" traceback, but it doesn't tell you which file it failed to write to.

You could try to go and figure out where the file handle was opened in the code, but often it can be far from trivial.

strace comes to help:


```
687222 openat(AT_FDCWD, "/tmp/tmp5_gxu46v", O_RDONLY|O_CLOEXEC <unfinished ...>
```


## emulate an almost full partition

cd /tmp

dd if=/dev/zero of=1g.bin bs=1G count=28


## emulate an almost full gpu
