# Fast Debugging Methodology

This chapter discusses various methodologies for fast and successful debugging.

## Quick Iterations, Small Payload

The most important needs of successful debugging

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
than that:
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
tensor([[1.0, 2.0],
        [3.0, 4.0]])
```

If you ever debugged an assembly code that is operates on hex code (`[0-9A-F]`) often someone would inject hex  sequences like `DEADBEEF` because it really stands out in a hex code like: `A0E92827A99DEADBEEF183E`. So, often, you can be creative and create synthetic data which will really make it easy to see what the problem is. At other times any **small** random data will do.

Even if you have to eventually use real data, still try to use the smallest possible real data.

For example, if you are debugging a Machine Learning project, that needs to train a 175B-parameter model, you can use a 1B- or even a 125M-parameter model, for basic quality testing. And a 10K-parameter model for debugging the functionality and exceptions.

If you are into Machine Learning and want to go into a detailed process please read:
[Faster debug and development with tiny models, tokenizers and datasets](https://github.com/stas00/ml-engineering/blob/master/transformers/make-tiny-models.md)


## Single process, single cpu, single gpu, local desktop/laptop

Large data requires big resources that lead to big overheads.

Large data often doesn't fit onto a single CPU or GPU and requires some parallelization method that would require multiple processes makes debugging very difficult. Most debuggers won't be able to handle such tasks and `print` will be your only friend and savior.

On the other hand if your data is small, you can fit onto a single small gpu and at times you don't even need a gpu - a cpu might be able to handle the load fast enough.

Of course, this saves costs as well. And reduces carbon footprint.

This also allows you to develop locally on your desktop/laptop w/o you needing the complication of a remote server. If you have a powerful recent CPU, you might not even need the GPU. If can't have a GPU in your laptop - get an eGPU.

footnote: as long as you're not a Mac user, who as I understand simply can't have a GPU as of this writing not in their laptop nor even use an eGPU.

footnote: I was able to get a [PCIe version of A100 working in my desktop](https://stasosphere.com/entrepreneur-being/262-getting-nvidia-a100-80gb-pcie-to-work-on-a-consumer-motherboard-with-custom-water-cooling/). I was hoping that I could use [Multi-Instance GPU](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) in which case I could emulate 7 tiny GPUs and be able to debug multi-node setups, but alas NCCL doesn't support that. But still I do pretty much most of development locally on my desktop, using a powerful GUI debugger in [PyCharm](https://www.jetbrains.com/help/pycharm/) because I use tiny models. But won't it be amazing to be able to emulate a full compute node on a desktop computer except with tiny gpus?




## Avoiding race conditions

There are times when a debugger can't help resolve a problem. For example, if you're dealing with a race condition or sometimes a deadlock, an introduction of a debugger changes the timing and the problem disappears.

In this case typically you have to resort to solutions like `print` and sometimes `strace` can help.

If it's a hanging you can, of course, still attach to the process with gdb if you're doing libc level debugging, or the higher level tracer like `py-spy` in python.


## Async vs sync mode

When an exception is thrown during an async operation the program is likely to fail to tell you where the problem originated from, which makes it for a very difficult cause hunting process.

For example, PyTorch uses CUDA which has a lot of its operations executed in async mode. And when things fail you typically get a cryptic error which tells you absolutely nothing about where it came from.

Luckily there is a little known environment variable `CUDA_LAUNCH_BLOCKING`, which when set turns the async nature off and suddenly on the same trigger you get a nice Python traceback which tells you exactly what the problem is and then it's trivial to fix. To activate it just do:

```
CUDA_LAUNCH_BLOCKING=1 python myprogram.py
```

One side effect of activating such flags is that it changes the timing of the execution and if the async kernel had a deadlock issue, it might disappear and you won't be able to debug the deadlock issue.

case study: when we were preparing for BLOOM-176B training we were getting a deadlock once we used more than a certain amount of gpus and no amount of lost hair helped finding the cause. And very early on, we discovered that when we used `CUDA_LAUNCH_BLOCKING=1` the hanging would disappear. Since we didn't have the luxury of spending a month on figuring it out - we measured the performance and discovered that there was no perceivable slowdown when async-execution was turned off, even though the manpage warns to only ever use `CUDA_LAUNCH_BLOCKING=1` for debug purposes. Of course, you might not be always that lucky, but don't be afraid to go against the recommendations if it unblocks your progress.


## Atomic debug cycles

If you need to run from your shell just these 2 lines on every restart:

```
rm -r data
./launch.sh
```
and the `data` folder impacts how `launch.sh` behaves, you're very likely to forget to reset `data` at some point and run `launch.sh` thinking that you did and get erroneous outcomes which could derail the whole debug process completely since you might accidentally discard the only salvation idea that mistakenly didn't get tested, but you thought you did and the outcome erroneously showed that it wasn't it.

So the foolproof methodology is to change the above 2 commands into a single compound one-liner:
```
rm -r data; ./launch.sh
```
Now you just need to hit `arrow up` in most Unix shells like Bash to repeat this command again and again.

Certainly, if you have multiple commands to deal with the one-liner approach might not work well, then put that commands sequence into a shell script and repetitively launch that script instead. Then you can even have a variety of different debug scripts with the variations that you need.

This idea in a way can be called an atomic operation, where the concept ensures that several sequential actions always happen in the exact same order and they happens a single operation.

This is also why notebook technologies like [Jupyter Notebook](./https://jupyter.org/), [IPython](https://en.wikipedia.org/wiki/IPython) and alike, which allow you to go back and forth between different lines of code and re-execute them selectively, are super-useful at quick prototyping, but can be terrible to use for debug purposes because the execution order can't be enforced easily and it is tempting to re-run only parts of the notebook and not wait for a potentially slow full re-run.

A need for a file auto-save falls into this category as well. Since edit-try, edit-try, edit-try cycle is always repeated in the debug process, if the edited change isn't always saved before it's tried, the exact same problem of testing the wrong hypotheses occurs and the discovery of a working solution could be missed completely.

footnote: as much as I like `nano` for quick remote editing on the node, I have to remember to `Ctrl-o` to manually save the change before trying it out.

Going back to the proposition of using `;` to concatenate multiple commands - this approach will always execute each command regardless of whether it succeeded or not. In some situations where any of the commands might fail you might want to use `&&` instead of `;` to do the concatenation so instead of doing:

```
rm -r data; echo start; ./launch.sh
```
you may choose to do:

```
rm -r data && echo start && ./launch.sh
```
now the next command in the sequence will only be executed if the previous one was successful.


## Alias frequently used commands

If you repeat the same commands often - consider using aliases. Especially in situations when commands use multiple difficult to remember flags.

suggestion: always add a few aliases at a time and start using them before adding new aliases. Like learning a new new languages if you don't use it you lose it.

Here are some practical examples of aliases that I type probably dozens of times daily:

```
# here is how I launch pytest
alias pyt="pytest --disable-warnings --instafail -rA"

# this is how I tell pytest to show me all the available tests
alias pytc="pytest --disable-warnings --collect-only -q"
```

Some aliases are easier to write as functions, but essentially they act the same as aliases. Examples:

```
# this is the most used alias
function l()  { ls -lNhF   --color=always "${@-.}" |more; }

# same as l but sort by latest
function lt() { ls -lNhtF  --color=always "${@-.}" |more; }

# same as l but include dot files
function la() { ls -lNhaF  --color=always "${@-.}" |more; }
```

principle: the more often the alias is used the less letters it should have. As I use `ls -lNhF   --color=always "${@-.}" |more` probably hundreds of times a day, I gave it a single letter. Other less often used `ls` aliases use more than one letter.

principle: have all the aliases for the same tool start with the same letter or letters. In the example above all `ls` candidates start with `l` (but of course, there could be other `l`-starting aliases that aren't aliases for `ls`. Another example is that all my `git` aliases start with `g`, e.g. I use `gp` to push and `gc` to commit since these are the most often used git commands that I use.

Aliases can be a problem when you write documentation and start accidentally including your command line dumps that includes aliases, but others don't have them and suddenly the code instructions in docs work only for you.

To see if something is an alias, use `type`:
```
$ type l
l is a function
l ()
{
    ls --color=auto -lNhF --color=always "${@-.}" | more
}

$ type pyt
pyt is aliased to `pytest --disable-warnings --instafail -rA'
```

As mentioned functions are more flexible than aliases since they allow arguments. Examples:

```
function wrap()       { tar -czf "$1.tgz" "$1"; }
function wraprar()    { rar a -rr -m5 "$1.rar" "$1"; }
```
and also you can do crazy things like:
```
# usage: git-replace oldstr newstr
#
function git-replace () {
    files=$(git grep -l "$1")
    if [[ ! -z "$files" ]];
    then
        perl -pi -e "s|$1|$2|g" $files
	    git diff --shortstat
    else
        echo "no match for: $1"
        false
    fi
}
```
This handy function replaces one string with another in all files under git.

If you aliased a program with the same name as the program itself:
```
alias df="/bin/df -h | grep -v /dev/loop"
```
and you want to do something different you then have to use the full path to the program:
```
/bin/df -ih
```
and then the alias won't be activated.

In the above example, I use `df` with human formatted sizes and I don't want to see dozens of `/dev/loop` entries. But sometimes I want to see inodes count, so I'd use `/bin/df -ih`. I could have created an alias for it, but I do it so rarely that I don't want to pollute my limited memory space in my head.


## Cheatsheets

While aliases are super handy, too many aliases can be difficult to remember, therefore it's also very useful to have various cheatsheets each specific to a sub-field of your occupation. I have a cheatsheet for git, python, gdb, transformers, conda, pip, bash, etc. In those cheatsheets I write very dense one-line comments of what the following line does, a compact output if relevant, and I constantly improve and re-organize them. This helps me to map things out in my head and know where I can quickly find a specific solution. StackOverflow is awesome, but having my own StasOverflow is priceless.

For example, here is a snipped of my git cheatsheet:

```
# ranges illustration
A ─┬─ E ── F ── G   master
   └─ B ── C ── D   fix
git log master..fix   BCD
git log master...fix  BCD and EFG

git log master 	      reachable parents from master
git log ^master 	  exclude reachable parents from master
git log master..fix   reachable from fix but not master
git log master...fix  reachable from fix and master, but not both
git log HEAD^@ 	      parents of HEAD
git log HEAD^! 	      HEAD, then excluding parents’s ancestors
git log HEAD^{:/fix}  search previous HEADs matching criteria

[...]

# reset branch's HEAD to a given commit hash:
# find the last commit that was supposed to be the HEAD, e.g.:
# https://github.com/fastai/fastai/commit/1c63e868d3d11e73d9f51f58cbd271e67a0fe983
# and now reset the branch's HEAD to it
git checkout release-1.0.36
git reset --hard 1c63e868d3
git push --force origin release-1.0.36
```

I carefully arrange similar entries so that the entries are vertically aligned for quick grasping.

For the recipes I use real urls, tags, files, so that I know exactly what is meant. Rather than URL, FILENAME, etc., placeholders as some instruction documents use.


Here is a snippet from the PyTorch cheatsheet:

```
# create a tensor
# torch.tensor - ***always copies data!***
# torch.tensor(x) is equivalent to x.clone().detach() if x is a tensor
torch.tensor(4)                     # scalar 4,        0 dims, size []
torch.tensor([4., 5.])              # vector [4., 5.], 1 dims, size [2]
torch.tensor(4, requires_grad=True) # w/ gradient tracking (default False)
torch.tensor(4, dtype=torch.int16)  # cast to data type
torch.tensor(4, device="cuda:0")    # put on a desired device

[...]

# create a 1D tensor of size ⌈end−start/step⌉ with values from the interval
# [start, end) taken with common difference step beginning from start.
# defaults: start=0, step=1
torch.arange(5)           # tensor([ 0.,  1.,  2.,  3.,  4.])
torch.arange(1, 4)        # tensor([ 1.,  2.,  3.])
torch.arange(1, 2.5, 0.5) # tensor([ 1.0000,  1.5000,  2.0000])


[...]
# method     # tensor             converts to: but not in place - need to assign to a new variable
#
t.half()     # torch.HalfTensor   torch.float16
t.float()    # torch.FloatTensor  torch.float32
t.double()   # torch.DoubleTensor torch.float64
#
t.int8()     # torch.CharTensor   torch.int8    8-bit integer (signed)
t.uint8()    # torch.ByteTensor   torch.uint8   8-bit integer (unsigned)
t.byte()     # torch.ByteTensor   torch.uint8   8-bit integer (unsigned)
t.short()    # torch.ShortTensor  torch.int16
t.int()      # torch.IntTensor    torch.int32
t.long()     # torch.LongTensor   torch.int64
```

Note how this is very dense and very easy to grasp quickly thanks to vertical alignment and I even manage to pack the output on the same line.


As I often load and troubleshoot HF models, datasets, tokenizers I have the following in my HF cheatsheet:
```
# cache transformer model + tokenizer
python -c 'import sys; from transformers import AutoModel; AutoModel.from_pretrained(sys.argv[1])' t5-small
python -c 'import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])' t5-small
python -c 'import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1])' t5-small

# cache dataset and metrics
python -c 'import sys; from datasets import load_dataset; ds=load_dataset(sys.argv[1])' stas/openwebtext-10k
```

Please note that not only it's a one-liner I can instantly copy into the shell, I make it so that the variable name of the model or dataset is an argument to the one-liner program, so it's trivial to replace. And again this is a complete example that includes an actual argument so it's self-documenting.

I use the above for testing that the resources can be downloaded and this method is also handy to pre-download and cache these resource. I use those as well when I build my own models and datasets locally.

Then in the same cheatsheet I have complicated recipes like:

```
### re-shard an existing model to 2GB shards

# use the PR version of transformers till it's merged
git clone https://github.com/huggingface/transformers -b add-model-idefics
cd transformers

# reshard this model - replace only this to do another model
git clone https://huggingface.co/HuggingFaceM4/idefics-9b
cd idefics-9b
mkdir -p 2GB

# reshard the normal files
PYTHONPATH=../src python -c 'import sys, transformers; transformers.IdeficsForVisionText2Text.from_pretrained(sys.argv[1]).save_pretrained("2GB", max_shard_size="2GB")' .

# update or create anew safetensors
cd 2GB
python -c "import re, sys, torch; from safetensors.torch import save_file; [save_file(torch.load(f), re.sub(r'.*?(model.*?)\.bin',r'\1.safetensors',f), metadata={'format': 'pt'}) for f in sys.argv[1:]]" *bin
cp pytorch_model.bin.index.json model.safetensors.index.json
perl -pi -e 's|pytorch_||; s|\.bin|.safetensors|' model.safetensors.index.json
cd -

# update the repo files
git rm pytorch_model-*.bin
git rm model*safetensors
mv 2GB/* .
git add *bin *safetensors

# check and push
git commit -am "re-shard to 2GB"
git push
```

Note that everything is one-liners so I can quickly tweak anything I want and I don't need to edit any files and then figure out where they are, scp them to a remote node, then clean up - it's just a copy-n-paste away and nothing to clean up.

I often freely switch between Bash, Python And Perl one-liners depending on what does the job the best for me. YMMV, but the point is that do whatever makes you most productive. For example, I lived and breathed Perl for more than 25 years and it's the best language for text processing, IMHO, so, of course, I continue using it whenever it suits my needs. If I have to write something that others need to understand easily I can then rewrite it in a long program in another language. But usually debugging is here and now, so if I have to write a program to debug a program and debug it too, I will never get to the finish line.

recommendation: Do not use other people's cheatsheets other than as a fodder and inspiration. Make your own and format it so that you can quickly find what you need and once found you can instantly get it.

Here are some of my cheatsheets if you need a starter:
[
[python](https://github.com/stas00/python-tools/blob/main/python.txt) |
[git](https://github.com/stas00/git-tools/blob/master/git.txt) |
[bash](https://github.com/stas00/bash-tools/blob/main/bash.txt) |
[make](https://github.com/stas00/make-tools/blob/main/make.txt) |
[conda](https://github.com/stas00/conda-tools/blob/master/conda.txt)
]




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

Sometimes it's even possible to completely automate the reporting w/o requiring any typing at all. When I try to debug memory leaks I use tools that report memory usage or deltas automatically. e.g. I developed [ipyexperiments](https://github.com/stas00/ipyexperiments) that auto-reports CPU and GPU memory usage and increments in Jupyter notebook environment. As mentioned earlier one has to be very careful using such environments for debugging so I have to always remember to re-run the whole notebook and not be tempted to re-run only parts of it. But the benefit here is that I can group the code into sections and get auto-reports at how each section of code consumed CPU and/or GPU memory. This is also a fantastic tool for diagnosing memory leaks, as I can re-run the same cell multiple times emulating a loop and see whether the memory usage grows or not.


XXX: link to useful gdb aliases - `compiled.md`?




## watch -n and multiple visible terminals

Everybody loves tools that report resource states in real time, e.g. `top` to watch CPU memory usage, core utilization and other essential system stats.

There is a way to turn any state reporting tool into a live updating one. `watch -n` does the trick.

For example, let's say we want to watch the output of `nvidia-smi` updating once a second. This is just:

```
watch -n 1 nvidia-smi
```
Change `1` to `0.5` or `2` to whatever number of seconds you want the refresh to happen at.

I, of course, have it as an alias since I use it all the time:
```
alias wn='watch -n 1 nvidia-smi'
alias wnm='nvidia-smi --query-gpu=timestamp,utilization.memory,memory.used --format=csv -l 1'
```

The second need here is to have more than one terminal - so that you have one terminal where you run the program and another where you run a monitoring program. GUI tools of course work as well, but the key here is that they shouldn't overlap, so that you can see both at the same time.

If you work on a tiny 14" laptop or even 17" one consider getting yourself a large wide monitor or two smaller ones - it'll change your debugging productivity dramatically.

Let's look at some more practical `watch -n` examples.

Do you have a problem with a program eating up a disk space on some partition and you want to correlate the execution with that partition? Say, it's a partition named `/tmp`
```
watch -n 'df -h | grep /tmp'
```

One critical methodology to notice here is that I carefully filter only the data I need to watch. While I can have the whole often huge output of `df` refreshing once a second, it'd make noticing the single entry very difficult. By filtering out all the noise I get the signal that I need much easier.

Now, say, you want to watch how your system handles programs that allocate too much memory and has `cgroups` killing it. You could run this little one-liner in a console which would allocate 1GB of cpu memory on each step and print how many were GBs allocated:
```
perl -le '$|=1; $gbs=10; $b="A"; $gb = $b x 2**30; $y .= $gb and print $_  for 2..$gbs; sleep 20'
```
meanwhile to watch RSS grow by 1GB in another console run:
```
watch -n 0.5 $'(ps auxc | head -1; ps auxc | grep perl | perl -plae \'$F[5]=sprintf q[%0.3fGB],$F[5]/2**20; $_=qq[@F]\') | column -t'
```

Replace `grep perl` with `grep python` and now you can nicely watch only Python processes in `top`-like style,

footnote: you can tell `top` to do the same (stat `top`, then hit `o`, then type `COMMAND=python`) but you can't automate it. Sometimes you can cheat with `top -p $(pgrep -d "," python)` but the target process has to be running already, I'd rather keep the diagnostics running non-stop.

Aliasing this requires a bunch of backslashes:
```
alias watch-python=$'watch -n 0.5 \'(ps auxc | head -1; ps auxc | grep python | perl -plae "\$F[5]=sprintf q[%0.3fGB],\$F[5]/2**20; \$_=qq[@F]") | column -t\''
```

Specifically for `top` limitations to filtering, `htop` is more flexible filtering-wise. For example: filter by python, sort by RSS, only show my procs
```
htop -F python -s M_RESIDENT -u `whoami`
```
To get the full list of cols to sort by `htop --sort-key help`

If there is an alternative tool that already does what you need by all means use it, this was really a demonstration of how `watch -n` can be super-useful.

But `ps` can give me much more narrowing power, since you can filter by more things. For example, `ps -ef` will give you the long command line for each process with all the arguments it was launched with - so if you have multiple Python programs and you want to filter only specific Python processes you can easily do it now based on the long command line output.




## Uses for sleep

Sometimes when you try to debug excessive resource usage, the code runs too fast to be able to see the level of consumption in an external monitor. In this situation injecting `sleep` calls in the code being debugged, usually right after the code which is a suspect to overuse a resource in question, can be very helpful during debug when you need to watch resource usages.

As you have seen in the example from the section above:
```
perl -le '$|=1; $gbs=10; $b="A"; $gb = $b x 2**30; $y .= $gb and print $_  for 2..$gbs; sleep 20'
```
After we made this one liner to allocate 10GB of CPU memory, we sleep for 20 seconds so that we could observe this memory being allocated in `top` or another tool.


## The power of one-liner programs

This is a program:

```
#!/bin/bash

echo this is time
date
```

This is a one-liner program
```
echo this is time; date
```

If a program can be reduced to a single line that can be copy-n-pasted into the command prompt and immediately executed you have a one-liner.

Use case: an import failing - move it into a single import one liner. For example, if you have `import torch` failing, test it standalone:

```
python -c "import torch"
```

Use case: a large program failing due to a model or tokenizer or config loading failing after a sizeable overhead of loading other things. Move that failing component out into a one-liner and test it alone. One common example is when you use a private HF hub repo and you are missing an auth token and you wan to debug that. Let's move it into a one liner:

```
python -c 'import sys; from transformers import AutoModel; AutoModel.from_pretrained(sys.argv[1])' t5-small
python -c 'import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])' t5-small
python -c 'import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1])' t5-small
```

I also use this method for pre-downloading/caching large models (which could take up to hours to download) as I can run this in parallel with developing the software that will use that.

Note how I parameterized the model name via `sys.argv[1]` so now it's very easy to switch from `t5-small` to any other model, rather than doing the same by needing to modify the code itself.

Key benefits of using one-liners over real programs:
- you can save many variations of the same one-liner is a file and then refer to them later and immediately make use of any of these with a simple copy-n-paste
- it's atomic during the debug, if you always run the same program name and change it, it could be difficult to track how the program has changed between the runs - the one liner doesn't change and you can run `history` and see exactly what was run and in what sequence
- if you want to change something quickly it's easier to make the change immediately on the command line
- when working on a remote system you don't need to copy the program from your desktop - you can just paste it
- once run it's in the shell history - so it's very easy to iterate over previously run one-liners

footnote: stepping aside from the discussion of debugging I use one-liners for refactoring/renaming - since I can then share these with others who are impacted by my changes and they can just copy-n-paste my commands and immediately update their code to adapt to the new API. I did a lot of those in HF transformers' PRs, e.g. [this one](https://github.com/huggingface/transformers/pull/7863) where I did things like:
```
find . -type d -name ".git" -prune -o -type f -exec \
perl -pi -e 's|require_multigpu|require_torch_multigpu|g' {} \;
```
where I rename a function in all files carefully skipping the `.git` directory. One can of course the same using their IDE, but then you have to tell others to do more work if they have to manually update their diverging branches, or to users if you're intentionally changing the API. So, hey, we have just changed the API, please run this one-liner on your code when you updated to the latest package - a breeze and things continue working for the users despite API changes.

As you warm up to the joy and profit of using one-liners make sure to empower yourself with [
Handy shell shortcuts](#handy-shell-shortcuts) to navigate the command line quickly and learn [how to use bash history](#use-bash-history) to quickly bring previously run one-liners to the fore.

Now you will observe how I use a lot more Perl one-liners in this guide, rather than Python one-liners as Python wasn't designed to be used in this way and so when it works it's mostly accidental. Whereas Perl was designed to be used as part of Unix toolchain and thus it has a myriad of shortcuts that let you do amazing things in just a few short instructions.

As long as you don't need to do anything that comes with `:` in Python, it should mostly work, but as soon as you need to do something like `if a: b()` it breaks due to its parser. It's possible to hack around it using methods like:

```
# delegate \n injection to shell:
python -c "$(echo -e "a='True'\nif a : print(1)")"
# same with exec:
python -c "import torch; exec('with torch.cuda.device(0):\n  x = torch.ones(1,1)')"
```
but this is already very difficult to comprehend so the befit is greatly reduced. Though I saved these recipes as sometimes I still want this available to me over a real program.


## Running out of resources: disk space, cpu memory, gpu memory



### Emulating an almost full disk partition

Sometimes you have to deal with running out of disk space while running a program. For example, let's say, you have a process which besides many other things untars various files and which comes down crashing because `/tmp` runs out of disk space while it runs, but this happens after 10min of running, which is too too long of a wait time to be productive.

You can then precipitate the event by filling up the partition to almost full and then the event will arrive much faster. One of the ways of doing that is to quickly create a large file of the size you desire. For example, if you have 29GB free and you want to leave only 1GB free, then create a 28GB file with:
```
cd /tmp
dd if=/dev/zero of=/tmp/tmp.bin bs=1G count=28
```

Here we tell `dd` to create a file `/tmp/tmp.bin` which is comprised of 28 1GB chunks.

And now when you re-run your program it'll should fail quickly since that partition is almost full from the get going.

This approach works, but it isn't great since it could impact your system's functioning. Instead you can just create a temporary RAM-based filesystem of 1GB and then tell the application to use it instead of `/tmp`.

Step 1. create a 1GB `tmpfs` and mount it at `~/ramdisk`:
```
mkdir ~/ramdisk
sudo mount -t tmpfs -o size=1G,user,mode=1777 tmpfs ~/ramdisk
```
Step 2. run the application, .e.g., `myprogram`:
```
TMPDIR=~/ramdisk myprogram
```

`TMPDIR` is a special environment variable that allows you to override where the application will write to for their tempfile/tmpfs needs, instead of the default `/tmp`.

The additional advantage is that this filesystem is much faster than your normal storage.

Just beware that `tmpfs` uses volatile virtual memory. When you unmount this partition or reboot your system all files created in this partition will disappear. You can read about tmpfs [here](https://www.kernel.org/doc/html/latest/filesystems/tmpfs.html).

When finished with the debug, unmount this partition to free up the borrowed CPU memory:
```
sudo umount ~/ramdisk
```

### Emulating running out of CPU memory

If the application fails because it runs out of memory, but it occurs after many minutes/hours of waiting and you want to precipitate that event you could reduce your available CPU memory by quickly allocating as many GBs as you need.

For example, here is how you can allocate 10GB and hold it in use - 1GB on each step and print how many GBs were allocated:

```
perl -le '$|=1; $gbs=10; $b="A"; $gb = $b x 2**30; $y .= $gb and print $_  for 2..$gbs; sleep 1000'
```

In another shell you can watch how the Resident memory usage grows in real time, 1GB at a time.
```
watch -n 0.5 $'(ps auxc | head -1; ps auxc | grep perl | perl -plae \'$F[5]=sprintf q[%0.3fGB],$F[5]/2**20; $_=qq[@F]\') | column -t'
```

You can control how many GBs to allocate by changing `$gbs=10` to another desired number. And you can adjust the value of `sleep 1000` to however many seconds you want this program to hold this memory in use.

Now you can run your original program with a much reduced available CPU memory.

But sometimes a much better approach is to create a special shell where you limit the desired resources to a controlled amount:

Step 1. launch a new shell:
```
systemd-run --user --scope -p MemoryHigh=5G -p MemoryMax=5G -p MemorySwapMax=3G --setenv="MEMLIMIT=5GB" bash
```

Now any process launched from this shell will get killed if it consumes more than 5GB of RAM and more than 3GB of SWAP memory.

`MemoryHigh` and `MemoryMax` are the soft and hard CPU memory limits correspondingly.

The `MemoryHigh` setting is useful if you want the system to throttle the processes when they allocate more than `MemoryHigh` of CPU memory. So in this example we set it to the same value as `MemoryMax` so it'll have no throttling impact.

I use `--setenv="MEMLIMIT=5GB"` as a helper so that later I can check what limit I had set and it is useful to check if we are inside the shell with a given limit:
```
$ echo $MEMLIMIT
5GB
```

Step 2. Run a program that consumes more than the limit you set in Step 1

Let's reuse the one liner that allocates 1GB at a time, up to 10GB:
```
$ perl -le '$|=1; $gbs=10; $b="A"; $gb = $b x 2**30; $y .= $gb and print $_  for 2..$gbs; sleep 1000'
2
3
4
5
Killed
```
As you can see this program got killed as soon as it allocated 5GB of Resident CPU memory.

You can read [A Deep Investigation into MMAP Not Leaking Memory](https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/) to see how I used this technique to figure out whether MMAP leaks memory or not.

For the detailed manpage see [systemd-run](https://www.freedesktop.org/software/systemd/man/systemd-run.html) and
for additional properties that can be set [this section](https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html#).




### Emulating running out of GPU memory

Now we have dealt with emulating running out of disk space, CPU memory and now we come to the GPU memory. We will use the same principle as before - we can pre-allocate a large chunk of the GPU memory leaving only enough for the test run to occur. This allows us to get to the problematic event much sooner than if we have a huge GPU with lots of free memory. Especially if the issue is an actual memory leak and not a cached memory and it's happening slowly.

First, install `ipyexperiments`
```
pip install ipyexperiments
```

Now, say, you want to leave 3GB of free memory on your GPU card before you launch your program.

Step 1. Run a magical one-liner to use-up all but 3GB of GPU memory

```
python -c 'import time, ipyexperiments.utils.mem; \
do_not_delete = ipyexperiments.utils.mem.gpu_mem_leave_free_mbs(3<<10); \
time.sleep(1000)'
```

You can run `nvidia-smi` to validate that only 3GB remain free.

If you need the allocation to last longer/shorter - adjust the sleep time.

footnote: `3<<10 == 3*2**10` - it's just less to type

Step 2. Run your program that you want to test how it performs with just 3GB of free memory.

You can also copy the code from the one-liner into the beginning of your program, but it's easier to keep the separated so that your program remains clean of debug code.

And how was the memory pre-allocated? Using a simple `torch.ones` allocator. For example here is how you can pre-allocate 10GB of GPU memory:
```
import torch
n=10
x = torch.ones((n*2**18)).cuda().contiguous()
```
Just make sure `x` doesn't go out of scope, since when it does the memory will get released.

If you have a recent NVIDIA GPU (A100, H100, and higher) you could also reshape your GPU using Multi-Instance GPU (MIG) as explained [here](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/). So if you're trying to ensure your software works on a specifically sized smaller GPU than the one you develop on, you may consider resizing your GPU to a smaller one for the duration of your tests.


## Finding a breaking commit by bisecting revisions

The discussed next approach should work for any revision control system that supports bisecting. We will use `git bisect` in this discussion.

`git bisect` helps to quickly find the commit that caused a certain problem.

Use case: Say, you were using `transformers==4.33.0` and then you needed a more recent feature so you upgraded to the bleed-edge `transformers@main` and your code broke. There could have been hundreds of commits between the two versions and it'd be very difficult to find the right commit that lead to the breakage by going through all the commits. Here is how you can quickly find out which commit was the cause.

footnote: HuggingFace Transformers is actually pretty good at not breaking often, but given its complexity and enormous size it happens nevertheless and the problems are fixed very quickly once reported. Since it's a very popular Machine Learning library it makes for a good debugging use case.

Solution: Bisecting all the commits between the known good and bad commits to find the one commit that's to blame.

We are going to use 2 shell terminals: A and B. Terminal A will be used for `git bisect` and terminal B for testing your software. There is no technical reason why you couldn't get away with a single terminal but it's easier with 2.

1. In terminal A fetch the git repo and install it in devel mode (`pip install -e .`) into your Python environment.
```
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
Now the code of this clone will be used automatically when you run your application, instead of the version you previously installed from PyPi or Conda or elsewhere.

Also for simplicity we assume that all the dependencies have already been installed.

2. next we launch the bisecting - In terminal A, run:

```
git bisect start
```

3. Discover the last known good and the first known bad commits

`git bisect` needs just 2 data points to do its work. It needs to know one earlier commit that is known to work (`good`) and one later commit that is know to break (`bad`). So if you look at the sequence of commits on a given branch it'd have 2 known points and many commits around these that are of an unknown quality:

```
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->---------------->----------------> time
```

So for example if you know that `transformers==4.33.0` was good and `transformers@main` (`HEAD`) is bad, find which commit is corresponding to the tag `4.33.0` by visiting [the releases page](https://github.com/huggingface/transformers/releases) and searching for `4.33.0`. We find that it was commit with SHA `[5a4f340d](https://github.com/huggingface/transformers/commit/5a4f340df74b42b594aedf60199eea95cdb9bed0)`.

footnote: typically the first 8 hex characters are enough to have a unique identifier for a given repo, but you can use the full 40 character string.


So now we specify which is the first known good commit:
```
git bisect good 5a4f340d
```

and as we said we will use `HEAD` (latest commit) as the bad one, in which case we can use `HEAD` instead finding out the corresponding SHA string:
```
git bisect bad HEAD
```

If however you know it broke in `4.34.0` you can find its latest commit as explained above and use that instead of `HEAD`.

We are now all set at finding out the commit that broke things for you.

And after you told `git bisect` the good and the bad commits it has already switched to a commit somewhere in the middle:

```
...... orig_good ..... .... current .... .... ..... orig_bad ........
------------->--------------->---------------->----------------> time
```

You can run `git log` to see which commit it has switched to.

And to remind, we installed this repo as `pip install -e .` so the Python environment is instantly updated to the current commit's code version.

4. Good or bad

The next stage is telling `git bisect` if the current commit is `good` or `bad`:

To do so in terminal B run your program once.

Then in terminal A run:
```
git bisect bad
```
If it fails, or:
```
git bisect good
```
if it succeeds.


If, for example, if the result was bad, `git bisect` will internally flag the last commit as new bad and will half the commits again, switching to a new current commit:
```
...... orig_good ..... current .... new_bad .... ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

And, vice versa, if the result was good, then you will have:
```
...... orig_good ..... .... new_good .... current ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

5. Repeat until no more commits left

Keep repeating step 4 step until the problematic commit is found.

Once you finished bisecting, `git bisect` will tell you which commit was responsible for breaking things.

```
...... orig_good ..... .... last_good first_bad .... .. orig_bad ....
------------->--------------->---------------->----------------> time
```
If you followed the little commit diagrams, it'd correspond for the`first_bad` commit.

You can then go to `https://github.com/huggingface/transformers/commit/` and append the commit SHA to that url which will take you to the commit, (e.g. `https://github.com/huggingface/transformers/commit/57f44dc4288a3521bd700405ad41e90a4687abc0` and which will then link to the PR from which it originated. And then you can ask for help by following up in that PR.

If your program doesn't take too long to run even if there are thousands of commits to search, you are facing `n` bisecting steps from `2**n` so 1024 commits can be searched in 10 steps.

If your program is very slow, try to reduce it to something small - ideally a small reproduction program that shows the problem really fast. Often, commenting out huge chunks of code that you deem irrelevant to the problem at hand, can be all it takes.

If you want to see the progress, you can ask it to show the current range of remaining commits to check with:
```
git bisect visualize --oneline
```

6. Clean up

So now restore the git repo clone to the same state you started from (most likely `HEAD) with:
```
git bisect reset
```

and possible reinstall the good version of the library while you report the issue to the maintainers.

Sometimes, the issue emerges from intentional backward compatibility breaking API changes, and you might just need to read the project's documentation to see what has changed. For example, if you switched from `transformers==2.0.0` to `transformers==3.0.0` it's almost guaranteed that your code will break, as major numbers difference are typically used to introduce major API changes.


7. Possible problems and their solutions:

a. skipping

If for some reason the current commit cannot be tested - it can be skipped with:
```
git bisect skip
```
and it `git bisect` will continue bisecting the remaining commits.

This is often helpful if some API has changed in the middle of the commit range and your program starts to fail for a totally different reason.

You might also try to make a variation of the program that adapts to the new API, and use it instead, but it's not always easy to do.

b. reversing the order

Normally git expects `bad` to be after `good`.


```
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

Now, if `bad` happens before `good` revision order-wise and you want to find the first revision that fixed a previously existing problem - you can reverse the definitions of `good` and `bad` - it'd be confusing to work with overloaded logic states, so it's recommended to use a new set of states instead - for example, `fixed` and `broken` - here is how you do that.

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

There are sometimes other complications, like when different revisions' dependencies aren't the same and for example one revision may require `numpy=1.25` and the other `numpy=1.26`. If the dependency package versions are backward compatible installing the newer version should do the trick. But that's not always the case. So sometimes one has to reinstall the right dependencies before re-testing the program.

Sometimes, it helps when there is a range of commits that are actually broken in a different way, you can either find a range of `good...bad` commits that isn't including the other bad range, or you can try to `git bisect skip` the other bad commits as explained earlier.


## Juggling multiple sets of configs for different debug experiments

There are times when one tweaks a single line in a single file to see a problem, but at times it can be many lines in many files. And it becomes very difficult to keep track of what's what and not make mistakes.

Here what helps is to either have a dedicated versioned config file per experiment, or full versioned directories with sets of files that vary.

So for example, say you need to tweak a directory with 2 files:

```
$ ls -1 experiment/
config.yaml
test.py
```
and you invoke the program as:
```
python experiment/test.py --config experiment/config.yaml
```

So let's rename the first set to `experiment1`:
```
mv experiment experiment1
```

So you can create now say 2 additional sets of files:

```
cp -r experiment experiment2
cp -r experiment experiment3
```
now tweak the files in each of these sets as you wish and when you are about to run the actual debug experiment you can simply symlink to the desired set atomically at execution time:

```
ln -s experiment1 experiment; python experiment/test.py --config experiment/config.yaml
```
and later if you want to do set 2:
```
ln -s experiment2 experiment; python experiment/test.py --config experiment/config.yaml
```
and same for 3:
```
ln -s experiment3 experiment; python experiment/test.py --config experiment/config.yaml
```

The critical nuance here is that we are changing a single source of truth as compared to changing the folder name in multiple places:

```
python experiment1/test.py --config experiment1/config.yaml
```

One can also use the `git commit` approach where each variation is committed and then one can quickly lookup the desired version with `git log` and then `git checkout SHA` for the wanted version.

At other times using an environment variable can accomplish a similar things, so here you'd do:

```
SOME_FLAG=exp1 python myprogram.py
SOME_FLAG=exp2 python myprogram.py
```
if you wrote the program to choose a different code path depending on the value of `SOME_FLAG`. And I'm stressing again that we want:
```
SOME_FLAG=exp1 python myprogram.py
```
and not:
```
export SOME_FLAG=exp1
python myprogram.py
```
because the latter is not atomic and can be forgotten and additionally with environment variables this is even more potentially problematic since it'd be easy to forget you set this environment variable a few hours later and be confused why you're getting unexpected results. Avoid actions at a distance as much as possible.



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




## Handy shell shortcuts

The following bash shortcuts will save you a lot of debug time.

You need to navigate the prompt line with the command on it quickly. You can use arrows but it can be too slow if your command line is long. Instead learn these handy shortcuts:

- `Ctrl+a`: moves the cursor to the beginning of the line.
- `Ctrl+e`: moves the cursor to the end of the line.
- `Alt+f`:  moves one word forward.
- `Alt+b`:  moves one word backward.

Similarly to how arrows are slow to move the cursor, `Del` and `Backspace` are slow to delete characters. You can instead:

- `Ctrl+u`: erases everything from the current cursor position to the beginning of the line.
- `Ctrl+k`: erases everything from the current cursor position to the end of the line.




## Use bash history

Do not re-type commands you have already executed - this is a huge time wasting and you're likely to make mistakes.

Instead, rely on either an experiment cheatsheet file where you write all the commands and you copy-n-paste from, or use the bash history (or whatever shell's history that you use). I do both since some of the history gets lost when other shells write to it as well.

Most shells let you cycle through past commands with arrow up and down, and this is the fastest way to repeat recent commands, but once you need to go back 10 commands it becomes tedious. So use bash history search feature to find things faster.

It works like this. You type `Ctrl-r` and then start typing the beginning of a string that's part of the command you are looking for, For example, you type: `git`. And then you continue hitting `Ctrl-r` to cycle through all commands in the history starting with `git`.

If you already started typing the command and decided to search then, you'd then hit `Ctrl-a` to move to the beginning of the string you're searching for and then `Ctrl-r` to start search, then `Ctrl-y` to paste the substring and then again `Ctrl-r` to cycle through matches. This is not very easy to remember.

Here is the full sequence again:
```
CTRL-a # This moves our cursor to the beginning of the line.
CTRL-r # copy the current command line
CTRL-y # paste it
CTRL-r # search the history (repeat this last command)
```

Here is a even easier approach to history search. Add this:

```
$ cat ~/.inputrc
"\e[A": history-search-backward
"\e[B": history-search-forward
```
and restart `bash`.

This setup allows me to type the beginning of the command, say `git` and then hit Arrow-Up key and search through previously executed commands starting with this string - Arrow-Down key will search backwards. This is a way simpler and more intuitive.

footnote: I think this feature comes from `tcsh` where it's using `Esc-p` and `Esc-n` - but really you can bind these actions to any keys you want to e.g. `"\ep"` and `"\en"` for `Esc-p` and `Esc-n` accordingly.

Finally, you can always search the history using other tools. For example, let's search for `git`
```
$ history | grep git
 1663  git stash
 1664  git checkout main
 1665  git pull
```
and now you can either copy-n-paste the command that you need or you can even run the wanted command by the number in the first column, so we can do:
```
$ !1665
git pull
Already up to date.
```
Here Bash echo'ed the command it is about to start and run it.


Here are some other useful settings related to managing bash history related to its size, duplicate management and whether it gets rewritten on every new shell.

```
$ cat ~/.bashrc
[...]
# don't put duplicate lines in the history. See bash(1) for more options
# ... or force ignoredups and ignorespace
HISTCONTROL=ignoredups:ignorespace

# append to the history file, don't overwrite it on starting a new shell
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000
[...]
```

The other useful setting for `~/.inputrc` is:

```
$ cat ~/.inputrc
[...]
# allow new line copy with the command
set enable-bracketed-paste Off
```

So if you're managing your experiment commands in a file this will allow you to copy multiple commands at once, by keeping the new lines and not require adding `;` at the end of each command.



## Am I editing the right file and the right class?

When just starting a debugging process it might take a few attempts to realize that the wrong file is being edited. That is changes are made but it's unclear if the change took place or even if it's the right file being edited.

In the modern complex systems, you could be editing the source file in the github repo, or its installed version and then you may have 10 different versions of it installed if you have multiple virtual environments.

footnote: If you use the HF hub - it allows bundling modeling code with the model data files, and then it's even more confusing which file should be edited, since now the code is placed where you don't expect it to be. Surely, there are many other situations like this one.

So my approach to quick discovery of the right file is very simple. I purposefully break the file I think I should be editing. That is if I have a file with code like:

```
def main():
    x = 5
    y = 6
```
and I know `main` is called, I add `die`:
```
def main():
    die
    x = 5
    y = 6
```
and launch the program. If I am editing the right file the program will throw an except (and "die") and then I know I can start doing the actual debugging after removing my hack. If the program didn't die that means that either I'm not editing the right file and/or the right function.

Then there is the issue of having multiple classes or functions with the same name and sometimes it can be non-trivial to find which one is the right one. Class inheritance is another use case, as it's not obvious from looking at the code if some method is overridden or not - and which one should be edited.

The exact same solution can be used - find the class/method you think you are debugging and break it. Now you know for sure.

There is nothing special about `die` - it can be whatever string you want that will lead to breakage at run time (not compile time). I use `die` since in Perl it's an actual built-in function and so I'm used to use it there and it so happens that it actually "works" in Python too - works as in serves my intention for the program to die.

Things are a bit more complicated if the program is not interpreted but compiled, as in C/C++ languages, in which case you usually need to first run some `make` command to ensure any modified files have been recompiled. But otherwise the same principle applies - except the intentional "breaking"-process, which would be different depending on the language.

For more Python nuances see [Debugging the right Python package](../python#debugging-the-right-python-package).


## Debugging with large payloads

At some point when the software doesn't fail to run any longer you have to switch to real large data and/or large model to do quality testing.

This often can be done gradually so that you increase only one dimension. For example with the Machine Learning you have the dimensions of data size, model size, and sequence length. Thus it's enough to start getting a high quality output with just using a larger pretrained model, while keeping your data small and using a very small sequence length.

Try to find the smallest model that produces quality output first, and not the full huge model if that's the final goal.

In this use case jupyter/ipython and similar persistent environments are excellent because they allow you not to wait for the data to get loaded again and again while you do experiments.

Install the interactive python environment first (or `jupyter` if you prefer that) and let's install HF's `transformers` that we will use in the demonstration:
```
pip install ipython
pip install transformers
```

Next launch it:
```
ipython
```

Now for example let's load a large model and its tokenizer once and then re-run your debug code on it multiple times:

```
from transformers import AutoModelForCausalLM, AutoTokenizer
mname = "gpt2"
model = AutoModelForCausalLM.from_pretrained(mname)
tokenizer = AutoTokenizer.from_pretrained(mname)
```

At this point the model is loaded into the memory and now we can query it:
```
query = "Winter follows autums"
model_inputs = tokenizer([query], return_tensors="pt")
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
This prints:
```
Out[2]: 'Winter follows autums from the late 1960s to the early 1970s. The film is based on'
```
and now you can modify the query and rerun:
```
query = "The best movies are"
model_inputs = tokenizer([query], return_tensors="pt")
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
This prints:
```
Out[3]: 'The best movies are the ones that are the most interesting. The best movies are the ones that are'
```

You can bring up the previous code snippets using arrow-up, edit them and re-run again and again while avoiding the repeated overhead of loading a large model. Of course, `gpt2` is actually not that large, but this is just a demo.

jupyter is easier to use since you can have multiple code cells and so all of the code is there to edit and re-run selectively.

Though as mentioned earlier ipython/jupyter and other similar persistent interpreter environments can be tricky because the code you run isn't necessarily sequential and thus debugging mistakes can be made.
