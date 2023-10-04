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

If you repeat the same commands often - use aliases. Especially in situations when commands use multiple difficult to remember flags.

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


# Cheatsheets

While aliases are super handy, too many aliases can be difficult to remember so it's also very useful to have topical cheatsheets. I have a cheatsheet for git, python, gdb, transformers, conda, pip, bash, etc. In those cheatsheets I write very dense one line comments of what the following line does and I constantly improve them. This helps me to map things out and know where I can quickly find a specific solution.

For example here is a snipped of my git cheatsheet:

```
# ranges illustration
A ─┬─ E ── F ── G   master
   └─ B ── C ── D   fix
git log master..fix 	BCD
git log master...fix 	BCD and EFG

git log master 	      reachable parents from master
git log ^master 	  exclude reachable parents from master
git log master..fix   reachable from fix but not master
git log master...fix  reachable from fix and master, but not both
git log HEAD^@ 	      parents of HEAD
git log HEAD^! 	      HEAD, then excluding parents’s ancestors
git log HEAD^{:/fix}  search previous HEADs matching criteria


# reset branch's HEAD to a given commit hash:
# find the last commit that was supposed to be the HEAD, e.g.:
# https://github.com/fastai/fastai/commit/1c63e868d3d11e73d9f51f58cbd271e67a0fe983
# and now reset the branch's HEAD to it
git checkout release-1.0.36
git reset --hard 1c63e868d3
git push --force origin release-1.0.36

```

Here is a snippet from PyTorch cheatsheet:

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

Note how this is very dense and very easy to grasp quickly.


As I often load and troubleshoot HF models, datasets, tokenizers I have:
```
# cache transformer model + tokenizer
python -c 'import sys; from transformers import AutoModel; AutoModel.from_pretrained(sys.argv[1])' t5-small
python -c 'import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])' t5-small
python -c 'import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1])' t5-small

# cache dataset and metrics
python -c 'import sys; from datasets import load_dataset; ds=load_dataset(sys.argv[1])' stas/openwebtext-10k
```
Please note that not only it's a one-liner I can instantly copy into the shell, I make it so the ever changing name of the model or dataset is an argument, so it's trivial to replace.

I use these for testing and this method is also handy to pre-download and cache these resource.

Then I have complicated recipes like:

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

recommendation: Do not use other people's cheatsheets other than as a fodder. Make your own and format it so that you can quickly find what you need and once found you can instantly get it.


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

Big import failing - move it into a single import one liner.


```
python -c 'import sys; from transformers import AutoModel; AutoModel.from_pretrained(sys.argv[1])' t5-small
python -c 'import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])' t5-small
python -c 'import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1])' t5-small
```


## Running out of resources: disc space, cpu memory, gpu memory



### emulate an almost full partition

cd /tmp

dd if=/dev/zero of=1g.bin bs=1G count=28




### Dealing with running out of cpu memory

CPU memory:

allocate 1gb on each step and print how many were allocated
perl -le '$|=1; $gbs=10; $b="A"; $gb = $b x 2**30; $y .= $gb and print $_  for 2..$gbs; sleep 20'

meanwhile to watch RSS grow by 1gb:
watch -n 0.5 $'(ps auxc | head -1; ps auxc | grep perl | perl -plae \'$F[5]=sprintf q[%0.3fGB],$F[5]/2**20; $_=qq[@F]\') | column -t'


### Dealing with running out of gpu memory

I also created some helpful tools for pre-filling GPU memory - XXX: ipyexperiments


### Emulating limited resources


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

Tools like [PyCharm](https://www.jetbrains.com/pycharm/) have incre



## Handy shell shortcuts

The following bash shortcuts will save you a lot of debug time.

You need to navigate the prompt line with the command on it quickly. You can use arrows but it can be too slow if your command line is long. Instead learn these handy shortcuts:

- `Ctrl+a`: moves the cursor to the beginning of the line.
- `Ctrl+e`: moves the cursor to the end of the line.
- `Alt+f`:  moves one word forward.
- `Alt+b`:  moves one word backward.

Similarly to how arrows are slow to move the curser, `Del` and `Backspace` are slow to delete characters. You can instead:

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





## Create useful time saving aliases

Whenever you find
