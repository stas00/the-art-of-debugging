# The Art of Debugging

This guide will teach you how to:
1. **Debug normal issues really fast**
2. **Make complicated issues possible to debug**

footnote: adapted from Perl's slogan: "Easy things should be easy and hard things should be possible".

This is a work-in-progress collection of methodologies and copy-n-paste recipes for successful debugging of simple and complicated software problems. Some sections are quite complete, while other will be finished at a later stage, and yet other haven't been started.


## Intro

I have been developing software since 1995 and a lot of this work involved debugging. Over the years I developed a various efficient methodologies for discovering the source of the problem, which is the most difficult stage to solving it. Since after the problem is understood, typically the resolution is relatively easy.

Every so often someone I would be debugging a problem with would suggest to share my approaches with the world. I always said that it'd be too difficult to generalize, but recently the planted seed seems to have sprouted and so in the following documents I will try to share some of the insights to ease this very difficult at times process.

Writing about debugging in the void is very difficult and since I haven't been saving use cases, it will take some time to build this up, so expect these pages to be a Work In Progress (WIP) for many moons. But hopefully some ideas could be relayed to you sooner than later, and they would help to ease your burden of debugging in your work and play projects.


## Table of Contents

1. **[Fast Debugging Methodology](./methodology/)**

2. **[Debugging Compiled Programs](./compiled-programs/)** - `gdb`, `ldd`, `nm`, `LD_LIBRARY_PATH`, `LD_PRELOAD`

3. **[Debugging Python](./python/)** - `py-spy`, paths, auto-print

4. **[Unix Tools For Debugging](./unix/)** - `bash`, `strace`, `make`, prompt, `nohup`

5. **[Debugging Machine Learning Projects](https://github.com/stas00/ml-engineering/tree/master/debug)** (external)



## Contributing

If you found a bug, typo or would like to propose an improvement please don't hesitate to open an [Issue](https://github.com/stas00/the-art-of-debugging/issues) or contribute a PR.



## License

The content of this site is distributed under [Attribution-ShareAlike 4.0 International](./LICENSE-CC-BY-SA).


## My repositories map

✔ **Machine Learning:**
 [ML Engineering](https://github.com/stas00/ml-engineering) |
 [ML ways](https://github.com/stas00/ml-ways) |
 [Porting](https://github.com/stas00/porting)

✔ **Guides:**
 [The Art of Debugging](https://github.com/stas00/the-art-of-debugging)

✔ **Applications:**
 [ipyexperiments](https://github.com/stas00/ipyexperiments)

✔ **Tools and Cheatsheets:**
 [bash](https://github.com/stas00/bash-tools) |
 [conda](https://github.com/stas00/conda-tools) |
 [git](https://github.com/stas00/git-tools) |
 [jupyter-notebook](https://github.com/stas00/jupyter-notebook-tools) |
 [make](https://github.com/stas00/make-tools) |
 [python](https://github.com/stas00/python-tools) |
 [tensorboard](https://github.com/stas00/tensorboard-tools) |
 [unix](https://github.com/stas00/unix-tools)
