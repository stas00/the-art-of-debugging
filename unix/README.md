# Debugging Unix Tools


## Terminal

### Searching terminal's output

Teeing all outputs to `| tee -a log` allows you to later search the outputs, but there are times when you want to search the outputs dumped to the terminal. Therefore it's critical to use a terminal (console) that allows you to search it.

For example, in `konsole` this is `Ctrl-Shift-f`, and it provides a rich searching functionality - normal/regex/highlighting matching/search direction.

To be able to use search when there is a lot of output you want to make sure that your scrollback buffer (i.e. how many lines the terminal shows before it truncates) to a large number - e.g. I have mine set to 20k lines.

Since multiple runs of the same program may produce different outputs it may be confusing to search the terminal outputs since it's hard to tell which is which. In this situation, I call `clear` which clears all outputs in the terminal, before invoking a new command. So that my debug cycle looks like:

```
clear; ./myprogram --args ...
```
So that it's atomic and I never forget to clear the screen. Then the searchable output is always of the last run.

At other times I don't `clear`, since I do want to search previous results.



## Bash


While the instructions use Bash If you use a different shell a lot of the suggestions will be quite similar with small variations for some settings being different, but the key here is to grasp the concepts and then translate them to the other shell's environment.


### Controlling script execution

TLDR: Most of the time it's the best to start Bash scripts with:

```
#!/bin/bash
set -euo pipefail
# the rest of the script here
```
where the flags are:
- `set -e` instructs Bash to immediately exit if any command has a non-zero exit status.
- `set -u` instructs Bash to immediately exit if any undefined variable is used (exceptions: `$*` and `$@`)
- `set -o pipefail` - if any command in a pipeline fails, that return code will be used as the return code of the whole pipeline and not the status of the last command.

and if you want to see what the script is doing:
- `set -x` instructs Bash to print out each command it runs with its values

```
#!/bin/bash
set -x
# the rest of the script here
```
The following sections will explain each of these options.

#### Trace the execution to see commands and values

Here is a small bash script `test.sh` that does a few assignments and uses sleep to emulate a slow process:
```
$ cat << "EOT" > test.sh
#!/bin/bash
set -x
x=5
y=$(($x+5))
sleep 5
EOT
```

If you run it:
```
$ bash ./test.sh
```
you have no idea what it is doing.

Let's turn the execution tracer on by adding `set -x`:

```
$ cat << "EOT" > test.sh
#!/bin/bash
set -x
x=
y=$(($x+5))
sleep 5
EOT
```

If you run it:
```
$ bash ./test.sh
+ x=5
+ y=10
+ sleep 5
```
you can now see exactly what it is doing, it'll show every command before it's running it.

Additionally, you can see the values as they get modified. e.g. here we got to see that `y=x+5` resulted in 10. Which is super useful for debugging since you can actually see what the values are. Let's say `x` didn't get set for whatever reason:
```
$ cat << "EOT" > test.sh
#!/bin/bash
set -x
x=
y=$(($x+5))
EOT
```

If you run it:
```
$ bash ./test.sh
+ x=
+ y=5
```
you can instantly see that something is wrong with `x`.




#### Abort on failures inside the script

By default Bash scripts will ignore intermediary command failures and will continue the execution, which most of the time is not what you want.

Here is a small bash script `test.sh` that has a typo in the `echo` command
```
$ cat << EOT > test.sh
#!/bin/bash
echooo "is this working"
echo "all is good"
EOT
```

Let's run it:
```
$ bash ./test.sh
./test.sh: line 2: echooo: command not found
all is good
```

So despite, `echooo` command failing, the script continues. Additionally if we check the exit code of this run:

```
$ echo $?
0
```

It indicates to the caller that the script finished successfully, when it didn't.

Let's fix that by adding: `set -e` which will now abort the execution of the script on the first error:
```
$ cat << EOT > test.sh
#!/bin/bash
set -e
echooo "is this working"
echo "all is good"
EOT
```


Let's run it:
```
$ bash ./test.sh
./test.sh: line 2: echooo: command not found
```

As you can see the script exited at the point of failure. And if we check the exit code of this run:

```
$ echo $?
127
```

So not only we got the correct exit code to the caller, we can sometimes even interpret what the failure type it was.

The possible exit codes are listed [here](https://tldp.org/LDP/abs/html/exitcodes.html). and `127` corresponds to "command not found" error - which is indeed the case here. Though most of the time the time the exit code will be `1`, which is a catch all, unless the user took care to set a custom exit code.



#### Abort on failures in the pipeline

By default Bash scripts will ignore intermediary stage failures in the pipe `|` commands and will continue the execution, which again most of the time is not what you want.

```
$ cat << EOT > test.sh
#!/bin/bash
set -e
echooo "is this working" | sort
echo "all is good"
EOT
```

Let's run it:
```
$ bash ./test.sh
./test.sh: line 3: echooo: command not found
all is good
```

So despite, `echooo` command failing, the script continues. Additionally if we check the exit code of this run:

```
$ echo $?
0
```

But wait a second, didn't the previous section explain that setting `set -e` should abort the script at the first problem and set `$?` to a non-0 state?

The problem is the `|` pipe. Since `sort` is successful it masks the failure and bash isn't the wiser that something has failed.

So we also have to add `set -o pipefail` for this to do what I mean:

```
$ cat << EOT > test.sh
#!/bin/bash
set -eo pipefail
echooo "is this working" | sort
echo "all is good"
EOT
```

Let's run it:
```
$ bash ./test.sh
./test.sh: line 3: echooo: command not found
```
So we can see that the program aborted at the broken command

```
$ echo $?
127
```
and the exit code is again non-0.




#### Abort on undefined variables

By default Bash scripts will ignore undefined variables used to construct new variables:
```
$ cat << "EOT" > test.sh
#!/bin/bash
x="xxx"
y="yyy"
z="$X $y"
echo $z
EOT
```

We hope to get the output of `xxx yyy` as the value of `$z`.

Let's run it:
```
$ bash ./test.sh
yyy
```

If you haven't noticed I made an intentional mistake replacing `$x` with `$X`. You can see the former being defined, but not the latter. Yet, the program runs without errors and generates a non-intended output `yyy`.

If we add `set -u` now Bash will be strict about all variables needing to be defined before they can be used.
```
$ cat << "EOT" > test.sh
#!/bin/bash
set -u
x="xxx"
y="yyy"
z="$X $y"
echo $z
EOT
```

Let's run it:
```
$ bash ./test.sh
./test.sh: line 5: X: unbound variable
```
So we can see that the program aborted at line 5 `z="$X $y"` since `$X` is indeed undefined.

I thought it'd work for using `eval` to do math as well, but the `eval` seems to run in its own space where `set -u` couldn't reach. i.e. this code doesn't fail:
```
set -u
x=
y=$(($x+5))
echo y=$y
```



#### Temporarily turning off set commands

Once any of the `set` commands have been enabled if you have an area where you need to disable the guards, you simply use the `set +` setting, for example, let's demo with `set -e`:

```
$ cat << EOT > test.sh
#!/bin/bash
set -e
# failing is not ok
echo "prep"
set +e
# failing is ok
echooo "is this working"
set -e
# failing is again not ok
echooo "is this working 2"
echo "all is good"
EOT
```

Let's run it:
```
$ bash ./test.sh
prep
./test.sh: line 5: echooo: command not found
./test.sh: line 7: echooo: command not found
```

As you can see the broken command on line 5 didn't abort the script, due to `set +e`, but the one at line 7 did, due to `set -e`.




### Being able to copy-n-paste multi-lines

At times I copy-n-paste multiple commands that include new line separators. I wanted this to work correctly and therefore I have this in my `~/.inputrc`:
```
set enable-bracketed-paste Off
```

This setting allows new line copied with the command being pasted instead of making them disappear. You need to restart `bash` for this setting to take an effect.

footnote: `man bash` for more information and if you're using a different shell check its manpage for the equivalent setting.


### Informative prompt

XXX:
