# Debugging Unix Tools


## Bash

Most of the time it's the best to start Bash scripts with:

```
#!/bin/bash
set -euo pipefail
```
where the flags are:
- set -e instructs bash to immediately exit if any command has a non-zero exit status.
- set -u instructs bash to immediately exit if any undefined variable is used (exceptions: `$*` and `$@`)
- set -o pipefail - if any command in a pipeline fails, that return code will be used as the return code of the whole pipeline and not the status of the last command.

The following sections will explain each of these options.

### Trace the execution to see commands and values

Here is a small bash script `test.sh` that does a few assignments and uses sleep to emulate a slow process:
```
$ cat << "EOT" > test.sh
#!/bin/bash
set -x
x=5
y=$(($x+5))
sleep 5
EOT

$ chmod a+x test.sh
```

If you run it:
```
$ ./test.sh
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
$ ./test.sh
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
$ ./test.sh
+ x=
+ y=5
```
you can instantly see that something is wrong with `x`.




### Abort on failures inside the script

By default Bash scripts will ignore intermediary command failures and will continue the execution, which most of the time is not what you want.

Here is a small bash script `test.sh` that has a typo in the `echo` command
```
$ cat << EOT > test.sh
#!/bin/bash
echooo "is this working"
echo "all is good"
EOT

$ chmod a+x test.sh
```

Let's run it:
```
$ ./test.sh
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
$ ./test.sh
./test.sh: line 2: echooo: command not found
```

As you can see the script exited at the point of failure. And if we check the exit code of this run:

```
$ echo $?
127
```

So not only we got the correct exit code to the caller, we can sometimes even interpret what the failure type it was.

The possible exit codes are listed [here](https://tldp.org/LDP/abs/html/exitcodes.html). and `127` corresponds to "command not found" error - which is indeed the case here. Though most of the time the time the exit code will be `1`, which is a catch all, unless the user took care to set a custom exit code.
