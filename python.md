# Debugging Python Programs


## auto-print what's being observed

Say, you want to see the output of summation:

```
$ python -c 'x=5; y=6; print(f"{x+y}")'
11
```
But then you don't know what's being summed in the printout, so you have to write then:

```
$ python -c 'x=5; y=6; print(f"x+y={x+y}")'
x+y=11
```
But this is both long and error-prone, because the 2 parts aren't atomic - you may choose to modify the expression in the second part `{x+y+1}`, but forget to update the first part and end up with wrong conclusions.

Since python-3.8 there is an atomic operand auto-description feature. Let's rewrite the last one liner to remove the description of what's being printed and append a magical `=` to the expression inside `{}`:
```
$ python -c 'x=5; y=6; print(f"{x+y=}")'
x+y=11
```
Now what's being evaluated is automatically printed. All you need to do is to add `=` before the closing `}`.

Here is another example:
```
$ python -c 'x=5; y=6; print(f"{x=}, {y=}")'
x=5, y=6
```

So once again you can see that atomic operations are ideal for fruitful debugging.
