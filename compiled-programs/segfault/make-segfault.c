#include <stdio.h>

int main(void)
{
    int* ptr;
    printf("%d", *ptr); // dereference an uninitialized pointer
    return (0);
}
