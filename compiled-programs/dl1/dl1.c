#include <stdio.h>

extern void util_a();
int main()
{
    printf("Inside main()\n");
    util_a();

    return 0;
}
