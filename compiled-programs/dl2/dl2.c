#include <stdio.h>

extern void util_a();
extern void util_b();

int main()
{
    printf("Inside main()\n");
    util_a();
    util_b();
    return 0;
}
