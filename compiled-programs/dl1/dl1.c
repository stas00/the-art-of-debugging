#include <stdio.h>

extern int util_a(void);
int main(void)
{
    printf("Inside main()\n");
    util_a();

    return 0;
}
