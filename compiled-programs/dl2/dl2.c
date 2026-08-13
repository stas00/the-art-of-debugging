#include <stdio.h>

extern int util_a(void);
extern int util_b(void);

int main(void)
{
    printf("Inside main()\n");
    util_a();
    util_b();
    return 0;
}
