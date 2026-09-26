#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <sys/wait.h>

int main()
{
    std::vector<int> vector_to_compute(99999, 2);

    size_t length = 2;

    int *addr = (int *)mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    if (addr == MAP_FAILED)
    {
        perror("mmap failed");
        exit(1);
    }

    pid_t pid = fork();
    if (pid < 0)
    {
        exit(1);
    }

    if (pid == 0)
    {
        // child
        int lsum = 0;
        for (int i = 0; i < (vector_to_compute.size() + 1) / 2; i++)
        {
            lsum += vector_to_compute[i];
        }

        addr[0] = lsum;
        exit(0);
    }
    else
    {
        // rsum
        int rsum = 0;
        for (int i = (vector_to_compute.size() + 1) / 2; i < vector_to_compute.size(); i++)
        {
            rsum += vector_to_compute[i];
        }

        addr[1] = rsum;

        waitpid(pid, NULL, 0);

        printf("output %d", (addr[0] + addr[1]));
    }

    if (munmap(addr, length) == -1)
    {
        perror("munmap failed");
        exit(1);
    }
    return 0;
}