#include <pthread.h>
#include <stdio.h>
#include <iostream>

int count_till = 100000;
long counter = 0;

void *work(void *arg)
{
    int val = *(int *)arg;
    std::cout << val << "\n";

    for (int i = 0; i < count_till; i++)
    {
        counter++;
    }

    return NULL;
}

int main()
{
    int threads_total = 4;

    pthread_t all_threads[threads_total];
    int ids[threads_total];
    for (int i = 0; i < threads_total; i++)
    {
        ids[i] = i;
        pthread_create(&all_threads[i], NULL, work, (void *)(&ids[i]));
    }
    for (int i = 0; i < threads_total; i++)
    {
        pthread_join(all_threads[i], NULL);
    }

    std::cout << counter << " final value" << std::endl;

    return 0;
}
