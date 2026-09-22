#include <pthread.h>
#include <stdio.h>
#include <iostream>
#include <atomic>
#include <mutex>

int count_till = 100000;

long counter = 0;                   // method 1
std::atomic<int> counter_atomic(0); // method 2: atomic
std::mutex m;

void *work(void *arg)
{
    int val = *(int *)arg;
    std::cout << val << "\n";

    for (int i = 0; i < count_till; i++)
    {
        std::lock_guard<std::mutex> guard(m);
        counter++;
        counter_atomic++;
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
    std::cout << counter_atomic << " final value atmoc" << std::endl;

    return 0;
}

// the best solution is no lock and split up the work so its evern faster