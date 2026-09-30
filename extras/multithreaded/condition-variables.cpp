#include <thread>
#include <stdio.h>
#include <iostream>
#include <atomic>
#include <mutex>
#include <condition_variable>

std::atomic<bool> thread1_completed{false};
std::mutex lock;
std::condition_variable done_with_thread;

void func1()
{
    // do work
    lock.lock();
    thread1_completed = true;
    done_with_thread.notify_all();
    lock.unlock();
}

void func2()
{
    while (!thread1_completed)
    {
        done_with_thread.wait();
    }
    return nullptr;
}

int main()
{
    std::thread t2(func2); // start func2 first to show it really waits
    std::thread t1(func1);
    t1.join();
    t2.join();
}