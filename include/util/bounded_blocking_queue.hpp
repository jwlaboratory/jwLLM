#pragma once
#include <atomic>
#include "semaphore.hpp"
#include "spinlock.hpp"
#include <queue>
#include <string>

class BoundedBlockingQueue
{

private:
    Semaphore count_live_work;
    Semaphore count_avail_slots;
    Spinlock gaurd_lock; // We'd rather use a spinlock. Because we know the critical section here is short and that low contention with large queue

    std::queue<std::string> work_queue;

public:
    BoundedBlockingQueue(int capacity)
    {
        count_avail_slots.count = capacity;
        count_live_work.count = 0;
    }

    void produce()
    {
        count_avail_slots.P(); // block unless we have space to give work

        gaurd_lock.spin_lock();
        work_queue.push("WORK");
        gaurd_lock.unlock();

        count_live_work.V(); // let them know we have work to consume
    }

    void consume()
    {
        count_live_work.P(); // block unless we have work to do

        gaurd_lock.spin_lock();
        std::string work_to_do = work_queue.front();
        work_queue.pop();
        gaurd_lock.unlock();

        // let them know we have more space for work
        count_avail_slots.V();
    }

    bool consume_non_blocking()
    {
        // check if we have work to do
        bool got_lock = count_live_work.P_no_wait();
        if (!got_lock)
        {
            return false;
        }

        // normal code now
        gaurd_lock.spin_lock();
        std::string work_to_do = work_queue.front();
        work_queue.pop();
        gaurd_lock.unlock();

        // let them know we have more space for work
        count_avail_slots.V();
        return true;
    }
};
