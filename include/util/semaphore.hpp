#pragma once
#include <atomic>
#include "spinlock.hpp"

class Semaphore
{

private:
    Spinlock gaurd_lock;

public:
    std::atomic<int> count;

    Semaphore()
    {
    }
    Semaphore(int n)
    {
        count = n;
    }

    void P()
    {
        while (true) // while cos we could think we have capacity after wait but it gets taken by someone else
        {
            gaurd_lock.spin_lock();
            // lock first, so we can check if the count is good and change it

            if (count > 0)
            {
                count -= 1;
                gaurd_lock.unlock();
                return;
            }

            // put to sleep
            gaurd_lock.unlock();
            // put this thread to sleep and wait for count to be > 0;
            count.wait(0); // waitif currently at 0 (which is true)
        }
    };

    bool P_no_wait()
    {
        gaurd_lock.spin_lock();
        // lock first, so we can check if the count is good and change it

        if (count > 0)
        {
            count -= 1;
            gaurd_lock.unlock();
            return true;
        }

        gaurd_lock.unlock();
        return false;
    };

    void V()
    {
        gaurd_lock.spin_lock();
        count += 1;
        gaurd_lock.unlock();

        count.notify_one();
    };
};