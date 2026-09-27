#pragma once
#include <atomic>

class Spinlock
{

private:
    std::atomic<bool> lock{false};

public:
    void spin_lock()
    {
        while (lock.exchange(true))
        {
            ; // keep spinning while it is already locked
        }
        return;
    }
    void unlock()
    {
        lock.exchange(false);
    }
};