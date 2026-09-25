
#include <atomic>

class Spinlock
{

private:
    std::atomic<bool> is_locked;

public:
    Spinlock()
    {
        is_locked = false;
    }

    void lock()
    {
        bool old_state = is_locked;

        while (is_locked.exchange(true)) // keep loopin guntil we can get access
        {
            // read only (faster) than write. keep on inside
            while (is_locked)
            {
            };
        };
    }

    void unlock()
    {
        is_locked = false;
    }

    bool try_lock()
    {
        bool old_state = false;
        bool state = is_locked.compare_exchange_strong(old_state, true);

        return state;
    }
};

int main()
{
    Spinlock s;

    s.lock();
    // do work
    s.unlock();

    return 0;
}