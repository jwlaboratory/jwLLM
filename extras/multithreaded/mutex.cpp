#include <atomic>
#include <queue>

// we can't do it fr because we dont have access to thread scheduling in this progrma
struct thread
{
    int pid;
    bool sleeping{false};

    // same issue as petersons. sleeping should be atomic otherwise buffer might just check
};

class Mutex
{

private:
    std::atomic<bool> spinning_gaurd_lock{false};
    std::queue<struct thread *> waiters;
    bool locked = false;

public:
    void
    lock(struct thread *thread)
    {
        // aquire quick temp lock
        acquireSpinlockGaurad();

        if (!locked)
        {
            locked = true;
        }
        else
        {
            waiters.push(thread);
            thread->sleeping = true;
        }

        spinning_gaurd_lock.exchange(false);

        // fake sleep. in reality CPU would schedule a new task
        while (thread->sleeping)
            ;
    };

    void unlock(struct thread *calling_thread)
    {
        acquireSpinlockGaurad();
        locked = false;

        if (waiters.size() > 0)
        {
            struct thread *cur = waiters.front();
            waiters.pop();
            cur->sleeping = false;
            locked = true;
        }

        spinning_gaurd_lock.exchange(false);
    };

    void acquireSpinlockGaurad()
    {

        while (spinning_gaurd_lock.exchange(true))
        {
            ;
        }
        return;
    };
};