#include <atomic>
#include <queue>
#include <mutex>
#include <semaphore>

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

class sempahore
{

public:
    sempahore(int initial)
    {
        val = initial;
    }

    void p(struct thread *curThread)
    {
        mtx.lock();

        if (val > 0)
        {
            val--;
            mtx.unlock();
            return;
        };

        // we need to put this to sleep and wait for a signal
        curThread->sleeping = true;
        waiters.push(curThread);
        mtx.unlock();

        // now we'll switch to a different thread, but in sim
        while (curThread->sleeping == true)
            ;
    }

    void v(struct thread *callerThread)
    {
        mtx.lock();

        if (waiters.size() != 0)
        {
            struct thread *top = waiters.front();
            waiters.pop();
            top->sleeping = false;
        }
        else
        {
            val++;
        }
        mtx.unlock();
    }

private:
    std::mutex mtx;
    int val;
    std::queue<struct thread *> waiters;
};

int main()
{
    // Print in Order (LeetCode 1114)

    sempahore a(0);
    semaphore b(0);

    // thread A
    // code
    // a.v() --> release/signal

    // thread B:
    //  a.p()
    //  code
    //  b.v()

    // thread C
    //  b.p()
    // code

    return 0;
}