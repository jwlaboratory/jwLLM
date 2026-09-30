#include <mutex>
#include <condition_variable>

class RWLock
{
    std::mutex global_lock;
    int num_readers = 0;
    int num_writers = 0;
    std::condition_variable_any canRead;
    std::condition_variable_any canWrite;

public:
    void read()
    {
        // LET THEM KNOW WE ARE HERE TO READ
        global_lock.lock();

        while (num_writers != 0)
        {
            canRead.wait(global_lock);
        }

        num_readers += 1;
        global_lock.unlock();

        // ACTUALLY READ

        // LET THEM KNOW WE ARE DONE READING
        global_lock.lock();
        num_readers -= 1;
        if (num_readers == 0)
        {
            canWrite.notify_one();
        }
        global_lock.unlock();
    }
    void write()
    {
        // LET THEM KNOW WE ARE HERE TO READ
        global_lock.lock();
        while (num_readers != 0 || num_writers != 0)
        {
            canWrite.wait(global_lock);
        }
        num_writers += 1;
        global_lock.unlock();

        // ACTUALLY WRITE

        // let them know done writing
        global_lock.lock();
        num_writers -= 1;
        canRead.notify_all();
        canWrite.notify_one();
        global_lock.unlock();
    }
};