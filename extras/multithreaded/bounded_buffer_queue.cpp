#include <atomic>
#include <semaphore>
#include <queue>
#include <mutex>

class BoundedBlockingQueue
{

private:
    std::counting_semaphore<1000> space_to_enter;
    std::counting_semaphore<1000> tasks_to_do;
    std::mutex m;
    std::vector<int> tasks;

public:
    BoundedBlockingQueue(int n)
    {
        space_to_enter(n);
        tasks_to_do(n);
    };

    void produe(int task_id)
    {
        // block if no space, else add to queue
        space_to_enter.p();

        m.lock();
        tasks.push_back(task_id);
        m.unlock();

        tasks_to_do.v();
    };

    int consume()
    {
        // if nothing to do, block
        tasks_to_do.s();

        // let them know more space and rmeove task

        m.lock() int task_id = tasks.front();
        tasks.pop_back();
        m.unlock();

        space_to_enter.v();

        return task_id;
    };
};