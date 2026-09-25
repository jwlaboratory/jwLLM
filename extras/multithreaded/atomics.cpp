#include <atomic>

std::atomic<int> balance;

bool withdraw(int amount)
{

    int old_val = balance;
    while (true)
    {
        if (old_val - amount < 0)
        {
            return false;
        }
        else
        {
            bool state = balance.compare_exchange_weak(old_val, old_val - amount);
            if (state)
            {
                return true;
            }
        }
    }
}

int main()
{
    balance = 99;
    withdraw(5);
    return 0;
}