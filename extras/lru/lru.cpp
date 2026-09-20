#include <cstdio>
#include <iostream>
#include <unordered_map>

struct Node
{
    struct Node *next;
    struct Node *prev;
    int key;
    int val;
};

class LRU
{

private:
    int capacity = 0;
    std::unordered_map<int, struct Node *> hashmap_lookup = {};
    struct Node MRU = {};

    void move_to_MRU(struct Node *node)
    {
        struct Node *prev = node->prev;
        struct Node *next = node->next;
        prev->next = next;
        next->prev = prev;

        // then move our node to front
        node->prev = &MRU;
        node->next = MRU.next;
        MRU = *node;
    };

public:
    LRU(int capacity)
    {
        if (capacity <= 0)
        {
            throw std::invalid_argument("capacity must be > 0");
        }
        this->capacity = capacity;
        MRU.next = &MRU;
        MRU.prev = &MRU;
    };

    int get(int key)
    {
        if (hashmap_lookup.find(key) != hashmap_lookup.end())
        {
            // move to front (MRU)
            move_to_MRU(hashmap_lookup.at(key));
            return hashmap_lookup.at(key)->val;
        }
        else
        {
            return -1;
        }
    }

    void put(int key, int val)
    {
        if (hashmap_lookup.find(key) != hashmap_lookup.end())
        {
            hashmap_lookup.at(key)->val = val;
            move_to_MRU(hashmap_lookup.at(key));
            return;
        }

        if (hashmap_lookup.size() == capacity)
        {
            // assuming more than 1 element
            struct Node *node_to_remove = MRU.prev;
            struct Node *prev = node_to_remove->prev;
            prev->next = &MRU;
            MRU.prev = prev;

            hashmap_lookup.erase(node_to_remove->key);
            delete node_to_remove;
            // evict from LRU
        }

        // NOTE THIS WONT WORK! WE NEED ON HEAP NOT ON STACK SO IT PERSISTS
        // struct Node newNode = {.key = key, .val = val, .prev = NULL, .next = NULL};

        struct Node *newNode = new struct Node(
            {nullptr, nullptr, key, val});

        hashmap_lookup[key] = newNode;
        move_to_MRU(hashmap_lookup.at(key));

        // add to MRU
    }
};

int main()
{
    LRU cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    std::cout << cache.get(1) << "\n"; // expect 1
    cache.put(3, 3);                   // evicts key 2
    std::cout << cache.get(2) << "\n"; // expect -1
}