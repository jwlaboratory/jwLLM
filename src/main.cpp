#include <iostream>
#include "tokenizer.hpp"
#include "safetensors.hpp"
#include "gpt.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::flush;
using std::getline;
using std::string;
using std::vector;

int main()
{
    Tokenizer tok("data/vocab.json", "data/merges.txt");
    SafeTensors weights("data/model.safetensors");
    GPT model(weights);

    cout << "User: ";
    string user_input;
    getline(cin, user_input);

    // 1) input -> 2) tokenize
    vector<int> ids = tok.encode(user_input);

    // 3) generate one token at a time so it streams
    int max_new_tokens = 20;
    cout << "GPT: " << user_input << flush;
    for (int i = 0; i < max_new_tokens && (int)ids.size() < model.max_context(); i++)
    {
        int next = model.next_token(ids); // note we dont use generate so we can get token by token
        ids.push_back(next);
        cout << tok.decode({next}) << flush;
    }
    cout << endl;

    return 0;
}
