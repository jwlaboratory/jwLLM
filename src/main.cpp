#include <iostream>

#include "tokenizer.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::getline;
using std::string;
using std::vector;

int main()
{
    Tokenizer tok("data/vocab.json", "data/merges.txt");
    cout << "loaded " << tok.sToT.size() << " tokens" << endl;

    cout << "User: ";
    string user_input;

    getline(cin, user_input);
    std::vector<int> tokenized_input = tok.encode(user_input);
    for (int tok : tokenized_input)
    {
        cout << tok << " ";
    }

    // tokenize
    // embed
    // apply positional encoding

    // loop 12 times
    // transformer
    // FFN

    // layernorm, final language head

    return 0;
}
