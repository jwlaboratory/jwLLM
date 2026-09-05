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
    Tokenizer tok(0, "data/vocab.json", "data/merges.txt");
    cout << "loaded " << tok.sToT.size() << " tokens" << endl;

    cout << "User: ";
    string user_input;

    getline(cin, user_input);
    cout << tok.decode({15, 14, 13});

    return 0;
}
