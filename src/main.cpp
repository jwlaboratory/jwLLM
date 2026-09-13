#include <iostream>
#include "matrix.hpp"
#include "tokenizer.hpp"
#include "embedding.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::getline;
using std::string;
using std::vector;

int main()
{
    Tokenizer tok("data/vocab.json", "data/merges.txt");
    Embedding embedding("data/model.safetensors");

    cout << "User: ";
    string user_input;
    getline(cin, user_input);

    // 1) input -> 2) tokenize
    Matrix tokenized_input = tok.encode(user_input);

    // 3) embed
    Matrix embedded = embedding.tokenized_to_embed(tokenized_input);

    // 4) positional encode
    embedding.apply_positional_encoding(embedded);

    return 0;
}
