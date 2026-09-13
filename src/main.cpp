#include <iostream>
#include "matrix.hpp"
#include "tokenizer.hpp"
#include "embedding.hpp"
#include "safetensors.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::getline;
using std::string;
using std::vector;

int main()
{
    Tokenizer tok("data/vocab.json", "data/merges.txt");
    SafeTensors weights("data/model.safetensors");
    Embedding embedding(weights);

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

// Tensor	Shape	What it is
// ln_1.weight	768	Layernorm gamma, before attention
// ln_1.bias	768	Layernorm beta, before attention
// attn.c_attn.weight	768 x 2304	Fused query, key, value projection
// attn.c_attn.bias	2304	Bias for the above
// attn.c_proj.weight	768 x 768	Output projection after head concat
// attn.c_proj.bias	768	Bias for the above
// ln_2.weight	768	Layernorm gamma, before the MLP
// ln_2.bias	768	Layernorm beta, before the MLP
// mlp.c_fc.weight	768 x 3072	Expand to 4x width
// mlp.c_fc.bias	3072	Bias for the above
// mlp.c_proj.weight	3072 x 768	Contract back to 768
// mlp.c_proj.bias	768	Bias for the above
// attn.bias	1 x 1 x 1024 x 1024	Not a bias. Ignore it.