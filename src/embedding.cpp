
#include "matrix.hpp"
#include "embedding.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace std;

Embedding::Embedding(std::string file_path_safetensors)
{

    ifstream f(file_path_safetensors, std::ios::binary);
    if (!f.is_open())
    {
        throw runtime_error("could not open mapping");
    };

    // get the first 8 bytes tells us the length
    uint64_t header_len = 0;
    f.read(reinterpret_cast<char *>(&header_len), 8);
    // assume same endiness?

    // get the header now
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    json header_json = json::parse(header);

    auto load_tensor = [&](const std::string &name)
    {
        auto tensor_info = header_json[name];
        std::vector<int> shape = tensor_info["shape"]; // e.g., {50257, 768}
        uint64_t data_begin = tensor_info["data_offsets"][0];
        uint64_t data_end = tensor_info["data_offsets"][1];
        uint64_t tensor_bytes = data_end - data_begin;
        f.seekg(8 + header_len + data_begin, std::ios::beg);      // skip header length + header size declaration + until data begin
        std::vector<float> weights(tensor_bytes / sizeof(float)); // total number of bytes / size of a float, so we knwo how long the array should be
        f.read(reinterpret_cast<char *>(weights.data()), tensor_bytes);
        return Matrix(shape[0], shape[1], weights);
    };

    WORD_TOKEN_EMBEDDING = load_tensor("wte.weight");
    WORD_POSITIONAL_EMBEDDING = load_tensor("wpe.weight");
}

Matrix Embedding::tokenized_to_embed(const Matrix &token_ids)
{
    // token_ids is a single sequence: 1 row, seq_len cols, ids stored as floats.
    int d_model = WORD_TOKEN_EMBEDDING.cols;
    int seq_len = token_ids.cols;
    std::vector<float> out(seq_len * d_model);

    for (int i = 0; i < seq_len; i++)
    {
        int token_id = static_cast<int>(token_ids.data[i]);

        for (int g = 0; g < d_model; g++)
        {
            out[i * d_model + g] = WORD_TOKEN_EMBEDDING.data[token_id * d_model + g];
            // out is flat array, word_token embedding is also flat array
        }
    }

    return Matrix(seq_len, d_model, out);
}

void Embedding::apply_positional_encoding(Matrix &token_embeddings)
{
    // seq len rows
    // dmodel cols

    // WORD_POSITIONAL_EMBEDDING is a table of size: position rows, dmodel cols

    int seq_len = token_embeddings.rows;
    int dmodel = token_embeddings.cols;

    for (int i = 0; i < seq_len; i++)
    {
        for (int g = 0; g < token_embeddings.cols; g++)
        {
            token_embeddings.data[i * dmodel + g] += WORD_POSITIONAL_EMBEDDING.data[i * dmodel + g];
        }
    }
}
