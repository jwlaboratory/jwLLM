
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

    WORD_TOKEN_EMBEDDING = load_tensor("transformer.wte.weight");
    WORD_POSITIONAL_EMBEDDING = load_tensor("transformer.wpe.weight");
}
