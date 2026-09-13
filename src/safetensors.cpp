#include "safetensors.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace std;

SafeTensors::SafeTensors(std::string file_path_safetensors)
{

    f.open(file_path_safetensors, std::ios::binary);
    if (!f.is_open())
    {
        throw runtime_error("could not open mapping");
    };

    // get the first 8 bytes tells us the length
    header_len = 0;
    f.read(reinterpret_cast<char *>(&header_len), 8);
    // assume same endiness?

    // get the header now
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    header_json = json::parse(header);
}

bool SafeTensors::has(const std::string &name)
{
    return header_json.contains(name);
}

Matrix SafeTensors::get(const std::string &name)
{
    if (!has(name))
    {
        throw runtime_error("no tensor named " + name);
    }

    auto tensor_info = header_json[name];
    std::vector<int> shape = tensor_info["shape"]; // e.g., {50257, 768}, or {768} for a bias
    uint64_t data_begin = tensor_info["data_offsets"][0];
    uint64_t data_end = tensor_info["data_offsets"][1];
    uint64_t tensor_bytes = data_end - data_begin;
    f.seekg(8 + header_len + data_begin, std::ios::beg);      // skip header length + header size declaration + until data begin
    std::vector<float> weights(tensor_bytes / sizeof(float)); // total number of bytes / size of a float, so we knwo how long the array should be
    f.read(reinterpret_cast<char *>(weights.data()), tensor_bytes);

    // 1d tensors (biases, layernorm gamma/beta) come back as a single row
    if (shape.size() == 1)
    {
        return Matrix(1, shape[0], weights);
    }
    return Matrix(shape[0], shape[1], weights);
}
