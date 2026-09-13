#pragma once

#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "matrix.hpp"

class SafeTensors
{
public:
    SafeTensors(std::string file_path_safetensors);
    Matrix get(const std::string &name); // e.g. "wte.weight", "h.0.attn.c_attn.weight"
    bool has(const std::string &name);

private:
    std::ifstream f;
    uint64_t header_len;
    nlohmann::json header_json;
};
