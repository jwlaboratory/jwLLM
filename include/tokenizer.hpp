#pragma once

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer
{
public:
    std::unordered_map<std::string, int> sToT;           // string to token#
    std::unordered_map<int, std::string> tToS;           // token# to string
    std::unordered_map<std::string, int> merge_priority; // merge_char1 (space) merge_char2 : what priority
    std::regex regex_splitter;

    Tokenizer(std::string _mapping_json_path, std::string _merge_txt_path);

    std::vector<int> encode(std::string in);
    std::string decode(std::vector<int> vector);
    std::vector<int> tokenize_chunk(std::string);
    std::unordered_map<int, char32_t> byte2unicode();

    std::vector<std::string> regex_split(const std::string &input, const std::regex &re);
};
