#include "tokenizer.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

using namespace std;

Tokenizer::Tokenizer(unsigned int merge_list_size, string _mapping_json_path, string _merge_txt_path)
{
    ifstream f(_mapping_json_path);
    if (!f.is_open())
    {
        throw runtime_error("could not open mapping");
    };

    json j;
    f >> j;

    for (auto &[key, value] : j.items())
    {
        int id = value.get<int>();
        sToT[key] = id;
        tToS[id] = key;
    }
    f.close();

    // merge list
    ifstream merge_f(_merge_txt_path);
    if (!merge_f.is_open())
    {
        throw runtime_error("could not open merge path");
    }

    string merge_line;
    int cur_priority = 0;
    while (getline(merge_f, merge_line))
    {
        // Output the text from the file
        merge_priority[merge_line] = cur_priority;
        cur_priority += 1;
    }
    merge_f.close();
    regex_splitter = std::regex("'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+");
}

vector<int> Tokenizer::encode(string in)
{
    std::vector<int> final_tokens;

    // 1 apply the regex
    std::vector<std::string> split_input_string = regex_split(in, regex_splitter);

    for (std::string chunk : split_input_string)
    {
        // run per chunk the tokenization

        std::vector<int> tokenized_chunk = tokenize_chunk(chunk);

        for (int tokenized_chunk_nums : tokenized_chunk)
        {
            final_tokens.push_back(tokenized_chunk_nums);
        }
    }
    return final_tokens;

    return {};
}

std::vector<int> Tokenizer::tokenize_chunk(std::string)
{
    // we should keep a priority queue that keeps the adjacent tokens and the score
    // -> put closest 2 together, lookup
    // -> keep scores : arraypos1, arraypos2
    // -> after loop, merge lowest score, delete extra entry in array or mark it as no longer used (so its skipped)
    // complete until no more scores

    struct MergeCandidate
    {
        int priority_score;
        int left_index;
        int right_index;

        bool operator>(const MergeCandidate &other) const
        {
            return priority_score > other.priority_score;
        }
    };
}

string Tokenizer::decode(vector<int> vector)
{
    string s;
    for (auto &v : vector)
    {
        s += tToS[v];
    }
    return s;
}

// from the internet: function to split into array based on regex
std::vector<std::string> regex_split(const std::string &input, const std::regex &re)
{
    // Pass 0 instead of -1 to capture the actual regex matches (tokens)
    std::sregex_token_iterator first{input.begin(), input.end(), re, 0};
    std::sregex_token_iterator last;

    std::vector<std::string> tokens;
    for (auto it = first; it != last; ++it)
    {
        if (!it->str().empty())
        {
            tokens.push_back(*it);
        }
    }
    return tokens;
}