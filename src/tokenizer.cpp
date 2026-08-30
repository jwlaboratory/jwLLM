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

    // 1 apply the regex

    // 2 for each string
    // -> split into each char by char array
    // -> for loop through array
    // -> put closest 2 together, lookup
    // -> keep scores : arraypos1, arraypos2
    // -> after loop, merge lowest score, delete extra entry in array or mark it as no longer used (so its skipped)
    // complete until no more scores

    return {};
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
