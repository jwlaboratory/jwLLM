#include "tokenizer.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

using namespace std;

Tokenizer::Tokenizer(string _mapping_json_path, string _merge_txt_path)
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
    regex_splitter = std::regex(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)");
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
}

std::vector<int> Tokenizer::tokenize_chunk(std::string chunk)
{
    // we should keep a priority queue that keeps the adjacent tokens and the score
    // -> put closest 2 together, lookup
    // -> keep scores : arraypos1, arraypos2
    // -> after loop, merge lowest score, delete extra entry in array or mark it as no longer used (so its skipped)
    // complete until no more scores

    // struct MergeCandidate
    // {
    //     int priority_score;
    //     string str;
    // };

    // vector<struct MergeCandidate> all_candidates;

    // for (int i = 0; i < chunk.size() - 1; i++)
    // {
    //     std::string candidate_as_string = chunk[i] + " " + chunk[i + 1];

    //     int prio;
    //     if (merge_priority.find(candidate_as_string) == merge_priority.end())
    //     {
    //         prio = -1;
    //     }
    //     else
    //     {
    //         prio = merge_priority[candidate_as_string];
    //     }

    //     struct MergeCandidate candidate = {.priority_score = prio, .str = candidate_as_string};
    //     all_candidates.push_back(candidate);
    // }

    // // main loop
    // bool finished_merges = false;

    // while (!finished_merges)
    // {
    //     int lowest_prio = 9999;
    //     int lowest_prio_index = -1;

    //     for (int i = 0; i < all_candidates.size(); i++)
    //     {
    //         if (all_candidates[i].priority_score != -1 && all_candidates[i].priority_score < lowest_prio)
    //         {
    //             lowest_prio = all_candidates[i].priority_score;
    //             lowest_prio_index = i;
    //         }
    //     }

    //     // now we want to check
    //     if (lowest_prio_index == -1)
    //     {
    //         finished_merges = true;
    //         break;
    //     }
    //     else
    //     {
    //         // here we have at least one merge to make

    //         // case 1: first token
    //         if (lowest_prio_index == 0)
    //         {
    //             // delete this, update index lowest_prio_index+1
    //             all_candidates[lowest_prio_index + 1].left_token_left_index = all_candidates[lowest_prio_index].left_token_left_index;
    //             // reindex

    //             all_candidates[lowest_prio_index + 1].priority_score = new_prio;

    //             all_candidates.erase(all_candidates.begin() + lowest_prio_index);
    //         }
    //         else if (lowest_prio_index == all_candidates.size() - 1)
    //         {
    //             // last index case
    //             all_candidates[lowest_prio_index - 1].right_index_right_index = all_candidates[lowest_prio_index].right_index_right_index;
    //             all_candidates.erase(all_candidates.begin() + lowest_prio_index);
    //         }
    //         else
    //         {
    //             // normal case
    //         }

    //         // #, #, #, #
    //         // a, b, c, d
    //         // a-b, b-c, c-d
    //         // bc is lowest. then itll be a-bc, bc-d
    //         // if cd is lowest. then itll be a-b, b-cd
    //         // if ab is lowest. then itll be ab-c, c-d
    //     }
    // }

    // lookup each token

    // return final answ;

    std::vector<std::string> symbols;
    for (size_t i = 0; i < chunk.size(); i++)
    {
        symbols.push_back(std::string{chunk[i]});
    }

    while (symbols.size() > 1)
    {
        // find lowest score index
        int lowest_index = -1;
        int lowest_pri = 9999999;

        for (size_t i = 0; i < symbols.size() - 1; i++)
        {
            string candidate = symbols[i] + " " + symbols[i + 1];

            int prio;
            if (merge_priority.find(candidate) != merge_priority.end())
            {
                prio = merge_priority[candidate];
                if (prio < lowest_pri)
                {
                    lowest_pri = prio;
                    lowest_index = i;
                }
            }
        }

        // if we found smth to merge, lets merge it
        if (lowest_index != -1)
        {
            symbols[lowest_index] = symbols[lowest_index] + symbols[lowest_index + 1];
            symbols.erase(symbols.begin() + lowest_index + 1);
        }
        else
        {
            // we need to break, no more merges
            break;
        }
    }

    // now we need to get the actual mapped indexes and return them
    std::vector<int> output;

    for (string symb : symbols)
    {
        output.push_back(sToT[symb]);
    }

    return output;
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
std::vector<std::string> Tokenizer::regex_split(const std::string &input, const std::regex &re)
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

std::unordered_map<int, char32_t> Tokenizer::byte2unicode()
{
    unordered_map<int, char32_t> output_map;

    // these are NORMAL ranges, ie normal characters, and should maintain the same value
    for (int i = 33; i <= 126; ++i)
        output_map[i] = i;
    for (int i = 161; i <= 172; ++i)
        output_map[i] = i;
    for (int i = 174; i <= 255; ++i)
        output_map[i] = i;

    // abnormal ranges
    int n = 0;
    for (int i = 0; i < 256; i++)
    {
        if (output_map.find(i) == output_map.end())
        {
            // we have a special case!
            output_map[i] = n + 256; // we do this to avoid ascii and move it into a safe range
            n++;
        }
    }
    return output_map;
}

// _chr = unichr if sys.version_info[0] == 2 else chr
// bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
// cs = bs[:]
// n = 0
// for b in range(2**8):
//     if b not in bs:
//         bs.append(b)
//         cs.append(2**8+n)
//         n += 1
// cs = [_chr(n) for n in cs]
// return dict(zip(bs, cs))