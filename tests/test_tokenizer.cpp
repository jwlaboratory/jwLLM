#include <gtest/gtest.h>

#include "tokenizer.hpp"

namespace
{
    Tokenizer make_tokenizer()
    {
        return Tokenizer("data/vocab.json", "data/merges.txt");
    }
} // namespace

TEST(Tokenizer, LoadsVocab)
{
    Tokenizer tok = make_tokenizer();
    EXPECT_GT(tok.sToT.size(), 0u);
    EXPECT_EQ(tok.sToT.size(), tok.tToS.size());
}

TEST(Tokenizer, LoadsMergeList)
{
    Tokenizer tok = make_tokenizer();
    EXPECT_GT(tok.merge_priority.size(), 0u);
}

TEST(Tokenizer, MappingsAreInverses)
{
    Tokenizer tok = make_tokenizer();
    for (const auto &[s, id] : tok.sToT)
    {
        ASSERT_EQ(tok.tToS.at(id), s);
    }
}

TEST(Tokenizer, ThrowsOnMissingVocabFile)
{
    EXPECT_THROW(Tokenizer("data/does_not_exist.json", "data/merges.txt"), std::runtime_error);
}

TEST(Tokenizer, DecodeConcatenatesTokenStrings)
{
    Tokenizer tok = make_tokenizer();
    std::string expected = tok.tToS.at(15) + tok.tToS.at(14) + tok.tToS.at(13);
    EXPECT_EQ(tok.decode({15, 14, 13}), expected);
}

// remove the DISABLED_ prefix once encode() is implemented
TEST(Tokenizer, EncodeDecodeRoundTrip)
{
    Tokenizer tok = make_tokenizer();
    std::string input = "Hello world!!!";
    EXPECT_EQ(tok.decode(tok.encode(input)), input);
}

TEST(Tokenizer, EncodeTest)
{
    Tokenizer tok = make_tokenizer();
    std::string input = "Hello my name is shrey how are you doing ajasjdu ! uaksjljnm r .asodp12031; salj1 sk a19 python code ! ;!!!";
    std::vector<int> output = {15496, 616, 1438, 318, 427, 4364, 703, 389, 345, 1804, 257, 28121, 73, 646, 5145, 334, 4730, 20362, 73, 21533, 374, 764, 292, 375, 79, 1065, 43637, 26, 3664, 73, 16, 1341, 257, 1129, 21015, 2438, 5145, 2162, 10185};
    EXPECT_EQ(tok.encode(input), output);
}