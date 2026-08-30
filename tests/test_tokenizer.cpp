#include <gtest/gtest.h>

#include "tokenizer.hpp"

namespace
{
Tokenizer make_tokenizer()
{
    return Tokenizer(0, "data/vocab.json", "data/merges.txt");
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
    EXPECT_THROW(Tokenizer(0, "data/does_not_exist.json", "data/merges.txt"), std::runtime_error);
}

TEST(Tokenizer, DecodeConcatenatesTokenStrings)
{
    Tokenizer tok = make_tokenizer();
    std::string expected = tok.tToS.at(15) + tok.tToS.at(14) + tok.tToS.at(13);
    EXPECT_EQ(tok.decode({15, 14, 13}), expected);
}

// remove the DISABLED_ prefix once encode() is implemented
TEST(Tokenizer, DISABLED_EncodeDecodeRoundTrip)
{
    Tokenizer tok = make_tokenizer();
    std::string input = "Hello world";
    EXPECT_EQ(tok.decode(tok.encode(input)), input);
}
