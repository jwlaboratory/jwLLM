add_test([=[Tokenizer.LoadsVocab]=]  /Users/shreybirmiwal/projects/jwlabs/jwLLM/build/jwllm_tests [==[--gtest_filter=Tokenizer.LoadsVocab]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Tokenizer.LoadsVocab]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM/tests/test_tokenizer.cpp:13]==]
    WORKING_DIRECTORY [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Tokenizer.LoadsMergeList]=]  /Users/shreybirmiwal/projects/jwlabs/jwLLM/build/jwllm_tests [==[--gtest_filter=Tokenizer.LoadsMergeList]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Tokenizer.LoadsMergeList]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM/tests/test_tokenizer.cpp:20]==]
    WORKING_DIRECTORY [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Tokenizer.MappingsAreInverses]=]  /Users/shreybirmiwal/projects/jwlabs/jwLLM/build/jwllm_tests [==[--gtest_filter=Tokenizer.MappingsAreInverses]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Tokenizer.MappingsAreInverses]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM/tests/test_tokenizer.cpp:26]==]
    WORKING_DIRECTORY [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Tokenizer.ThrowsOnMissingVocabFile]=]  /Users/shreybirmiwal/projects/jwlabs/jwLLM/build/jwllm_tests [==[--gtest_filter=Tokenizer.ThrowsOnMissingVocabFile]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Tokenizer.ThrowsOnMissingVocabFile]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM/tests/test_tokenizer.cpp:35]==]
    WORKING_DIRECTORY [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Tokenizer.DecodeConcatenatesTokenStrings]=]  /Users/shreybirmiwal/projects/jwlabs/jwLLM/build/jwllm_tests [==[--gtest_filter=Tokenizer.DecodeConcatenatesTokenStrings]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Tokenizer.DecodeConcatenatesTokenStrings]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM/tests/test_tokenizer.cpp:40]==]
    WORKING_DIRECTORY [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Tokenizer.EncodeDecodeRoundTrip]=]  /Users/shreybirmiwal/projects/jwlabs/jwLLM/build/jwllm_tests [==[--gtest_filter=Tokenizer.DISABLED_EncodeDecodeRoundTrip]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Tokenizer.EncodeDecodeRoundTrip]=]
  PROPERTIES
    DISABLED YES
    DEF_SOURCE_LINE [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM/tests/test_tokenizer.cpp:48]==]
    WORKING_DIRECTORY [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Tensor.StoresDimsAndData]=]  /Users/shreybirmiwal/projects/jwlabs/jwLLM/build/jwllm_tests [==[--gtest_filter=Tensor.StoresDimsAndData]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Tensor.StoresDimsAndData]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM/tests/test_tensor.cpp:5]==]
    WORKING_DIRECTORY [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Tensor.ThrowsWhenDataSizeMismatchesDims]=]  /Users/shreybirmiwal/projects/jwlabs/jwLLM/build/jwllm_tests [==[--gtest_filter=Tensor.ThrowsWhenDataSizeMismatchesDims]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Tensor.ThrowsWhenDataSizeMismatchesDims]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM/tests/test_tensor.cpp:12]==]
    WORKING_DIRECTORY [==[/Users/shreybirmiwal/projects/jwlabs/jwLLM]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(jwllm_tests_TESTS [==[Tokenizer.LoadsVocab]==] [==[Tokenizer.LoadsMergeList]==] [==[Tokenizer.MappingsAreInverses]==] [==[Tokenizer.ThrowsOnMissingVocabFile]==] [==[Tokenizer.DecodeConcatenatesTokenStrings]==] [==[Tokenizer.EncodeDecodeRoundTrip]==] [==[Tensor.StoresDimsAndData]==] [==[Tensor.ThrowsWhenDataSizeMismatchesDims]==])
