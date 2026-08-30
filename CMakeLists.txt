cmake_minimum_required(VERSION 3.20)

project(MyProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# core logic lives in a library so both the app and the tests can link it
add_library(jwllm
    src/tokenizer.cpp
    src/tensor.cpp
)
target_include_directories(jwllm PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(my_app
    src/main.cpp
)
target_link_libraries(my_app PRIVATE jwllm)

# ---- tests ----
enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

add_executable(jwllm_tests
    tests/test_tokenizer.cpp
    tests/test_tensor.cpp
)
target_link_libraries(jwllm_tests PRIVATE jwllm GTest::gtest_main)

include(GoogleTest)
# run tests from the repo root so relative paths like data/vocab.json resolve
gtest_discover_tests(jwllm_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
