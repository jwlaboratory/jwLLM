CXX := c++
CXXFLAGS := -std=c++20 -Wall -Wextra -Iinclude -Ibuild/deps/json/include
CPPFLAGS :=
LDFLAGS :=
LDLIBS :=

JSON_VERSION := 3.12.0
GTEST_VERSION := 1.15.2
JSON_HEADER := build/deps/json/include/nlohmann/json.hpp
GTEST_DIR := build/deps/googletest
GTEST_STAMP := $(GTEST_DIR)/.downloaded

LIB_SOURCES := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
LIB_OBJECTS := $(patsubst src/%.cpp,build/obj/%.o,$(LIB_SOURCES))
APP_OBJECTS := build/obj/main.o

TEST_SOURCES := $(wildcard tests/test_*.cpp)
TEST_OBJECTS := $(patsubst tests/%.cpp,build/obj/%.o,$(TEST_SOURCES)) build/obj/gtest-all.o build/obj/gtest_main.o

.PHONY: all my_app test clean

all: my_app

my_app: build/bin/my_app

build/bin/my_app: $(LIB_OBJECTS) $(APP_OBJECTS)
	mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

test: build/bin/jwllm_tests
	./build/bin/jwllm_tests

build/bin/jwllm_tests: $(LIB_OBJECTS) $(TEST_OBJECTS)
	mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -pthread -o $@

$(JSON_HEADER):
	mkdir -p $(@D)
	curl -L --fail --silent --show-error \
		-o $@ \
		https://github.com/nlohmann/json/releases/download/v$(JSON_VERSION)/json.hpp

$(GTEST_STAMP):
	mkdir -p build/deps
	curl -L --fail --silent --show-error \
		-o build/deps/googletest.tar.gz \
		https://github.com/google/googletest/archive/refs/tags/v$(GTEST_VERSION).tar.gz
	tar -xzf build/deps/googletest.tar.gz -C build/deps
	mv build/deps/googletest-$(GTEST_VERSION) $(GTEST_DIR)
	touch $@

build/obj/%.o: src/%.cpp $(JSON_HEADER)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

build/obj/test_%.o: tests/test_%.cpp $(JSON_HEADER) $(GTEST_STAMP)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(GTEST_DIR)/googletest/include -c $< -o $@

build/obj/gtest-all.o: $(GTEST_DIR)/googletest/src/gtest-all.cc $(GTEST_STAMP)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(GTEST_DIR)/googletest -I$(GTEST_DIR)/googletest/include -c $< -o $@

build/obj/gtest_main.o: $(GTEST_DIR)/googletest/src/gtest_main.cc $(GTEST_STAMP)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(GTEST_DIR)/googletest/include -c $< -o $@

clean:
	rm -rf build