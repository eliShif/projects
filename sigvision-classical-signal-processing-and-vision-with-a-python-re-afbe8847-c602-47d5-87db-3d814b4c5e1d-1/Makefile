CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off
BUILD := build
INC := -Icpp/include
LIB := cpp/src/sigcore.cpp
HDR := cpp/include/sigcore.hpp

.PHONY: build test-cpp clean

build: $(BUILD)/sigcore $(BUILD)/sigcore_tests

$(BUILD)/sigcore: cpp/src/main.cpp $(LIB) $(HDR)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ cpp/src/main.cpp $(LIB)

$(BUILD)/sigcore_tests: cpp/tests/test_sigcore.cpp $(LIB) $(HDR)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ cpp/tests/test_sigcore.cpp $(LIB)

test-cpp: $(BUILD)/sigcore_tests
	./$(BUILD)/sigcore_tests

clean:
	rm -rf $(BUILD)
