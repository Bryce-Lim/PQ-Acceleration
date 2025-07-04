# Makefile for AMX Inner Product with Arrow/Parquet support and Threading
CXX = g++
CXXFLAGS = -flax-vector-conversions -fopenmp -std=c++17 -O2 -march=native -fno-strict-aliasing -mavx512bf16 -pthread
LIBS = -larrow -lparquet -pthread

# Find Arrow/Parquet include directories
ARROW_INCLUDE = $(shell pkg-config --cflags arrow)
PARQUET_INCLUDE = $(shell pkg-config --cflags parquet)
ARROW_LIBS = $(shell pkg-config --libs arrow)
PARQUET_LIBS = $(shell pkg-config --libs parquet)

# If pkg-config doesn't work, try these common paths
ifeq ($(ARROW_INCLUDE),)
    ARROW_INCLUDE = -I/usr/include/arrow -I/usr/local/include/arrow
endif
ifeq ($(PARQUET_INCLUDE),)
    PARQUET_INCLUDE = -I/usr/include/parquet -I/usr/local/include/parquet
endif
ifeq ($(ARROW_LIBS),)
    ARROW_LIBS = -larrow
endif
ifeq ($(PARQUET_LIBS),)
    PARQUET_LIBS = -lparquet
endif

INCLUDES = $(ARROW_INCLUDE) $(PARQUET_INCLUDE)
ALL_LIBS = $(ARROW_LIBS) $(PARQUET_LIBS) -pthread

# Object files
OBJECTS = large_testing.o AMXInnerProduct.o ScalarInnerProduct.o BatchInnerProductCalculator.o

# Targets
all: large_testing

AMXInnerProduct.o: AMXInnerProduct.cpp AMXInnerProduct.h
	$(CXX) $(CXXFLAGS) -c AMXInnerProduct.cpp -o AMXInnerProduct.o

ScalarInnerProduct.o: ScalarInnerProduct.cpp ScalarInnerProduct.h
	$(CXX) $(CXXFLAGS) -c ScalarInnerProduct.cpp -o ScalarInnerProduct.o

BatchInnerProductCalculator.o: BatchInnerProductCalculator.cpp BatchInnerProductCalculator.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c BatchInnerProductCalculator.cpp -o BatchInnerProductCalculator.o

large_testing.o: large_testing.cpp AMXInnerProduct.h ScalarInnerProduct.h BatchInnerProductCalculator.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c large_testing.cpp -o large_testing.o

large_testing: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(ALL_LIBS) -o large_testing

# Debug version with additional debugging symbols
debug: CXXFLAGS += -g -DDEBUG -O0
debug: large_testing

# Clean up
clean:
	rm -f *.o large_testing

# Run with different thread counts for testing
test-threads: large_testing
	@echo "Running single-threaded test..."
	@./large_testing
	@echo "Testing completed."

# Memory usage analysis
valgrind-check: large_testing
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./large_testing

.PHONY: all clean debug test-threads valgrind-check
