BUILD_DIR ?= build
BUILD_TYPE ?= Release
PYTHON ?= python3

.PHONY: help build test bench python-test check clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_\/-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(BUILD_DIR)/CMakeCache.txt: CMakeLists.txt
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: $(BUILD_DIR)/CMakeCache.txt ## Build static + shared libkittyops, tests, and bench
	cmake --build $(BUILD_DIR) -j

test: build ## Run C unit tests
	ctest --test-dir $(BUILD_DIR) --output-on-failure

python-test: build ## Run Python binding tests
	KITTYOPS_LIB=$(CURDIR)/$(BUILD_DIR)/libkittyops.dylib PYTHONPATH=$(CURDIR)/python $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

bench: build ## Run 1080p diff / scroll / compression benchmarks
	./$(BUILD_DIR)/bench_kittyops

check: test python-test ## Pre-commit check (C + Python tests)

clean: ## Remove build artifacts
	rm -rf $(BUILD_DIR)
