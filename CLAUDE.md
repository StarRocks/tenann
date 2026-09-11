# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TenANN is a fast and flexible approximate nearest neighbor (ANN) search library built on top of FAISS. It provides C++ and Python APIs for building and searching vector indexes with multiple index types (HNSW, IVF-PQ) and distance metrics (L2, cosine, inner product).

## Build Commands

### Prerequisites

Set required environment variables:
```bash
export TENANN_GCC_HOME=/opt/gcc/usr  # GCC 5.3.1+ required
```

### Building the Library

```bash
# Standard build (Release mode)
sh build.sh

# Build with tests
sh build.sh --with-tests

# Build with examples
sh build.sh --with-examples

# Build with Python bindings
sh build.sh --with-python

# Debug build
BUILD_TYPE=Debug sh build.sh

# Asan build for memory debugging
BUILD_TYPE=Asan sh build.sh

# Clean build
sh build.sh --clean

# Parallel build with custom jobs
sh build.sh -j 8
```

Build artifacts are placed in `output/` directory.

There is one library for every CPU. FAISS is built with `FAISS_OPT_LEVEL=dd`, so
`libtenann.a` carries the AVX2, AVX-512 and (on ARM) NEON/SVE kernels together and
selects one by probing the CPU at load time. `tenann::SimdLevelName()` reports which
one got selected; neither the build flags nor the library name tell you.

To build against a fixed instruction set instead, for example to compare two levels:

```bash
FAISS_OPT_LEVEL_OVERRIDE=avx2 sh thirdparty/build-thirdparty.sh faiss
```

### Building Third-Party Dependencies

```bash
cd thirdparty
sh build-thirdparty.sh
```

This builds FAISS, fmt, GoogleTest, OpenBLAS, and pybind11 from source.

## Testing Commands

```bash
# Build with tests enabled
sh build.sh --with-tests

# Run all tests via CTest
cd build_Release
ctest

# Run specific test binary
./build_Release/test/tenann_test

# Run specific test case
./build_Release/test/tenann_test --gtest_filter=FaissHnswIndexBuilderTest.*

# Generate code coverage report
cd build_Release
make coverage
# View report at build_Release/coverage_html/index.html
```

Test logs are in `build_Release/Testing/Temporary/LastTest.log`.

## Architecture Overview

### Core Abstractions

**Index Pipeline**:
- `IndexMeta` (JSON config) → `IndexFactory` → `IndexBuilder` → Build → `IndexWriter` → Disk
- Disk → `IndexReader` → `IndexCache` → `Searcher` → Search Results

**Key Components**:

1. **Index Management** (`tenann/index/`)
   - `Index`: Type-safe wrapper around FAISS index pointers with custom deleters
   - `IndexMeta`: JSON-based configuration with typed parameter access
   - `IndexReader`/`IndexWriter`: Persistence layer with optional caching

2. **Index Building** (`tenann/builder/`)
   - Base: `IndexBuilder` (abstract) → `FaissIndexBuilder` (FAISS wrapper)
   - Implementations: `FaissHnswIndexBuilder`, `FaissIvfPqIndexBuilder`
   - Lifecycle: `Open()` → `Add()` → `Flush()` → `Close()`
   - Supports custom row IDs and runtime profiling

3. **Search** (`tenann/searcher/`)
   - Template-based CRTP pattern: `Searcher<T>` → `AnnSearcher`
   - Implementations: `FaissHnswAnnSearcher`, `FaissIvfPqAnnSearcher`
   - Operations: Top-K search, range search, filtered search (range/array/bitmap filters)

4. **Storage** (`tenann/store/`)
   - `IndexCache`: Global LRU cache with `IndexCacheHandle` for RAII
   - `IndexMeta`: Sections (common_params, index_params, search_params, extra_params)
   - Supports both JSON and MessagePack serialization

5. **Factories** (`tenann/factory/`)
   - `IndexFactory`: Creates readers/writers/builders based on `IndexMeta`
   - `AnnSearcherFactory`: Creates searchers based on index type

### Index Types

Supported via `IndexType` enum in `tenann/store/index_type.h`:
- `kFaissHnsw`: Hierarchical Navigable Small World (graph-based)
- `kFaissIvfFlat`: Inverted File with flat quantization
- `kFaissIvfPq`: Inverted File with product quantization
- `kFaissIvfPqOneInvertedList`: Special IVF-PQ variant for block caching

### Distance Metrics

Supported via `MetricType` enum:
- `kL2Distance`: Euclidean L2
- `kCosineSimilarity`: Cosine similarity
- `kInnerProduct`: Dot product
- `kCosineDistance`: 1 - cosine similarity

### Data Views (Zero-Copy Abstractions)

Located in `tenann/common/seq_view.h`:
- `PrimitiveSeqView`: Scalar sequences (e.g., IDs)
- `ArraySeqView`: Fixed-dimension arrays (e.g., vector matrix)
- `VlArraySeqView`: Variable-length arrays
- `SeqView`: Generic union wrapper

These enable efficient data passing without copying between user code and FAISS.

### FAISS Integration Pattern

TenANN wraps FAISS indexes while providing:
1. Type-safe C++ interface (no raw pointers exposed)
2. Automatic memory management (custom deleters)
3. Unified parameter system (JSON-based `IndexMeta`)
4. Consistent metric conversion
5. Built-in caching and persistence

The integration happens at:
- **Build time**: `InitIndex()` in builders creates FAISS index objects
- **I/O**: `IndexReader`/`IndexWriter` use FAISS serialization
- **Search time**: Direct FAISS `search()` / `range_search()` calls

### Python Bindings

Located in `python_bindings/bindings.cc`:
- Module: `tenann_py`
- Class: `TenANN` (unified builder/searcher interface)
- NumPy integration for input/output arrays
- Mirrors C++ builder/searcher lifecycle but with Python-friendly API

Build with: `sh build.sh --with-python`

### Code Organization

```
tenann/
├── bench/          # Benchmarking framework
├── builder/        # Index building (HNSW, IVF-PQ)
├── common/         # Core types (seq_view, macros, errors)
├── factory/        # Factory pattern implementations
├── index/          # Index abstraction and I/O
│   └── internal/   # Internal implementations (not installed)
├── searcher/       # Search APIs
│   └── internal/   # Internal utilities
├── store/          # Metadata, caching, persistence
└── util/           # Utilities (profiling, threading, brute-force)
```

Headers in `internal/` subdirectories are excluded from installation.

## Development Patterns

### Parameter Definitions

Parameters use macro-based compile-time definitions in `tenann/index/parameters.h`:

```cpp
struct FaissHnswIndexParams {
  DEFINE_OPTIONAL_PARAM(int, M, 16);              // Edges per node
  DEFINE_OPTIONAL_PARAM(int, efConstruction, 40);  // Build-time search width
};

struct FaissHnswSearchParams {
  DEFINE_OPTIONAL_PARAM(int, efSearch, 16);        // Query-time search width
};
```

Access via `IndexMeta`:
```cpp
int M = meta.GetIndexParam<int>("M");  // Required, throws if missing
int ef = meta.GetIndexParam<int>("efSearch", 16);  // Optional with default
```

### Error Handling

Custom exception types in `tenann/common/error.h`:
- `Error`: Recoverable errors (include file/line/backtrace)
- `FatalError`: Unrecoverable errors

Throw with: `T_THROW_EXCEPTION("message")`

### Memory Management Patterns

- **RAII**: `IndexCacheHandle` prevents premature cache eviction
- **Move semantics**: `Index` forbids copying, supports moving
- **Custom deleters**: FAISS index cleanup via lambda deleters

Use `T_FORBID_COPY_AND_ASSIGN(ClassName)` macro to disable copying.

### Extending with New Index Types

1. Add enum to `IndexType` in `tenann/store/index_type.h`
2. Create builder class extending `FaissIndexBuilder`
3. Define parameter struct in `tenann/index/parameters.h`
4. Create searcher class extending `AnnSearcher`
5. Create reader/writer if custom serialization needed
6. Update factory dispatch in `IndexFactory` and `AnnSearcherFactory`
7. Add tests mirroring structure in `test/`

## Common Development Workflows

### Adding a New Test

1. Create test file in `test/<component>/test_<feature>.cc`
2. Extend `FaissTestBase` for vector generation and recall evaluation
3. Add source file to `TENANN_TEST_SRC` in `test/CMakeLists.txt`
4. Build and run: `sh build.sh --with-tests && cd build_Release && ctest`

### Modifying Index Parameters

1. Update parameter struct in `tenann/index/parameters.h`
2. Update builder/searcher to use new parameter
3. Update tests with new parameter in `IndexMeta` JSON
4. Rebuild: `sh build.sh --clean --with-tests`

### Working with IndexMeta

IndexMeta uses JSON structure:
```json
{
  "common_params": {"dim": 128, "metric_type": 0},
  "index_params": {"M": 32, "efConstruction": 200},
  "search_params": {"efSearch": 64},
  "extra_params": {}
}
```

Load from JSON string or file, access via typed getters.

## Version Control

- Main development branch: `v0.4-dev`
- Current development branch: `v0.5-dev`
- Version format: `v<major>.<minor>.<patch>-RELEASE` (tags)

## License

Apache License 2.0
