# TenANN

Fast and flexible approximate nearest neighbor (ANN) search library built on top of FAISS.

TenANN provides C++ and Python APIs for building and searching vector indexes with multiple index types and distance metrics.

## Features

- **Multiple Index Types**: HNSW (graph-based), IVF-PQ (product quantization), IVF-SQ (scalar quantization)
- **Distance Metrics**: L2 distance, cosine similarity, inner product, cosine distance
- **Filtered Search**: Top-K search, range search, and filtered search (range/array/bitmap filters)
- **Zero-Copy Data Views**: Efficient data passing between user code and FAISS without copying
- **Python Bindings**: NumPy-integrated Python API mirroring the C++ interface
- **Built-in Caching**: File-level and block-level LRU index caching
- **AVX2 / SVE Support**: Optional SIMD acceleration for x86_64 and ARM64

## Prerequisites

- GCC 5.3.1+ required

Set the required environment variable:
```bash
export TENANN_GCC_HOME=/opt/gcc/usr
```

## Building

### Build Third-Party Dependencies

```bash
cd thirdparty
sh build-thirdparty.sh
```

This builds FAISS, fmt, GoogleTest, OpenBLAS, and pybind11 from source.

### Build the Library

```bash
# Standard build (Release mode)
sh build.sh

# Build with tests
sh build.sh --with-tests

# Build with examples
sh build.sh --with-examples

# Build with AVX2 support (produces both libtenann.a and libtenann_avx2.a)
sh build.sh --with-avx2

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

Build artifacts are placed in the `output/` directory.

## Testing

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
```

## Quick Start

### Building an Index (C++)

```cpp
#include "tenann/factory/index_factory.h"
#include "tenann/store/index_meta.h"

tenann::IndexMeta meta;
meta.SetMetaVersion(0);
meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
meta.SetIndexType(tenann::IndexType::kFaissHnsw);
meta.common_params()["dim"] = 128;
meta.common_params()["metric_type"] = tenann::MetricType::kL2Distance;
meta.index_params()["M"] = 32;
meta.index_params()["efConstruction"] = 200;
meta.search_params()["efSearch"] = 64;

auto builder = tenann::IndexFactory::CreateBuilderFromMeta(meta);
builder->Open("/path/to/index")
    .Add({base_view})
    .Flush()
    .Close();
```

### Searching an Index (C++)

```cpp
#include "tenann/factory/ann_searcher_factory.h"

auto searcher = tenann::AnnSearcherFactory::CreateSearcherFromMeta(meta);
searcher->ReadIndex("/path/to/index");

std::vector<int64_t> result_ids(k);
searcher->AnnSearch(query_view, k, result_ids.data());
```

## Architecture Overview

### Index Pipeline

```
IndexMeta (JSON config) -> IndexFactory -> IndexBuilder -> Build -> IndexWriter -> Disk
Disk -> IndexReader -> IndexCache -> Searcher -> Search Results
```

### Supported Index Types

| Index Type | Description |
|---|---|
| HNSW | Hierarchical Navigable Small World (graph-based) |
| IVF-PQ | Inverted File with Product Quantization |
| IVF-SQ | Inverted File with 4-bit or 8-bit Scalar Quantization |

### Distance Metric Support

|        | L2 Distance | Cosine Similarity | Cosine Distance | Inner Product |
|--------|-------------|-------------------|-----------------|---------------|
| HNSW   | Yes | Yes | - | - |
| IVF-PQ | Yes | Yes | - | Experimental |
| IVF-SQ | Yes | Yes | - | Experimental |

### Query Type Support

|        | ANN Search | ANN Search with Filter | Range Search | Range Search with Filter |
|--------|------------|------------------------|--------------|--------------------------|
| HNSW   | Yes | Yes | Yes | Yes |
| IVF-PQ | Yes | Yes | Yes | Yes |
| IVF-SQ | Yes | Yes | Yes | Yes |

### Code Organization

```
tenann/
├── bench/          # Benchmarking framework
├── builder/        # Index building (HNSW, IVF-PQ, IVF-SQ)
├── common/         # Core types (seq_view, macros, errors)
├── factory/        # Factory pattern implementations
├── index/          # Index abstraction and I/O
│   └── internal/   # Internal implementations (not installed)
├── searcher/       # Search APIs
│   └── internal/   # Internal utilities
├── store/          # Metadata, caching, persistence
└── util/           # Utilities (profiling, threading, brute-force)
```

## License

Apache License 2.0
