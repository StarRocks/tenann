# TenANN v0.2.x

## v0.2.2-RLELASE
Download URL: [tenann-v0.2.2-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.2.2-RELEASE.tar.gz)

##  New Features
- Added API for setting the number of parallel threads used by faiss, example:
```c++
#include "tenann/util/threads.h"
tenann::OmpSetNumThreads(8);
```
  - Note: this parameter is global; all parallel computations within faiss share this setting, including batch insertion and batch search (batch search is not yet supported by TenANN, so it is currently unaffected)
  - Testing shows that setting this parameter greatly improves index building performance, achieving near-linear speedup

### Improvements
- Patched HNSW to faiss-1.7.4, slightly improving recall performance

### Bug Fix
- Fixed issue where Searcher query parameters were not taking effect
- Fixed issue in faiss where only the efSearch stored in the HNSW index took effect; dynamically passed efSearch at query time was ignored (temporarily fixed via patch, upstream PR to be submitted)

## v0.2.1-RLELASE
Download URL: [tenann-v0.2.1-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.2.1-RELEASE.tar.gz)

### New Features
- Added brute-force KNN search support (temporarily provided via `util/bruteforce_ann.h`); will be added as a new index type with corresponding Builder and Searcher in the future

### Improvements
- Optimized Searcher initialization logic; added `OnIndexLoaded` virtual method that subclasses must override to perform initialization and validation after the index is loaded into memory

### Bug Fix
- Fixed incorrect HNSW index building and query logic when using cosine similarity as the distance metric
- Temporarily disabled IVF-PQ support for cosine similarity

## v0.2.0-RLELASE
Download URL: [tenann-v0.2.0-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.2.0-RELEASE.tar.gz)

### New Features
- Added IVF-PQ index support
- Added cosine similarity support
- Added AVX2 support
- Added Python wrapper
- Added `/tenann/index/parameters.h` to document all index parameters; users can refer to this file for required parameters and default values
- Implemented Searcher parameter setting interface; users can use `Searcher.SetSearchParamItem()` and `SetSearchParams()` to configure parameters

### Improvements
- Refactored underlying code, simplifying index building and search implementation complexity
- Index cache size set to 1GB with 2 shards, preventing large indexes from causing cache overflow and frequent I/O
