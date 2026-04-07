# TenANN v0.3.x

## v0.3.3-RELEASE
Download URL: [tenann-v0.3.3-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.3-RELEASE.tar.gz)

### Improvements

- Replaced the underlying BLAS library with OpenBLAS, improving IVFPQ build speed by 10x

## v0.3.2-RELEASE
Download URL: [tenann-v0.3.2-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.2-RELEASE.tar.gz)

### New Features

- Added `tenann/index/index_ivfpq_util.h`, providing the `GetIvfPqMinRows` method to get the minimum number of rows required to build an IvfPq index

## v0.3.1-RELEASE
Download URL: [tenann-v0.3.1-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.1-RELEASE.tar.gz)

### API Changes
- Removed `AnnSearcher::ResultOrder::Unordered`; only `Ascending` and `Descending` are now supported
- Renamed `AnnSearcher::ResultOrder::Asending` to `Ascending`; users need to migrate to the new naming

### New Features

- Added range search support for HNSW indexes
- Added `RangeSearchEvaluator` for range search testing and benchmarking
- Added brute-force range search algorithm for internal testing

## v0.3.0-RELEASE
Download URL: [tenann-v0.3.0-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RELEASE.tar.gz)

### New Features

- Added Python wrapper
- Added index_file_tool.py for inspecting index file information
- Added hybrid search support: filtering invalid data via IdFilter during search
- Added range search support for IVF-PQ indexes
- Added range search interface that returns only IDs without distances
- Added IdFilter support for range search interface

### Improvements
- Cleaned up redundant code in the custom IndexIvfPq implementation

### Bug Fix
- Fixed IVF-PQ index parameter `nlist` (previously misspelled as `nlists`)

## v0.3.0-RC3
Download URL: [tenann-v0.3.0-RC3.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC3.tar.gz)

### New Features
- Added IdFilter support for range search interface
- Added range search interface that returns only IDs without distances

### Improvements
- Cleaned up redundant code in the custom IndexIvfPq implementation

## v0.3.0-RC2
Download URL: [tenann-v0.3.0-RC2.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC2.tar.gz)

###  New Features
- Added range search support for IVF-PQ indexes

### Bug Fix
- Fixed IVF-PQ index parameter `nlist` (previously misspelled as `nlists`)

## v0.3.0-RC1
Download URL: [tenann-v0.3.0-RC1.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC1.tar.gz)

###  New Features
- Added hybrid search support: filtering invalid data via IdFilter during search
- Added Python wrapper
- Added index_file_tool.py for inspecting index file information
