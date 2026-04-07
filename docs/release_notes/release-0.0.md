# TenANN v0.0.x

## v0.0.2-RELEASE
Download URL: [tenann-v0.0.2-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.0.2-RELEASE.tar.gz)

Release date: 2023.08.31

### New Features
- Added common index management interfaces:
  - Index
  - IndexMeta
  - IndexReader
  - IndexWriter
  - IndexBuilder
  - IndexFactory
  - Searcher
  - AnnSearcher & AnnSearcherFactory
- Support for generic index metadata management: IndexMeta
- Support for building, querying, reading, and writing Faiss HNSW indexes
- Support for building Faiss HNSW indexes via `ArraySeqView` or `VlArraySeqView` input types
- Support for building Faiss HNSW indexes via `Build` or `BuildWithPrimaryKey` methods
- Support for file-level LRU index caching
- All dependencies bundled into a single `libtenann-bundle.a` for distribution
- Added Error and FatalError exception classes to distinguish recoverable and unrecoverable errors
- Added T_LOG macro series for simple log output with error throwing based on log level
- Added initial unit tests

## v0.0.2-RC3
Download URL: [tenann-v0.0.2-RC3.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.0.2-RC3.tar.gz)

Release date: 2023.08.31

### New Features
- Added Docker image build scripts

### Build Changes
- Removed cmake find_package support, simplified CMakeLists
- CMakeLists no longer hardcodes faiss dependency paths; now specified via `TENANN_THIRDPARTY` environment variable
- lapack and blas are no longer installed via yum; now built from source
- All dependencies except glibc are now statically linked
- All static library dependencies are bundled with `libtenann.a` into `libtenann-bundle.a`
- Changed the release directory structure to:
```
output
├── include
│   └── tenann
└── lib
    └── libtenann-bundle.a
```

## v0.0.2-RC2

Download URL: [tenann-v0.0.2-RC2.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.0.2-RC2.tar.gz)

Release date: 2023.08.29

### API Changes
- `IndexBuilder.SetIndexWriter` now accepts `IndexWriterRef` (`shared_ptr<IndexWriter>`) instead of raw pointers
- `Searcher.SetIndexReader` now accepts `IndexReaderRef` (`shared_ptr<IndexReader>`) instead of raw pointers

### New Features
- IndexBuilder supports VlArraySeqView input type
- IndexBuilder supports BuildWithPrimaryKey

### Improvements
- All Faiss indexes share the same IndexReader/Writer

### Bug Fixes
- Fixed Faiss build failure introduced in v0.0.2-RC1
