# TenANN v0.4.x

## v0.4.2-RELEASE
Download URL: [tenann-v0.4.2-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.4.2-RELEASE.tar.gz)

### Improvements
- Disabled parallel search for Faiss HNSW indexes

## v0.4.1-RELEASE
Download URL: [tenann-v0.4.1-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.4.1-RELEASE.tar.gz)

### Improvements
- Changed Faiss parallel normalization to sequential, resolving coredump caused by conflict between HNSW(cos) index and StarRocks thread pool
- Added concurrent index building and search stress testing tool: stress_tool.cc
- Added additional unit tests

### Bug Fix
- Fixed incorrect HNSW range search results when using Cosine Similarity distance metric
- Fixed memory lifetime issue with CacheHandle when using BlockCache, resolving out-of-bounds access risk
- Fixed multiple instances where Faiss exceptions were not caught in TenANN public interfaces

## v0.4.0-RELEASE
Download URL: [tenann-v0.4.0-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.4.0-RELEASE.tar.gz)

### New Feature
- Added RangeSearch support for IVFPQ indexes with Cosine Similarity distance metric

### Bug Fix
- Fixed incorrect index size estimation when using Cosine Similarity distance metric

## v0.4.0-RC1
Download URL: [tenann-v0.4.0-RC1.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.4.0-RC1.tar.gz)

### API Changes

- Creating Searcher and IndexBuilder no longer requires passing IndexReader and IndexWriter
- AnnSearcher's k parameter changed from int to int64 type
- Whether to use caching is now controlled by IndexMeta:

A complete AnnSearcher example:
```c++
    IndexMeta meta;
    meta.SetMetaVersion(0);
    meta.SetIndexFamily(IndexFamily::kVectorIndex);
    meta.SetIndexType(IndexType::kFaissIvfPq);
    meta.common_params()["metric_type"] = MetricType::kL2Distance;
    meta.common_params()["dim"] = 768;
    meta.common_params()["is_vector_normed"] = false;
    // Use BlockCache
    meta.index_reader_options()[IndexReaderOptions::cache_index_block_key] = true;
    // Or use IndexFile Cache (note: the two are incompatible, choose only one)
    // meta.index_reader_options()[IndexReaderOptions::cache_index_file_key] = true;

    auto index_path = "new.vi";
    auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
    ann_searcher->ReadIndex(index_path);
    ann_searcher->AnnSearch(...);
```

A complete IndexBuilder example:
```c++

  // set meta values
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissHnsw);
  meta.common_params()["dim"] = 128;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric;
  meta.index_params()["efConstruction"] = 500;
  meta.index_params()["M"] = 128;
  meta.search_params()["efSearch"] = 80;
  meta.extra_params()["comments"] = "my comments";
  meta.index_writer_options()["write_index_cache"] = true;

  auto index_builder1 = tenann::IndexFactory::CreateBuilderFromMeta(meta);
  index_builder1->Open(index_path)
      .Add(...)
      .Flush();
```

### New Features

- Added BlockCache support for IVFPQ
- Added CosineSimilarity support for IVFPQ
- Added experimental InnerProduct support (TopKSearch only, RangeSearch not supported)
