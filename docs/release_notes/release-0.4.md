# TenANN v0.4.x

## v0.4.0-RELEASE
Download URL: [tenann-v0.4.0-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.4.0-RELEASE.tar.gz)

### New Feature
- 支持了距离度量为Cosine Similarity的IVFPQ索引的RangeSearch

### Bug Fix
- 修复距离度量为Cosine Similarity时，索引大小估计错误的问题

## v0.4.0-RC1
Download URL: [tenann-v0.4.0-RC1.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.4.0-RC1.tar.gz)

### API Changes

- 创建Searcher和IndexBuilder时无需传入IndexReader和IndexWriter
- AnnSearcher的k由int类型改为int64类型
- 是否使用缓存现在由IndexMeta控制：

一个完整的AnnSearcher例子：
```c++
    IndexMeta meta;
    meta.SetMetaVersion(0);
    meta.SetIndexFamily(IndexFamily::kVectorIndex);
    meta.SetIndexType(IndexType::kFaissIvfPq);
    meta.common_params()["metric_type"] = MetricType::kL2Distance;
    meta.common_params()["dim"] = 768;
    meta.common_params()["is_vector_normed"] = false;
    // 使用BlockCache
    meta.index_reader_options()[IndexReaderOptions::cache_index_block_key] = true;
    // 或使用IndexFile Cache，注意两者不兼容，只能选其一
    // meta.index_reader_options()[IndexReaderOptions::cache_index_file_key] = true;

    auto index_path = "new.vi";
    auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
    ann_searcher->ReadIndex(index_path);
    ann_searcher->AnnSearch(...);
```

一个完整的IndexBuilder例子：
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

- 新增IVFPQ的BlockCache支持
- 新增IVFPQ的CosineSimilarity支持
- 新增实验性质的InnerProduct支持（仅支持TopKSearcher，不支持RangeSearch）


