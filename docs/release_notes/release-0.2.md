# TenANN v0.2.x

## v0.2.2-RLELASE
Download URL: [tenann-v0.2.2-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.2.2-RELEASE.tar.gz)

##  New Features
- 增加接口函数，用于设置faiss库用到的并行线程数，示例：
```c++
#include "tenann/util/threads.h"
tenann::OmpSetNumThreads(8);
```
  - 注意，此参数是全局变量，faiss内部全部的并行计算逻辑均共享此参数，包括批量写入与批量查询（批量查询TenANN暂未支持，所以暂时不受影响）
  - 经测试，设置此参数能够极大的改善索引构建性能，得到近似线性的加速比

### Improvements
- Patch HNSW到faiss-1.7.4，略微提升Recall表现

### Bug Fix
- 修复设置Searcher的查询参数不生效的问题
- 修复faiss库中的HNSW只有索引中的efSearch生效，查询时动态传入的efSearch不生效的问题（暂时通过patch实现，后面向社区提交PR)

## v0.2.1-RLELASE
Download URL: [tenann-v0.2.1-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.2.1-RELEASE.tar.gz)

### New Features
- 增加了暴力KNN查询的支持（暂时通过`util/bruteforce_ann.h`提供），后续会作为新的索引类型增加相应的Builder和Searcher

### Improvements
- 优化了Searcher初始化的逻辑，增加了OnIndexLoaded虚方法，子类需重载此方法，在索引加载到内存后做一些初始化和检查工作

### Bug Fix
- 修复了在距离度量为cosine similarity时，HNSW索引构建和查询逻辑错误的问题
- 暂时禁用了IVF-PQ对cosine similarity的支持

## v0.2.0-RLELASE
Download URL: [tenann-v0.2.0-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.2.0-RELEASE.tar.gz)

### New Features
- 新增IVF-PQ索引
- 新增cosine similarity支持
- 新增avx2支持
- 新增python wrapper
- 新增`/tenann/index/parameters.h`指示索引全部参数，用户可以查看此文件查看各索引需要的参数及默认值
- 实现了Seacher的参数设置接口，用户可以使用`Searcher.SetSearchParamItem()`和`SetSearchParams()`设置需要的参数

### Improvements
- 重构底层代码，简化了索引构建和搜索的实现复杂度
- 索引缓存的大小设为1GB，索引缓存的shards数设为2，避免大索引导致缓存超限，频繁IO的问题
