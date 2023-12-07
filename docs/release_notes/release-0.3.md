# TenANN v0.3.x

## v0.3.3-RELEASE
Download URL: [tenann-v0.3.3-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.3-RELEASE.tar.gz)

### Improvements

- 替换底层BLAS库为OpenBLAS，IVFPQ构建速度提升10倍


## v0.3.2-RELEASE
Download URL: [tenann-v0.3.2-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.2-RELEASE.tar.gz)

### New Features

- 新增`tenan/index/index_ivfpq_util.h`，提供`GetIvfPqMinRows`方法，用于获取构建IvfPq索引所需的最小行数，

## v0.3.1-RELEASE
Download URL: [tenann-v0.3.1-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.1-RELEASE.tar.gz)

### API Changes
- 删除`AnnSearcher::ResultOrder::Unordered`，仅支持`Ascending`和`Descending`
- `AnnSearcher::ResultOrder::Asending`重命名为`Ascending`，用户需要迁移到新命名

### New Features

- HNSW索引新增范围查询支持
- 新增`RangeSearchEvaluator`，用于范围查询的测试及Benchmark
- 新增范围查询的暴力算法，用于内部测试

## v0.3.0-RELEASE
Download URL: [tenann-v0.3.0-RELEASE.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RELEASE.tar.gz)

### New Features

- 新增Python Wrapper
- 新增index_file_tool.py检查索引文件信息
- 新增混合查询支持：支持Search时通过IdFilter过滤无效数据
- 新增IVF-PQ索引的范围查询支持
- 新增只返回ID，不返回距离的范围查询接口
- 范围查询接口新增IdFilter支持

### Improvements
- 清理了自研IndexIvfPq中的部分冗余代码

### Bug Fix
- 修复IVF-PQ索引的参数`nlist`（之前的`nlists`为拼写错误）

## v0.3.0-RC3
Download URL: [tenann-v0.3.0-RC3.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC3.tar.gz)

### New Features
- 范围查询接口新增IdFilter支持
- 新增只返回ID，不返回距离的范围查询接口

### Improvements
- 清理了自研IndexIvfPq中的部分冗余代码

## v0.3.0-RC2
Download URL: [tenann-v0.3.0-RC2.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC2.tar.gz)

###  New Features
- 新增IVF-PQ索引的范围查询支持

### Bug Fix
- 修复IVF-PQ索引的参数`nlist`（之前的`nlists`为拼写错误）

## v0.3.0-RC1
Download URL: [tenann-v0.3.0-RC1.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC1.tar.gz)

###  New Features
- 新增混合查询支持：支持Search时通过IdFilter过滤无效数据
- 新增Python Wrapper
- 新增index_file_tool.py检查索引文件信息
