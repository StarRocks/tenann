# TenANN v0.3.x

## v0.3.0-RC3
Download URL: [tenann-v0.3.0-RC3.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC3.tar.gz)

## New Features
- 范围查询接口新增IdFilter支持
- 新增只返回ID，不返回距离的范围查询接口

## v0.3.0-RC2
Download URL: [tenann-v0.3.0-RC2.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC2.tar.gz)

##  New Features
- 新增IVF-PQ索引的范围查询支持

## Bug Fix
- 修复IVF-PQ索引的参数`nlist`（之前的`nlists`为拼写错误）

## v0.3.0-RC1
Download URL: [tenann-v0.3.0-RC1.tar.gz](https://mirrors.tencent.com/repository/generic/doris_thirdparty/tenann-v0.3.0-RC1.tar.gz)

##  New Features
- 新增混合查询支持：支持Search时通过IdFilter过滤无效数据
- 新增Python Wrapper
- 新增index_file_tool.py检查索引文件信息
