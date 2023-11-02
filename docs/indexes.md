# Indexes

- [x] HNSW
- [x] IVF-PQ 

## 距离度量支持情况

| | l2_distance | cosine_similarity | cosine_distance | inner_product |
| --- | --- | --- | --- | --- |
| HNSW   | √ | √ | x | x |
| IVF-PQ | √ | x | x | x |

## 查询类型支持情况

| | ANN Search | ANN Search with Filter | Range Search | Range Search with Filter |
| --- | --- | --- | --- | --- |
| HNSW   | √ | √ | x | x |
| IVF-PQ | √ | √ | √ | √ |