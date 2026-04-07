# Indexes

- [x] HNSW
- [x] IVF-PQ

## Distance Metric Support

|     | l2_distance | cosine_similarity | cosine_distance | inner_product |
| --- | --- | --- | --- | --- |
| HNSW   | ✅ | ✅ | x | x |
| IVF-PQ | ✅ | x | x | x |

## Query Type Support

|     | ANN Search | ANN Search with Filter | Range Search | Range Search with Filter |
| --- | --- | --- | --- | --- |
| HNSW   | ✅ | ✅ | ✅  | ✅  |
| IVF-PQ | ✅ | ✅ | ✅  | ✅  |
