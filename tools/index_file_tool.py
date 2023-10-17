# -*- coding: utf-8 -*-

import sys
import faiss

# 检查 Python 版本是否大于 3.9
if sys.version_info < (3, 9):
    print("This script requires Python 3.9 or above.")
    sys.exit(1)

def uint32_to_fourcc(value):
    result = bytearray()
    result.append(value & 0xFF)
    result.append((value >> 8) & 0xFF)
    result.append((value >> 16) & 0xFF)
    result.append((value >> 24) & 0xFF)
    return result.decode('utf-8')

def read_uint32_from_file(file_path):
    with open(file_path, 'rb') as file:
        data = file.read(4)
        if len(data) != 4:
            raise ValueError("文件中的字节数不足四个")
        value = int.from_bytes(data, byteorder='little')
        return value

def show_index_header(index):
    print("索引类型 (index):", type(index))
    print("维度 (d):", index.d)
    print("总向量数量 (ntotal):", index.ntotal)
    print("是否已训练 (is_trained):", index.is_trained)
    print("度量类型 (metric_type):", index.metric_type)

def show_ivfpq_index(index):
    print("----------IVFPQ 索引信息-----------------")
    print("MAGIC:", magic)
    show_index_header(index)
    print("是否进行残差编码 (by_residual):", index.by_residual)
    print("量化后的向量编码字节大小 (code_size):", index.code_size)
    print("是否拥有自己的内存 (own_fields):", index.own_fields)
    print("是否拥有倒排链表 (own_invlists):", index.own_invlists)
    print("------------默认查询参数------------------")
    print("nprobe:", index.nprobe)
    print("max_codes:", index.max_codes)
    print("scan_table_threshold:", index.scan_table_threshold)
    print("polysemous_ht:", index.polysemous_ht)
    print("------------量化器信息------------------")
    print("量化器类型 (quantizer):", type(index.pq))
    print("原始向量维度 (d):", index.pq.d)
    print("每个聚类中心的子向量数量 (M):", index.pq.M)
    print("每个子向量的位数 (nbits):", index.pq.nbits)
    print("dsub:", index.pq.dsub)
    print("ksub:", index.pq.ksub)
    print("------------倒排链表信息------------------")
    print("倒排链表类型 (invlists):", type(index.invlists))
    print("聚类中心数量 (nlist):", index.invlists.nlist)
    print("向量编码字节大小 (code_size):", index.invlists.code_size)

def show_hnsw_index(index):
    print("----------HNSW 索引信息-----------------")
    print("MAGIC:", magic)
    show_index_header(index)
    print("----------HNSW 索引参数-----------------")
    print("HNSW实例类型 (hnsw):", type(index.hnsw))
    print("entry_point:", index.hnsw.entry_point)
    print("max_level:", index.hnsw.max_level)
    print("efConstruction:", index.hnsw.efConstruction)
    print("upper_beam:", index.hnsw.upper_beam)
    print("------------默认查询参数------------------")
    print("efSearch:", index.hnsw.efSearch)
    print("check_relative_distance:", index.hnsw.check_relative_distance)
    print("----------Index 结点参数-----------------")
    print("结点索引类型 (index):", type(index.storage))
    show_index_header(index.storage)
    if(isinstance(index.storage, faiss.IndexFlat)):
        print("索引数量 (codes.size):", index.storage.code_size)

def show_idmap_index(index):
    print("----------IDMap 索引信息-----------------")
    print("MAGIC:", magic)
    show_index_header(index)
    print("IDMap:", type(index.id_map))
    print("IDMap Size:", index.id_map.size())

def show_index(file_dir):
    try:
        index = faiss.read_index(file_dir)
        print("Index successfully loaded:")
        print(index)
        
        if isinstance(index, faiss.IndexIVFPQ):
            print("Successfully parsed as IVFPQ index.")
            show_ivfpq_index(index)
        elif isinstance(index, faiss.IndexHNSW):
            print("Successfully parsed as HNSW index.")
            show_hnsw_index(index)
        elif isinstance(index, faiss.IndexIDMap):
            print("Successfully parsed as IDMap index.")
            show_idmap_index(index)
        else:
            print("Unknown index type. MAGIC:", magic)

    except faiss.Exception as e:
        print("Error occurred while loading index:")
        print(e)

def search_index(file_dir, query_vector):
    try:
        index = faiss.read_index(file_dir)
        D, I = index.search(query_vector, 1)
        print("查询结果:", I)
    except faiss.Exception as e:
        print("加载索引文件时出错:", str(e))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3.9 index_file_tool.py show file_dir")
        print("       python3.9 index_file_tool.py search file_dir query_vector")
        sys.exit(1)

    command = sys.argv[1]
    file_dir = sys.argv[2]

    global magic
    magic = uint32_to_fourcc(read_uint32_from_file(file_dir))

    if command == "show":
        show_index(file_dir)
    elif command == "search":
        if len(sys.argv) < 5:
            print("Usage: python3.9 index_file_tool.py search file_dir query_vector")
            sys.exit(1)
        query_vector = np.array(sys.argv[3:], dtype=np.float32)
        search_index(file_dir, query_vector)
    else:
        print("Invalid command. Please use 'show' or 'search' command.")