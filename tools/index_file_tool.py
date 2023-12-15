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

def print_ClusteringParameters(cp):
    if isinstance(cp, faiss.ClusteringParameters):
        print("------------ ClusteringParameters 信息------------------")
        print("ClusteringParameters 类型:", type(cp))
        print("迭代次数 (niter):", cp.niter)
        print("重复次数 (nredo):", cp.nredo)
        print("是否输出详细信息 (verbose):", cp.verbose)
        print("是否使用球形聚类 (spherical):", cp.spherical)
        print("是否使用整数聚类中心 (int_centroids):", cp.int_centroids)
        print("是否更新索引 (update_index):", cp.update_index)
        print("每个聚类中心的最小点数 (min_points_per_centroid):", cp.min_points_per_centroid)
        print("是否冻结聚类中心 (frozen_centroids):", cp.frozen_centroids)
        print("每个聚类中心的最大点数 (max_points_per_centroid):", cp.max_points_per_centroid)
        print("随机种子 (seed):", cp.seed)
        print("解码块大小 (decode_block_size):", cp.decode_block_size)

def show_ivfpq_index(index):
    print("----------IVFPQ 索引信息-----------------")
    print("MAGIC: 索引类型标识", magic)
    show_index_header(index)
    print("是否进行残差编码 (by_residual):", index.by_residual)
    print("量化后的向量编码字节大小 (code_size):", index.code_size)
    print("是否拥有自己的内存 (own_fields):", index.own_fields)
    print("是否拥有倒排链表 (own_invlists):", index.own_invlists)
    print("------------ IndexIVFPQ.quantizer 粗量化器信息 ------------------")
    print("粗量化器类型 (quantizer):", type(index.quantizer))
    print("粗量化器是否独立训练 (quantizer_trains_alone):", index.quantizer_trains_alone)
    print("聚类参数 (ClusteringParameters):", type(index.cp))
    print_ClusteringParameters(index.cp)
    print("用于聚类的索引类型 (clustering_index):", type(index.clustering_index))
    print("------------ ProductQuantizer 精量化器信息------------------")
    print("精(乘积)量化器类型 (ProductQuantizer):", type(index.pq))
    print("原始向量维度 (d):", index.pq.d)
    print("每个聚类中心的子向量数量 (M):", index.pq.M)
    print("每个子向量的位数 (nbits):", index.pq.nbits)
    print("子向量的维度 (dsub):", index.pq.dsub)
    print("每个子向量的聚类中心数量 (ksub):", index.pq.ksub)
    print("是否输出详细信息 (verbose):", index.pq.verbose)
    print("训练类型 (train_type):", index.pq.train_type)
    print_ClusteringParameters(index.pq.cp)
    print("------------默认查询参数------------------")
    print("搜索过程中要访问的聚类中心数量 (nprobe):", index.nprobe)
    print("搜索过程中要访问的最大编码数量 (max_codes):", index.max_codes)
    print("多义性哈希的参数 (scan_table_threshold):", index.scan_table_threshold)
    print("多义性哈希的参数 (polysemous_ht):", index.polysemous_ht)
    print("------------倒排链表信息------------------")
    print("倒排链表类型 (invlists):", type(index.invlists))
    print("聚类中心数量 (nlist):", index.invlists.nlist)
    print("向量编码字节大小 (code_size):", index.invlists.code_size)

def show_hnsw_index(index):
    print("----------HNSW 索引信息-----------------")
    print("MAGIC: 索引类型标识", magic)
    show_index_header(index)
    print("----------HNSW 索引参数-----------------")
    print("HNSW实例类型 (hnsw):", type(index.hnsw))
    print("入口点 (entry_point):", index.hnsw.entry_point)
    print("最大层级 (max_level):", index.hnsw.max_level)
    print("构建图时的扩展因子 (efConstruction):", index.hnsw.efConstruction)
    print("上层节点的搜索宽度 (upper_beam):", index.hnsw.upper_beam)
    print("------------默认查询参数------------------")
    print("搜索时的扩展因子 (efSearch):", index.hnsw.efSearch)
    print("是否检查相对距离 (check_relative_distance):", index.hnsw.check_relative_distance)
    print("----------Index 结点参数-----------------")
    print("结点索引类型 (index):", type(index.storage))
    show_index_header(index.storage)
    if(isinstance(index.storage, faiss.IndexFlat)):
        print("索引数量 (codes.size):", index.storage.code_size)

def show_idmap_index(index):
    print("----------IDMap 索引信息-----------------")
    print("MAGIC:", magic)
    show_index_header(index)
    print("IDMap类型:", type(index.id_map))
    print("IDMap大小:", index.id_map.size())

def show_pretransform_index(index):
    print("---------- IndexPreTransform 索引信息-----------------")
    print("MAGIC:", magic)
    show_index_header(index)
    print("VectorTransform类型:", type(index.chain))
    print("VectorTransform大小:", index.chain.size())
    print("索引类型:", type(index.index))
    print_index(index.index)
    show_index_header(index.index)

def print_index(index):
    if isinstance(index, faiss.IndexIVFPQ):
        print("成功解析为IVFPQ索引。")
        print("IVFPQ索引是一种基于倒排文件的乘积量化索引,用于大规模向量搜索")
        show_ivfpq_index(index)
    elif isinstance(index, faiss.IndexHNSW):
        print("成功解析为HNSW索引。")
        print("HNSW索引是一种基于层次化的小世界图的索引,用于近似最近邻搜索")
        show_hnsw_index(index)
    elif isinstance(index, faiss.IndexIDMap):
        print("成功解析为IDMap索引。")
        print("IDMap索引用于将向量ID映射到连续的整数")
        show_idmap_index(index)
    elif isinstance(index, faiss.IndexPreTransform):
        print("成功解析为IndexPreTransform索引。")
        print("IndexPreTransform索引用于在查询和添加向量之前对向量进行转换")
        show_pretransform_index(index)
    else:
        print("未知的索引类型。MAGIC:", magic)

def show_index(file_dir):
    try:
        index = faiss.read_index(file_dir)
        print("Index successfully loaded:")
        print_index(index)

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