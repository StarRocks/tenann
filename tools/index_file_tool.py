# -*- coding: utf-8 -*-

import sys
import faiss

# Check if Python version is 3.9 or above
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
            raise ValueError("File contains fewer than four bytes")
        value = int.from_bytes(data, byteorder='little')
        return value

def show_index_header(index):
    print("Index type (index):", type(index))
    print("Dimension (d):", index.d)
    print("Total vectors (ntotal):", index.ntotal)
    print("Is trained (is_trained):", index.is_trained)
    print("Metric type (metric_type):", index.metric_type)

def print_ClusteringParameters(cp):
    if isinstance(cp, faiss.ClusteringParameters):
        print("------------ ClusteringParameters Info ------------------")
        print("ClusteringParameters type:", type(cp))
        print("Number of iterations (niter):", cp.niter)
        print("Number of redos (nredo):", cp.nredo)
        print("Verbose output (verbose):", cp.verbose)
        print("Spherical clustering (spherical):", cp.spherical)
        print("Integer centroids (int_centroids):", cp.int_centroids)
        print("Update index (update_index):", cp.update_index)
        print("Min points per centroid (min_points_per_centroid):", cp.min_points_per_centroid)
        print("Frozen centroids (frozen_centroids):", cp.frozen_centroids)
        print("Max points per centroid (max_points_per_centroid):", cp.max_points_per_centroid)
        print("Random seed (seed):", cp.seed)
        print("Decode block size (decode_block_size):", cp.decode_block_size)

def show_ivfpq_index(index):
    print("---------- IVFPQ Index Info -----------------")
    print("MAGIC: index type identifier", magic)
    show_index_header(index)
    print("Encode by residual (by_residual):", index.by_residual)
    print("Encoded vector byte size (code_size):", index.code_size)
    print("Owns fields (own_fields):", index.own_fields)
    print("Owns inverted lists (own_invlists):", index.own_invlists)
    print("------------ IndexIVFPQ.quantizer Coarse Quantizer Info ------------------")
    print("Coarse quantizer type (quantizer):", type(index.quantizer))
    print("Quantizer trains alone (quantizer_trains_alone):", index.quantizer_trains_alone)
    print("Clustering parameters (ClusteringParameters):", type(index.cp))
    print_ClusteringParameters(index.cp)
    print("Clustering index type (clustering_index):", type(index.clustering_index))
    print("------------ ProductQuantizer Fine Quantizer Info ------------------")
    print("Product quantizer type (ProductQuantizer):", type(index.pq))
    print("Original vector dimension (d):", index.pq.d)
    print("Number of sub-vectors per centroid (M):", index.pq.M)
    print("Bits per sub-vector (nbits):", index.pq.nbits)
    print("Sub-vector dimension (dsub):", index.pq.dsub)
    print("Number of centroids per sub-vector (ksub):", index.pq.ksub)
    print("Verbose output (verbose):", index.pq.verbose)
    print("Training type (train_type):", index.pq.train_type)
    print_ClusteringParameters(index.pq.cp)
    print("------------ Default Search Parameters ------------------")
    print("Number of clusters to visit during search (nprobe):", index.nprobe)
    print("Max codes to visit during search (max_codes):", index.max_codes)
    print("Polysemous hash parameter (scan_table_threshold):", index.scan_table_threshold)
    print("Polysemous hash parameter (polysemous_ht):", index.polysemous_ht)
    print("------------ Inverted Lists Info ------------------")
    print("Inverted lists type (invlists):", type(index.invlists))
    print("Number of clusters (nlist):", index.invlists.nlist)
    print("Vector encoding byte size (code_size):", index.invlists.code_size)

def show_hnsw_index(index):
    print("---------- HNSW Index Info -----------------")
    print("MAGIC: index type identifier", magic)
    show_index_header(index)
    print("---------- HNSW Index Parameters -----------------")
    print("HNSW instance type (hnsw):", type(index.hnsw))
    print("Entry point (entry_point):", index.hnsw.entry_point)
    print("Max level (max_level):", index.hnsw.max_level)
    print("Graph construction expansion factor (efConstruction):", index.hnsw.efConstruction)
    print("Upper layer search width (upper_beam):", index.hnsw.upper_beam)
    print("------------ Default Search Parameters ------------------")
    print("Search expansion factor (efSearch):", index.hnsw.efSearch)
    print("Check relative distance (check_relative_distance):", index.hnsw.check_relative_distance)
    print("---------- Index Storage Parameters -----------------")
    print("Storage index type (index):", type(index.storage))
    show_index_header(index.storage)
    if(isinstance(index.storage, faiss.IndexFlat)):
        print("Code size (codes.size):", index.storage.code_size)

def show_idmap_index(index):
    print("---------- IDMap Index Info -----------------")
    print("MAGIC:", magic)
    show_index_header(index)
    print("IDMap type:", type(index.id_map))
    print("IDMap size:", index.id_map.size())
    print_index(index.index)
    show_index_header(index.index)

def show_pretransform_index(index):
    print("---------- IndexPreTransform Index Info -----------------")
    print("MAGIC:", magic)
    show_index_header(index)
    print("VectorTransform type:", type(index.chain))
    print("VectorTransform size:", index.chain.size())
    print("Index type:", type(index.index))
    print_index(index.index)
    show_index_header(index.index)

def print_index(index):
    if isinstance(index, faiss.IndexIVFPQ):
        print("Successfully parsed as IVFPQ index.")
        print("IVFPQ is an inverted file index with product quantization, used for large-scale vector search")
        show_ivfpq_index(index)
    elif isinstance(index, faiss.IndexHNSW):
        print("Successfully parsed as HNSW index.")
        print("HNSW is a hierarchical navigable small world graph index, used for approximate nearest neighbor search")
        show_hnsw_index(index)
    elif isinstance(index, faiss.IndexIDMap):
        print("Successfully parsed as IDMap index.")
        print("IDMap index maps vector IDs to contiguous integers")
        show_idmap_index(index)
    elif isinstance(index, faiss.IndexPreTransform):
        print("Successfully parsed as IndexPreTransform index.")
        print("IndexPreTransform index transforms vectors before querying and adding")
        show_pretransform_index(index)
    else:
        print("Unknown index type. MAGIC:", magic)

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
        print("Search results:", I)
    except faiss.Exception as e:
        print("Error loading index file:", str(e))

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
