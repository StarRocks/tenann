import json
import numpy as np
from enum import Enum
from datetime import datetime

import tenann_py

class IndexFamily(Enum):
    kVectorIndex = 0
    kTextIndex = 1

class IndexType(Enum):
    kFaissHnsw = 0
    kFaissIvfFlat = 1
    kFaissIvfPq = 2

class MetricType(Enum):
    kL2Distance = 0
    kCosineSimilarity = 1
    kInnerProduct = 2
    kCosineDistance = 3

current_time = datetime.now()
print("Current time:", current_time, "\n")

# init index meta
meta = {}

meta["meta_version"] = 0
meta["family"] = IndexFamily.kVectorIndex.value
meta["type"] = IndexType.kFaissHnsw.value

common_params = {}
common_params["dim"] = 128
common_params["is_vector_normed"] = False
common_params["metric_type"] = MetricType.kL2Distance.value

index_params = {}
index_params["efConstruction"] = 40
index_params["M"] = 32

search_params = {}
search_params["efSearch"] = 40

extra_params = {}
extra_params["comments"] = "my comments"

meta["common"] = common_params
meta["index"] = index_params
meta["search"] = search_params
meta["extra"] = extra_params

json_str = json.dumps(meta, indent=4)

# init data
d = 128                       # dimension
nb = 200                      # database size
nq = 10                       # nb of queries
np.random.seed(1234)          # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

dir(tenann_py)
ann = tenann_py.TenANN()
print("meta:", json_str)
ann.create_builder(json_str).open("/tmp/faiss_hnsw_index_py").add(xb).flush()
ann.create_searcher(json_str).read_index("/tmp/faiss_hnsw_index_py")
for i, query_vector in enumerate(xq):
    result_ids = ann.ann_search(query_vector, 10)
    print("Query {}: {}".format(i, result_ids))