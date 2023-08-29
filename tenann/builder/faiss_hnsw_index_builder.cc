/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "tenann/builder/faiss_hnsw_index_builder.h"

#include <sstream>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_factory.h"
#include "faiss_hnsw_index_builder.h"
#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"

namespace tenann {

// @TODO(petri): refactor those dumplicate code
void FaissHnswIndexBuilder::BuildWithPrimaryKeyImpl(const std::vector<SeqView>& input_columns,
                                                    int primary_key_column_index) {
  T_CHECK(input_columns.size() == 2);
  int data_col_index;
  int64_t* ids;
  size_t num_ids;

  FetchIds(input_columns, primary_key_column_index, &data_col_index, &ids, &num_ids);

  auto input_seq_type = input_columns[data_col_index].seq_view_type;
  T_CHECK(input_seq_type == SeqViewType::kArraySeqView ||
          input_seq_type == SeqViewType::kArraySeqView);
  T_CHECK(input_columns[data_col_index].seq_view.array_seq_view.elem_type == PrimitiveType::kFloatType);

  // @TODO(petri): provide a unified macro to report parameter errors
  T_LOG_IF(ERROR, !index_meta_.common_params().contains("dim"))
      << "required common parameter `dim` is not set in index meta";
  T_LOG_IF(ERROR, !index_meta_.common_params().contains("metric_type"))
      << "required common parameter `metric_type` is not set in index meta";

  try {
    std::ostringstream oss;
    oss << "IDMap,HNSW";
    if (index_meta_.index_params().contains("M")) {
      oss << index_meta_.index_params()["M"].get<int>();
    }

    T_LOG_IF(ERROR,
             index_meta_.common_params()["metric_type"].get<int>() != MetricType::kL2Distance)
        << "only l2 distance is supported now";

    auto dim = index_meta_.common_params()["dim"].get<int>();
    auto index = std::unique_ptr<faiss::Index>(
        faiss::index_factory(dim, oss.str().c_str(), faiss::METRIC_L2));
    auto hnsw = static_cast<faiss::IndexHNSW*>(static_cast<faiss::IndexIDMap*>(index.get())->index);

    if (index_meta_.index_params().contains("efConstruction")) {
      hnsw->hnsw.efConstruction = index_meta_.index_params()["efConstruction"].get<int>();
    }

    if (index_meta_.search_params().contains("efSearch")) {
      hnsw->hnsw.efSearch = index_meta_.search_params()["efSearch"].get<int>();
    }

    if (input_seq_type == SeqViewType::kArraySeqView) {
      /* build hnsw index with ArraySeqView */
      auto array_seq_view = input_columns[data_col_index].seq_view.array_seq_view;
      T_DCHECK_EQ(array_seq_view.size, num_ids);

      index->add_with_ids(array_seq_view.size, reinterpret_cast<float*>(array_seq_view.data), ids);
    } else if (input_seq_type == SeqViewType::kVlArraySeqView) {
      /* build hnsw index with VlArraySeqView */
      auto vl_array_seq_view = input_columns[data_col_index].seq_view.vl_array_seq_view;
      T_DCHECK_EQ(vl_array_seq_view.size, num_ids);

      TypedVlArraySeqView<float> typed_seq_view(vl_array_seq_view);

      // check vector sizes
      size_t i = 0;
      for (auto slice : typed_seq_view) {
        if (slice.size != dim) {
          T_LOG(ERROR) << "invalid size for vector " << i << " : expected " << dim << " but got "
                       << slice.size;
        }
        i += 1;
      }

      // build index
      index->add_with_ids(vl_array_seq_view.size, reinterpret_cast<float*>(vl_array_seq_view.data),
                          ids);
    } else {
      T_LOG(ERROR) << "invalid SeqView type: " << input_seq_type;
    }

    index_ref_ =
        std::make_shared<Index>(index.release(),        //
                                IndexType::kFaissHnsw,  //
                                [](void* index) { delete static_cast<faiss::Index*>(index); });

  } catch (faiss::FaissException& e) {
    T_LOG(ERROR) << e.what();
  } catch (nlohmann::json::exception& e) {
    T_LOG(ERROR) << e.what();
  }
}

void FaissHnswIndexBuilder::BuildImpl(const std::vector<SeqView>& input_columns) {
  /* check input parameters*/
  T_CHECK(input_columns.size() == 1);
  auto input_seq_type = input_columns[0].seq_view_type;
  T_CHECK(input_seq_type == SeqViewType::kArraySeqView ||
          input_seq_type == SeqViewType::kArraySeqView);

  T_CHECK(input_columns[0].seq_view.array_seq_view.elem_type == PrimitiveType::kFloatType);

  // @TODO(petri): provide a unified macro to report parameter errors
  T_LOG_IF(ERROR, !index_meta_.common_params().contains("dim"))
      << "required common parameter `dim` is not set in index meta";
  T_LOG_IF(ERROR, !index_meta_.common_params().contains("metric_type"))
      << "required common parameter `metric_type` is not set in index meta";

  try {
    std::ostringstream oss;
    oss << "HNSW";
    if (index_meta_.index_params().contains("M")) {
      oss << index_meta_.index_params()["M"].get<int>();
    }

    T_LOG_IF(ERROR,
             index_meta_.common_params()["metric_type"].get<int>() != MetricType::kL2Distance)
        << "only l2 distance is supported now";

    auto dim = index_meta_.common_params()["dim"].get<int>();
    auto index = std::unique_ptr<faiss::IndexHNSW>(static_cast<faiss::IndexHNSW*>(
        faiss::index_factory(dim, oss.str().c_str(), faiss::METRIC_L2)));

    if (index_meta_.index_params().contains("efConstruction")) {
      index->hnsw.efConstruction = index_meta_.index_params()["efConstruction"].get<int>();
    }

    if (index_meta_.search_params().contains("efSearch")) {
      index->hnsw.efSearch = index_meta_.search_params()["efSearch"].get<int>();
    }

    if (input_seq_type == SeqViewType::kArraySeqView) {
      /* build hnsw index with ArraySeqView */
      auto array_seq_view = input_columns[0].seq_view.array_seq_view;
      index->add(array_seq_view.size, reinterpret_cast<float*>(array_seq_view.data));
    } else if (input_seq_type == SeqViewType::kVlArraySeqView) {
      /* build hnsw index with VlArraySeqView */
      auto vl_array_seq_view = input_columns[0].seq_view.vl_array_seq_view;
      TypedVlArraySeqView<float> typed_seq_view(vl_array_seq_view);

      // check vector sizes
      size_t i = 0;
      for (auto slice : typed_seq_view) {
        if (slice.size != dim) {
          T_LOG(ERROR) << "invalid size for vector " << i << " : expected " << dim << " but got "
                       << slice.size;
        }
        i += 1;
      }

      // build index
      index->add(vl_array_seq_view.size, reinterpret_cast<float*>(vl_array_seq_view.data));
    } else {
      T_LOG(ERROR) << "invalid SeqView type: " << input_seq_type;
    }

    index_ref_ =
        std::make_shared<Index>(index.release(),        //
                                IndexType::kFaissHnsw,  //
                                [](void* index) { delete static_cast<faiss::Index*>(index); });

  } catch (faiss::FaissException& e) {
    T_LOG(ERROR) << e.what();
  } catch (nlohmann::json::exception& e) {
    T_LOG(ERROR) << e.what();
  }
}

void FaissHnswIndexBuilder::FetchIds(const std::vector<SeqView>& input_columns,
                                     int primary_key_column_index, int* data_col_index,
                                     int64_t** ids, size_t* size) {
  T_DCHECK(primary_key_column_index == 0 || primary_key_column_index == 1);
  *data_col_index = primary_key_column_index == 0 ? 1 : 0;

  auto id_seq = input_columns[primary_key_column_index];
  T_CHECK(id_seq.seq_view_type == SeqViewType::kPrimitiveSeqView &&
          id_seq.seq_view.primitive_seq_view.elem_type == PrimitiveType::kInt64Type);

  *ids = reinterpret_cast<int64_t*>(id_seq.seq_view.primitive_seq_view.data);
  *size = id_seq.seq_view.primitive_seq_view.size;
}

}  // namespace tenann