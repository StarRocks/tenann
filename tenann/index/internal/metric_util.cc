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

#include "tenann/index/internal/metric_util.h"

#include "tenann/common/logging.h"

namespace tenann {

CosineBackend ParseCosineBackend(const std::string& value) {
  if (value == "l2") {
    return CosineBackend::kL2;
  }
  if (value == "inner_product") {
    return CosineBackend::kInnerProduct;
  }
  T_LOG(ERROR) << "unsupported cosine backend: " << value;
  return CosineBackend::kL2;
}

MetricType ResolveBuildMetric(MetricType logical_metric, CosineBackend cosine_backend) {
  switch (logical_metric) {
    case MetricType::kL2Distance:
      return MetricType::kL2Distance;
    case MetricType::kInnerProduct:
      return MetricType::kInnerProduct;
    case MetricType::kCosineSimilarity:
      return cosine_backend == CosineBackend::kInnerProduct ? MetricType::kInnerProduct
                                                            : MetricType::kL2Distance;
    default:
      T_LOG(ERROR) << "unsupported logical metric: " << static_cast<int>(logical_metric);
      return MetricType::kL2Distance;
  }
}

MetricType FromFaissMetric(faiss::MetricType metric) {
  switch (metric) {
    case faiss::METRIC_L2:
      return MetricType::kL2Distance;
    case faiss::METRIC_INNER_PRODUCT:
      return MetricType::kInnerProduct;
    default:
      T_LOG(ERROR) << "unsupported faiss metric: " << static_cast<int>(metric);
      return MetricType::kL2Distance;
  }
}

faiss::MetricType ToFaissMetric(MetricType metric) {
  switch (metric) {
    case MetricType::kL2Distance:
      return faiss::METRIC_L2;
    case MetricType::kInnerProduct:
      return faiss::METRIC_INNER_PRODUCT;
    default:
      T_LOG(ERROR) << "unsupported physical metric: " << static_cast<int>(metric);
      return faiss::METRIC_L2;
  }
}

void ValidateLoadedMetric(MetricType logical_metric, MetricType physical_metric) {
  const bool valid =
      (logical_metric == MetricType::kL2Distance && physical_metric == MetricType::kL2Distance) ||
      (logical_metric == MetricType::kInnerProduct &&
       physical_metric == MetricType::kInnerProduct) ||
      (logical_metric == MetricType::kCosineSimilarity &&
       (physical_metric == MetricType::kL2Distance ||
        physical_metric == MetricType::kInnerProduct));
  T_CHECK(valid) << "logical metric " << static_cast<int>(logical_metric)
                 << " is incompatible with loaded physical metric "
                 << static_cast<int>(physical_metric);
}

bool NeedsL2ToCosine(MetricType logical_metric, MetricType physical_metric) {
  return logical_metric == MetricType::kCosineSimilarity &&
         physical_metric == MetricType::kL2Distance;
}

}  // namespace tenann
