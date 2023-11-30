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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <random>

#include "tenann/builder/index_builder.h"
#include "tenann/common/json.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/store/index_meta.h"

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

class TenANN {
 public:
  TenANN() = default;
  ~TenANN() = default;

  TenANN& CreateBuilderFromMeta(const std::string& input_meta) {
    tenann::IndexMeta meta(tenann::json::parse(input_meta));
    index_builder_ = tenann::IndexFactory::CreateBuilderFromMeta(meta);
    return *this;
  }

  /// Open an in-memory index builder.
  TenANN& Open() {
    if (!index_builder_) {
      throw std::runtime_error("IndexBuilder is nullptr. Call create_builder first.");
    }

    index_builder_->Open();
    return *this;
  }

  /// Open a disk-based index builder with the specified path for index file or directory.
  TenANN& Open(const std::string& index_save_path) {
    if (!index_builder_) {
      throw std::runtime_error("IndexBuilder is nullptr. Call create_builder first.");
    }
    index_builder_->Open(index_save_path);
    return *this;
  }

  // Clean resources and close this builder.
  TenANN& Close() {
    if (!index_builder_) {
      throw std::runtime_error("IndexBuilder is nullptr. Call create_builder first.");
    }
    index_builder_->Close();
    return *this;
  }

  TenANN& Add(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    if (!index_builder_) {
      throw std::runtime_error("IndexBuilder is nullptr. Call create_builder first.");
    }
    py::buffer_info buf_info = arr.request();
    if (buf_info.ndim != 2) {
      throw std::runtime_error("Unsupported array dimensionality, must be 2");
    }
    uint32_t nb = buf_info.shape[0];
    uint32_t d = buf_info.shape[1];
    float* data = static_cast<float*>(buf_info.ptr);
    auto base_view = tenann::ArraySeqView{.data = reinterpret_cast<uint8_t*>(data),
                                          .dim = d,
                                          .size = static_cast<uint32_t>(nb),
                                          .elem_type = tenann::PrimitiveType::kFloatType};
    index_builder_->Add({base_view});
    return *this;
  }

  TenANN& Flush() {
    if (!index_builder_) {
      throw std::runtime_error("IndexBuilder is nullptr. Call create_builder first.");
    }
    index_builder_->Flush();
    return *this;
  }

  TenANN& CreateSearcherFromMeta(const std::string& input_meta) {
    tenann::IndexMeta meta(tenann::json::parse(input_meta));
    ann_searcher_ = tenann::AnnSearcherFactory::CreateSearcherFromMeta(meta);
    return *this;
  }

  TenANN& ReadIndex(const std::string& index_save_path) {
    if (!ann_searcher_) {
      throw std::runtime_error("AnnSearcher is nullptr. Call CreateSearcherFromMeta first.");
    }
    ann_searcher_->ReadIndex(index_save_path);
    return *this;
  }

  py::array_t<int64_t> AnnSearch(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                                 int k) {
    if (!ann_searcher_) {
      throw std::runtime_error("AnnSearcher is nullptr. Call CreateSearcherFromMeta first.");
    }
    py::buffer_info buf_info = arr.request();
    if (buf_info.ndim != 1) {
      throw std::runtime_error("Unsupported array dimensionality, must be 1");
    }
    uint32_t d = buf_info.shape[0];
    float* data = static_cast<float*>(buf_info.ptr);
    auto query_view = tenann::PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(data),
                                               .size = d,
                                               .elem_type = tenann::PrimitiveType::kFloatType};
    std::vector<int64_t> result_ids(k);
    ann_searcher_->AnnSearch(query_view, k, result_ids.data());
    py::array_t<int64_t> result_ids_py(result_ids.size());
    auto result_data = result_ids_py.mutable_unchecked<1>();

    for (size_t i = 0; i < result_ids.size(); ++i) {
      result_data(i) = result_ids[i];
    }

    return result_ids_py;
  }

 protected:
  std::unique_ptr<tenann::IndexBuilder> index_builder_;
  std::unique_ptr<tenann::AnnSearcher> ann_searcher_;
};

PYBIND11_MODULE(tenann_py, m) {
  py::class_<TenANN>(m, "TenANN")
      .def(py::init<>())
      .def("create_builder", &TenANN::CreateBuilderFromMeta)
      .def("open", (TenANN & (TenANN::*)()) & TenANN::Open)
      .def("open", (TenANN & (TenANN::*)(const std::string&)) & TenANN::Open)
      .def("close", (TenANN & (TenANN::*)()) & TenANN::Close)
      .def("add", &TenANN::Add)
      .def("flush", &TenANN::Flush)
      .def("create_searcher", &TenANN::CreateSearcherFromMeta)
      .def("read_index", &TenANN::ReadIndex)
      .def("ann_search", &TenANN::AnnSearch);
}