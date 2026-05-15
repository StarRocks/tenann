#include <filesystem>

#include "fmt/format.h"
#include "sstream"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/index/index_cache.h"
#include "tenann/store/index_meta.h"
#include "tenann/util/pretty_printer.h"
#include "tenann/util/random.h"
#include "tenann/util/runtime_profile.h"
#include "tenann/util/runtime_profile_macros.h"
#include "tenann/util/threads.h"

using namespace tenann;

static constexpr const int dim = 128;
static constexpr const int nb = 10000;
static constexpr const int nq = 1;
static constexpr const int verbose = VERBOSE_DEBUG;

bool FileExists() { return true; }

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
  if (numThreads <= 0) {
    numThreads = std::thread::hardware_concurrency();
  }

  if (numThreads == 1) {
    for (size_t id = start; id < end; id++) {
      fn(id, 0);
    }
  } else {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
      threads.push_back(std::thread([&, threadId] {
        while (true) {
          size_t id = current.fetch_add(1);

          if (id >= end) {
            break;
          }

          try {
            fn(id, threadId);
          } catch (tenann::Error& e) {
            std::cerr << e.what();
          } catch (...) {
            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
            lastException = std::current_exception();
            /*
             * This will work even when current is the largest value that
             * size_t can fit, because fetch_add returns the previous value
             * before the increment (what will result in overflow
             * and produce 0 instead of current + 1).
             */
            current = end;
            break;
          }
        }
      }));
    }
    for (auto& thread : threads) {
      thread.join();
    }
    if (lastException) {
      std::rethrow_exception(lastException);
    }
  }
}

IndexMeta PrepareIvfPqMeta(MetricType metric_type, int dim, int nlist, int M, int nbits,
                           bool use_block_cache) {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  meta.common_params()["dim"] = dim;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric_type;
  meta.index_params()["nlist"] = nlist;
  meta.index_params()["M"] = M;
  meta.index_params()["nbtis"] = nbits;

  if (use_block_cache) {
    meta.index_writer_options()["write_index_cache"] = false;
    meta.index_reader_options()["cache_index_block"] = true;
    meta.index_reader_options()["cache_index_file"] = false;
  } else {
    meta.index_writer_options()["write_index_cache"] = true;
    meta.index_reader_options()["cache_index_block"] = false;
    meta.index_reader_options()["cache_index_file"] = true;
  }
  return meta;
}

IndexMeta PrepareHnswMeta(MetricType metric_type, int dim, int M, int efConstruction) {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissHnsw);
  meta.common_params()["dim"] = dim;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric_type;
  meta.index_params()["M"] = M;
  meta.index_params()["efConstruction"] = efConstruction;

  meta.index_writer_options()["write_index_cache"] = true;
  meta.index_reader_options()["cache_index_file"] = true;
  return meta;
}

void Build(IndexCache* cache, const tenann::IndexMeta& meta, const std::string& index_path,
           tenann::ArraySeqView base_col) {
  tenann::OmpSetNumThreads(4);
  cache->SetCapacity(1);
  // build and write index
  auto index_builder = tenann::IndexFactory::CreateBuilderFromMeta(meta);
  index_builder->index_writer()->SetIndexCache(cache);
  index_builder->Open(index_path).Add({base_col}).Flush().Close();
}

void Search(IndexCache* cache, const tenann::IndexMeta& meta, const std::string& index_path,
            tenann::PrimitiveSeqView query_view) {
  tenann::OmpSetNumThreads(4);
  cache->SetCapacity(1);
  auto ann_searcher = tenann::AnnSearcherFactory::CreateSearcherFromMeta(meta);
  ann_searcher->index_reader()->SetIndexCache(cache);
  ann_searcher->ReadIndex(index_path);

  constexpr const int k = 10;
  std::vector<int64_t> result_ids(1 * k);
  std::vector<float> result_distances(1 * k);

  ann_searcher->AnnSearch(query_view, k, result_ids.data(),
                          reinterpret_cast<uint8_t*>(result_distances.data()));
}

int main(int argc, char const* argv[]) {
  tenann::SetLogLevel(T_LOG_LEVEL_FATAL);
  tenann::SetVLogLevel(VERBOSE_CRITICAL);

  tenann::SetLogLevel(T_LOG_LEVEL_DEBUG);
  tenann::SetVLogLevel(VERBOSE_DEBUG);

  auto cache = std::make_shared<tenann::IndexCache>(10);
  auto base = RandomVectors(nb, dim, 0);
  auto query = RandomVectors(nq, dim, 1);

  auto base_view = tenann::ArraySeqView{.data = reinterpret_cast<uint8_t*>(base.data()),
                                        .dim = dim,
                                        .size = static_cast<uint32_t>(nb),
                                        .elem_type = tenann::PrimitiveType::kFloatType};

  auto query_view = tenann::PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(query.data()),
                                             .size = dim,
                                             .elem_type = tenann::PrimitiveType::kFloatType};
  auto hnsw_meta = PrepareHnswMeta(MetricType::kCosineSimilarity, dim, 8, 40);
  auto ivfpq_meta = PrepareIvfPqMeta(MetricType::kCosineSimilarity, dim, 2, 2, dim / 2, true);

  // ParallelFor(0, 100, 0, [=](size_t id, size_t thread_id) {
  //   Build(cache.get(), hnsw_meta,
  //         fmt::format("/data/home/petrizhang/data/index/hnsw_{}_{}d", id, dim), base_view);
  //   T_LOG(INFO) << "Built hnsw_" << id;
  // });

  // ParallelFor(0, 100, 0, [=](size_t id, size_t thread_id) {
  //   auto meta = PrepareIvfPqMeta(MetricType::kCosineSimilarity, dim, 2, 64, 8, true);
  //   Build(cache.get(), ivfpq_meta,
  //         fmt::format("/data/home/petrizhang/data/index/ivfpq_{}_{}d", id, dim), base_view);
  //   T_LOG(INFO) << "Built ivfpq_" << id;
  // });

  ParallelFor(100, 140, 16, [=](size_t id, size_t thread_id) {
    if (id % 4 == 0) {
      T_LOG(INFO) << "Building hnsw_" << id;
      Build(cache.get(), hnsw_meta,
            fmt::format("/data/home/petrizhang/data/index/hnsw_{}_{}d", id, dim), base_view);
      T_LOG(INFO) << "Built hnsw_" << id;
    } else if (id % 3 == 0) {
      T_LOG(INFO) << "Building ivfpq_" << id;
      Build(cache.get(), ivfpq_meta,
            fmt::format("/data/home/petrizhang/data/index/ivfpq_{}_{}d", id, dim), base_view);
      T_LOG(INFO) << "Built ivfpq_" << id;
    } else if (id % 2 == 0) {
      T_LOG(INFO) << "Searching hnsw_" << thread_id;
      Search(cache.get(), hnsw_meta,
             fmt::format("/data/home/petrizhang/data/index/hnsw_{}_{}d", thread_id, dim),
             query_view);
    } else {
      T_LOG(INFO) << "Searching ivfpq_" << thread_id;
      Search(cache.get(), ivfpq_meta,
             fmt::format("/data/home/petrizhang/data/index/ivfpq_{}_{}d", thread_id, dim),
             query_view);
    }
  });

  return 0;
}
