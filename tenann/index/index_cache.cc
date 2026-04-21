#include "tenann/index/index_cache.h"

#include <atomic>

namespace tenann {

namespace {
std::atomic<IndexCache*> g_index_cache{nullptr};
}  // namespace

void SetGlobalIndexCache(IndexCache* cache) {
    g_index_cache.store(cache, std::memory_order_release);
}

IndexCache* GetGlobalIndexCache() {
    return g_index_cache.load(std::memory_order_acquire);
}

}  // namespace tenann
