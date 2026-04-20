#include "tenann/index/index_cache_interface.h"

#include <atomic>

namespace tenann {

namespace {
std::atomic<IndexCacheInterface*> g_index_cache{nullptr};
}  // namespace

void SetGlobalIndexCache(IndexCacheInterface* cache) {
    g_index_cache.store(cache, std::memory_order_release);
}

IndexCacheInterface* GetGlobalIndexCache() {
    return g_index_cache.load(std::memory_order_acquire);
}

}  // namespace tenann
