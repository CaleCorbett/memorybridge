import re

with open("db/store.py", "r") as f:
    code = f.read()

# 1. Import functools
code = code.replace("import warnings\n", "import warnings\nimport functools\n")

# 2. Add _EmbeddingCache
class_def = """    return None


class _EmbeddingCache:
    \"\"\"Process-local cache of (profile -> (ids[], matrix)) for cosine search.
    Invalidated by any write to the profile's memories.\"\"\"
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def get(self, profile: str):
        with self._lock:
            return self._cache.get(profile)
            
    def set(self, profile: str, ids: list, matrix):
        with self._lock:
            self._cache[profile] = (ids, matrix)
            
    def invalidate(self, profile: str):
        with self._lock:
            self._cache.pop(profile, None)


class MemoryStore:"""
code = code.replace("    return None\n\n\nclass MemoryStore:", class_def)

# 3. Initialize cache in MemoryStore.__init__
init_replacement = """        self._pending_embed_lock = threading.Lock()
        self._embedding_cache = _EmbeddingCache()
        db_path.parent.mkdir(parents=True, exist_ok=True)"""
code = code.replace("        self._pending_embed_lock = threading.Lock()\n        db_path.parent.mkdir(parents=True, exist_ok=True)", init_replacement)

# 4. _embed_query_cached
embed_query = """        self._ensure_maintenance()

    @functools.lru_cache(maxsize=64)
    def _embed_query_cached(self, query: str) -> tuple[float, ...]:
        \"\"\"Cache the most recent queries to avoid re-embedding.\"\"\"
        return tuple(self._embed_texts([query])[0])

    def add_profile("""
code = code.replace("        self._ensure_maintenance()\n\n    def add_profile(", embed_query)

# 5. Invalidate cache on add, edit, delete, backfill
code = code.replace("        return merged_id\n\n        try:", "        self._embedding_cache.invalidate(profile)\n        return merged_id\n\n        try:")
code = code.replace("                self._conn.commit()\n            # Embed-on-write", "                self._conn.commit()\n            self._embedding_cache.invalidate(profile)\n            # Embed-on-write")
code = code.replace("        if updated and \"content\" in fields:\n            threading.Thread", "        if updated and \"content\" in fields:\n            self._embedding_cache.invalidate(profile)\n            threading.Thread")
code = code.replace("            self._conn.commit()\n        return tc", "            self._conn.commit()\n        self._embedding_cache.invalidate(profile)\n        return tc")
code = code.replace("                self._conn.commit()\n        return to_archive", "                self._conn.commit()\n            self._embedding_cache.invalidate(profile)\n        return to_archive")
code = code.replace("        self._conn.commit()\n        return len(ids)", "        self._conn.commit()\n        self._embedding_cache.invalidate(profile)\n        return len(ids)")

# 6. Cache usage in search_semantic
q_vec_repl = """        try:
            q_vec = self._embed_query_cached(query)
        except Exception as e:"""
code = code.replace("        try:\n            q_vec = self._embed_texts([query])[0]\n        except Exception as e:", q_vec_repl)

mat_repl = """        import numpy as np
        
        cached = self._embedding_cache.get(profile)
        if cached is not None:
            ids, M = cached
            mismatched = 0
        else:
            ids: list[str] = []
            mat: list[list[float]] = []
            mismatched = 0
            for row in rows:
                vec = _unpack_vector(row["vector"])
                if not isinstance(vec, list) or len(vec) != q_len:
                    mismatched += 1
                    continue
                ids.append(row["id"])
                mat.append(vec)
            
            if mat:
                M = np.asarray(mat, dtype=np.float32)
                self._embedding_cache.set(profile, ids, M)
            else:
                M = np.array([])

        scored = []
        if len(ids) > 0 and len(M) > 0:
            q = np.asarray(q_vec, dtype=np.float32)
            q_norm = float(np.linalg.norm(q))
            if q_norm > 0:
                denom = np.linalg.norm(M, axis=1) * q_norm
                sims = np.divide(M @ q, denom,
                                 out=np.zeros(len(M), dtype=np.float32),
                                 where=denom > 0)
                scored = list(zip(ids, sims.tolist()))"""

old_mat_repl = """        import numpy as np
        ids: list[str] = []
        mat: list[list[float]] = []
        mismatched = 0
        for row in rows:
            vec = _unpack_vector(row["vector"])
            if not isinstance(vec, list) or len(vec) != q_len:
                mismatched += 1
                continue
            ids.append(row["id"])
            mat.append(vec)

        scored = []
        if mat:
            q = np.asarray(q_vec, dtype=np.float32)
            q_norm = float(np.linalg.norm(q))
            if q_norm > 0:
                M = np.asarray(mat, dtype=np.float32)          # (n, d)
                denom = np.linalg.norm(M, axis=1) * q_norm
                sims = np.divide(M @ q, denom,
                                 out=np.zeros(len(mat), dtype=np.float32),
                                 where=denom > 0)
                scored = list(zip(ids, sims.tolist()))"""

code = code.replace(old_mat_repl, mat_repl)

# 7. Concurrency in search_hybrid
old_hybrid = """        keyword_results = self.search(profile, query, category=category,
                                      limit=limit * 2, max_tokens=max_tokens * 2,
                                      min_confidence=min_confidence)
        semantic_results = self.search_semantic(profile, query, limit=limit * 2,
                                                max_tokens=max_tokens * 2)"""

new_hybrid = """        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            kw_future = pool.submit(self.search, profile, query, category=category,
                                    limit=limit * 2, max_tokens=max_tokens * 2,
                                    min_confidence=min_confidence)
            sem_future = pool.submit(self.search_semantic, profile, query, limit=limit * 2,
                                     max_tokens=max_tokens * 2)
            
            keyword_results = kw_future.result()
            semantic_results = sem_future.result()"""

code = code.replace(old_hybrid, new_hybrid)

with open("db/store.py", "w") as f:
    f.write(code)

with open("server.py", "r") as f:
    scode = f.read()

# server.py changes
old_server_search = """    profile: str = None,
    recency_boost: bool = True,
    include_related: bool = False,
    min_confidence: float = 0.0,
) -> str:
    \"\"\"
    Search memories using FTS5 BM25 with optional token budget."""

new_server_search = """    profile: str = None,
    recency_boost: bool = True,
    include_related: bool = False,
    min_confidence: float = 0.0,
    search_mode: str = "hybrid",
) -> str:
    \"\"\"
    Search memories using BM25, semantic, or hybrid (RRF) search."""

scode = scode.replace(old_server_search, new_server_search)

old_server_logic = """    if category and category not in VALID_CATEGORIES:
        return json.dumps({"error": f"Invalid category. Valid: {VALID_CATEGORIES}"})

    # Phase 4: hybrid BM25 + semantic search (falls back to FTS5 if no embeddings built)
    results = _store.search_hybrid(profile, query, category=category,
                                   limit=limit, max_tokens=max_tokens,
                                   recency_boost=recency_boost,
                                   include_related=include_related,
                                   min_confidence=min_confidence)"""

new_server_logic = """    if category and category not in VALID_CATEGORIES:
        return json.dumps({"error": f"Invalid category. Valid: {VALID_CATEGORIES}"})

    if search_mode not in ("hybrid", "keyword", "semantic"):
        return json.dumps({"error": "Invalid search_mode. Use 'hybrid', 'keyword', or 'semantic'"})

    if search_mode == "keyword":
        results = _store.search(profile, query, category=category,
                                limit=limit, max_tokens=max_tokens,
                                recency_boost=recency_boost,
                                include_related=include_related,
                                min_confidence=min_confidence)
    elif search_mode == "semantic":
        results = _store.search_semantic(profile, query, limit=limit, max_tokens=max_tokens)
    else:
        results = _store.search_hybrid(profile, query, category=category,
                                       limit=limit, max_tokens=max_tokens,
                                       recency_boost=recency_boost,
                                       include_related=include_related,
                                       min_confidence=min_confidence)"""

scode = scode.replace(old_server_logic, new_server_logic)

with open("server.py", "w") as f:
    f.write(scode)

print("Done")
