"""LangGraph 캐시(BaseCache)의 PostgreSQL 구현.

LangGraph에는 InMemoryCache · RedisCache만 내장되어 있어, 앱 재시작 후에도 캐시가 유지되도록
이미 사용 중인 PostgreSQL에 저장하는 구현을 둡니다. 그래프 컴파일 시 cache로 넘기면 CachePolicy가 붙은
노드/task 결과가 저장되고, 이 앱에서는 GroundingCacheMiddleware가 직접 get/set을 호출해 Grounding 답변을 저장합니다.
"""

from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol
from psycopg_pool import ConnectionPool

CACHE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS langgraph_cache (
    ns         TEXT[] NOT NULL,
    key        TEXT NOT NULL,
    enc        TEXT NOT NULL,
    val        BYTEA NOT NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ns, key)
)
"""


class PostgresCache(BaseCache[ValueT]):
    def __init__(self, pool: ConnectionPool, *, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self.pool = pool

    def setup(self):
        with self.pool.connection() as conn:
            conn.execute(CACHE_TABLE_DDL)

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """만료되지 않은 캐시 값을 조회합니다."""
        values: dict[FullKey, ValueT] = {}
        with self.pool.connection() as conn:
            for ns, key in keys:
                row = conn.execute(
                    """
                    SELECT enc, val FROM langgraph_cache
                    WHERE ns = %s AND key = %s AND (expires_at IS NULL OR expires_at > now())
                    """,
                    (list(ns), key),
                ).fetchone()
                if row:
                    values[(ns, key)] = self.serde.loads_typed((row["enc"], row["val"]))
        return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        return self.get(keys)

    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """값을 저장합니다. ttl(초)이 None이면 만료 없음."""
        with self.pool.connection() as conn, conn.transaction():
            for (ns, key), (value, ttl) in pairs.items():
                enc, val = self.serde.dumps_typed(value)
                conn.execute(
                    """
                    INSERT INTO langgraph_cache (ns, key, enc, val, expires_at)
                    VALUES (%s, %s, %s, %s, CASE WHEN %s::int IS NULL THEN NULL
                                                 ELSE now() + make_interval(secs => %s::int) END)
                    ON CONFLICT (ns, key) DO UPDATE
                    SET enc = EXCLUDED.enc, val = EXCLUDED.val,
                        expires_at = EXCLUDED.expires_at, created_at = now()
                    """,
                    (list(ns), key, enc, val, ttl, ttl),
                )
            # 만료된 항목 정리
            conn.execute("DELETE FROM langgraph_cache WHERE expires_at <= now()")

    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        self.set(pairs)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """네임스페이스의 캐시를 삭제합니다. 지정하지 않으면 전체 삭제."""
        with self.pool.connection() as conn:
            if namespaces is None:
                conn.execute("DELETE FROM langgraph_cache")
            else:
                for ns in namespaces:
                    conn.execute("DELETE FROM langgraph_cache WHERE ns = %s", (list(ns),))

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        self.clear(namespaces)

    def count_valid(self) -> int:
        """만료되지 않은 캐시 항목 수."""
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT count(*) AS n FROM langgraph_cache WHERE expires_at IS NULL OR expires_at > now()"
            ).fetchone()["n"]
