"""PostgreSQL 저장소.

- checkpoints* 테이블 (PostgresSaver 관리): 모델이 이어서 대화할 그래프 상태 전체
- chat_threads / chat_messages 테이블 (앱 관리): 사이드바 목록, 화면 메시지, 토큰·비용 메타데이터
"""

import streamlit as st
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from chat_app.cache import PostgresCache
from chat_app.config import DATABASE_URL

# psycopg3는 한 번의 execute에 여러 SQL 문을 보낼 수 없으므로 문장별로 나눠 둠
CHAT_TABLES_DDL = [
    """
    CREATE TABLE IF NOT EXISTS chat_threads (
        thread_id  TEXT PRIMARY KEY,
        title      TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id         BIGSERIAL PRIMARY KEY,
        thread_id  TEXT NOT NULL REFERENCES chat_threads (thread_id) ON DELETE CASCADE,
        role       TEXT NOT NULL,
        content    TEXT NOT NULL,
        meta       JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS chat_messages_thread_idx ON chat_messages (thread_id, id)",
]


@st.cache_resource
def get_db_pool() -> ConnectionPool:
    # PostgresSaver 요구사항: autocommit(setup 테이블 생성 커밋), dict_row(행을 dict로 읽음), prepare_threshold=0
    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    pool.open(wait=True, timeout=5)
    return pool


@st.cache_resource
def get_checkpointer() -> PostgresSaver:
    pool = get_db_pool()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # checkpoints 관련 테이블 생성/마이그레이션 (이미 있으면 건너뜀)
    with pool.connection() as conn:
        for ddl in CHAT_TABLES_DDL:
            conn.execute(ddl)
    return checkpointer


@st.cache_resource
def get_graph_cache() -> PostgresCache:
    """LangGraph 캐시 (GroundingCacheMiddleware가 Vertex AI Grounding 답변 저장에 사용)."""
    cache = PostgresCache(get_db_pool())
    cache.setup()
    return cache


def list_threads() -> list[dict]:
    with get_db_pool().connection() as conn:
        return conn.execute(
            "SELECT thread_id, title FROM chat_threads ORDER BY updated_at DESC LIMIT 50"
        ).fetchall()


def load_messages(thread_id: str) -> list[dict]:
    with get_db_pool().connection() as conn:
        rows = conn.execute(
            "SELECT role, content, meta FROM chat_messages WHERE thread_id = %s ORDER BY id",
            (thread_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], **r["meta"]} for r in rows]


def save_message(thread_id: str, role: str, content: str, meta: dict | None = None, title: str | None = None):
    """메시지를 저장하고 대화의 updated_at을 갱신합니다. 첫 메시지면 대화(thread)를 함께 생성합니다."""
    with get_db_pool().connection() as conn, conn.transaction():
        conn.execute(
            """
            INSERT INTO chat_threads (thread_id, title) VALUES (%s, %s)
            ON CONFLICT (thread_id) DO UPDATE SET updated_at = now()
            """,
            (thread_id, title or content[:30]),
        )
        conn.execute(
            "INSERT INTO chat_messages (thread_id, role, content, meta) VALUES (%s, %s, %s, %s)",
            (thread_id, role, content, Jsonb(meta or {})),
        )


def rename_thread(thread_id: str, title: str):
    with get_db_pool().connection() as conn:
        conn.execute("UPDATE chat_threads SET title = %s WHERE thread_id = %s", (title, thread_id))


def delete_thread(thread_id: str):
    """대화를 완전히 삭제합니다: 화면용 목록/메시지(ON DELETE CASCADE) + 에이전트 기억(체크포인트)."""
    get_checkpointer().delete_thread(thread_id)
    with get_db_pool().connection() as conn:
        conn.execute("DELETE FROM chat_threads WHERE thread_id = %s", (thread_id,))


def get_usage_totals() -> dict:
    with get_db_pool().connection() as conn:
        return conn.execute(
            """
            SELECT COALESCE(SUM((meta->>'total_tokens')::bigint), 0) AS tokens,
                   COALESCE(SUM((meta->>'cost_usd')::float8), 0)     AS cost_usd
            FROM chat_messages WHERE role = 'assistant'
            """
        ).fetchone()
