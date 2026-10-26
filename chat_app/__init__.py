"""Gemini 스타일 웹 챗봇(web_chat.py)의 기능별 모듈.

- config: 환경변수 · 모델/검색 옵션 · 요금 상수
- styles: Gemini 다크 테마 CSS
- db: PostgreSQL 연결 풀 · PostgresSaver · 대화 목록/메시지 테이블 · LangGraph 캐시
- cache: LangGraph BaseCache의 PostgreSQL 구현 (PostgresCache)
- grounding_cache: Vertex AI Grounding 답변 캐시 미들웨어 (wrap_model_call)
- agent: 모델 생성 + 검색 방식별 에이전트 빌드
- usage: 턴별 토큰 합산 · 예상 비용 · 검색 반영 여부 판단
- state: 세션 상태(현재 대화, 대기 중인 질문)와 대화 전환 동작
- ui: 사이드바 · 상단 바 · 첫 화면 · 메시지 · 스트리밍 답변 렌더링
"""
