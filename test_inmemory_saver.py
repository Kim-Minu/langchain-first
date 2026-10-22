import os
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. API 키 확인
google_api_key = os.getenv("GCP_API_KEY")
if not google_api_key or google_api_key == "your_gemini_api_key_here":
    raise ValueError(
        "GCP_API_KEY가 설정되지 않았습니다.\n"
        "환경변수 GCP_API_KEY를 설정하거나 .env 파일에 실제 발급받은 API 키를 입력해 주세요."
    )

# 2. 모델 설정
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    google_api_key=google_api_key,
    project="iceu-songpa05",
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

# 3. InMemorySaver 인스턴스 생성
checkpointer = InMemorySaver()

# 4. checkpointer를 등록한 에이전트 생성
agent = create_agent(
    model=model,
    tools=[],  # 대화 기억 자체를 테스트하기 위해 tools는 빈 목록
    checkpointer=checkpointer,
    system_prompt="당신은 사용자와의 대화 내용을 정확하게 기억하는 비서입니다.",
)


def run_test():
    print("=" * 60)
    print("🚀 InMemorySaver 동작 검증 테스트 시작")
    print("=" * 60)

    # ---------------------------------------------------------
    # [시나리오 1] 동일 thread_id ("session_minwoo") 연속 대화
    # ---------------------------------------------------------
    config_session_1 = {"configurable": {"thread_id": "session_minwoo"}}

    print("\n[시나리오 1-1] 세션 1: 정보 알려주기")
    q1 = "안녕! 내 이름은 민우야. 내가 제일 좋아하는 과일은 '망고'야."
    print(f"👤 사용자: {q1}")
    res1 = agent.invoke({"messages": [{"role": "user", "content": q1}]}, config=config_session_1)
    ans1 = res1["messages"][-1].content
    print(f"🤖 에이전트: {ans1}\n")

    print("[시나리오 1-2] 세션 1: 이전 정보 물어보기 (기억 여부 확인)")
    q2 = "내 이름이 뭐라고 했지? 그리고 내가 좋아하는 과일은?"
    print(f"👤 사용자: {q2}")
    res2 = agent.invoke({"messages": [{"role": "user", "content": q2}]}, config=config_session_1)
    ans2 = res2["messages"][-1].content
    print(f"🤖 에이전트: {ans2}\n")

    # 1차 검증
    has_name = "민우" in ans2
    has_fruit = "망고" in ans2
    if has_name and has_fruit:
        print("✅ [검증 성공] 세션 1에서 이전 대화(이름, 과일)를 정확하게 기억하고 있습니다!")
    else:
        print("❌ [검증 실패] 세션 1에서 이전 정보를 기억하지 못했습니다.")

    # ---------------------------------------------------------
    # [시나리오 2] 새로운 thread_id ("session_stranger") 대화
    # ---------------------------------------------------------
    config_session_2 = {"configurable": {"thread_id": "session_stranger"}}

    print("\n" + "-" * 60)
    print("[시나리오 2] 세션 2: 새로운 사용자 세션에서 질문 (격리 여부 확인)")
    q3 = "내 이름이 뭔지 알고 있니?"
    print(f"👤 사용자: {q3}")
    res3 = agent.invoke({"messages": [{"role": "user", "content": q3}]}, config=config_session_2)
    ans3 = res3["messages"][-1].content
    print(f"🤖 에이전트: {ans3}\n")

    # 2차 검증 (새 세션에서는 민우라는 이름을 몰라야 정상)
    if "민우" not in ans3:
        print("✅ [검증 성공] 세션 2에서는 세션 1의 정보가 완벽히 격리되어 모릅니다!")
    else:
        print("❌ [검증 실패] 세션 간 메모리가 격리되지 않고 유출되었습니다.")

    # ---------------------------------------------------------
    # [시나리오 3] InMemorySaver에 실제로 저장된 State 직접 확인
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[시나리오 3] InMemorySaver에 저장된 세션 1의 메시지 히스토리 직접 조회")
    saved_state = agent.get_state(config_session_1)
    saved_messages = saved_state.values.get("messages", [])

    print(f"📦 세션 1에 누적 저장된 총 메시지 수: {len(saved_messages)}개")
    for idx, msg in enumerate(saved_messages, 1):
        role = getattr(msg, "type", "unknown")
        # 내용이 긴 경우 말줄임표 처리
        preview = msg.content if isinstance(msg.content, str) else str(msg.content)
        preview = preview.replace("\n", " ")[:60]
        print(f"   [{idx}] {role.upper():5s}: {preview}...")

    print("\n" + "=" * 60)
    print("🎉 모든 InMemorySaver 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    run_test()

