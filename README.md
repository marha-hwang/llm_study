# 프로젝트 마인드맵 예제

```mermaid
graph TD
    A[프로젝트] 
    A --> B[기획]
    A --> C[개발]
    A --> D[테스트]

    B --> B1[시장 조사]
    B --> B2[요구사항 정의]
    C --> C1[프론트엔드]
    C --> C2[백엔드]
    D --> D1[단위 테스트]
    D --> D2[통합 테스트]

    %% 클릭 링크 설정
    click B "https://example.com/plan" "기획 페이지로 이동"
    click C "https://example.com/dev" "개발 페이지로 이동"
    click D "https://example.com/test" "테스트 페이지로 이동"
    click B1 "https://example.com/research" "시장 조사 참고자료"
    click C1 "https://example.com/frontend" "프론트엔드 문서"
