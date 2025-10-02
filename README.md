# LLM 개념지도

```mermaid

flowchart LR
  A["LLM"]

  %% 트랜스포머
  A --> B["트랜스포머"]
    subgraph 트랜스포머
        B --> BA["인코더, 디코더"]
          BA --> BAA["멀티헤드 어텐션"]
          BA --> BAB["피드포워드"]
          BA --> BAC["층정규화"]
    end

  %% 모델학습
  A --> C["모델학습"]
    subgraph 모델학습
        C --> CA["사전학습"]
        C --> CB["지도 미세조정"]
          CB --> CBA["지도 데이터셋"]
        C --> CC["선호 학습"]
          CC --> CCA["강화학습"]
            CCA --> CCAA["RLHF"]
              CCAA --> CCAAA["리워드 모델"]
              CCAA --> CCAAB["선호 데이터셋"]
          CC --> CCB["DPO 학습"]
    end


  %% 학습성능
  A --> D["학습성능"]
    subgraph 학습성능
        D --> DA["GPU메모리 구성요소"]
          DA --> DAA["데이터의 종류"]
          DA --> DAB["데이터 타입"]
          DA --> DAC["양자화"]
        D --> DB["단일 GPU효율"]
          DB --> DBA["체크포인팅"]
          DB --> DBB["그레이디언트 누적"]
        D --> DC["분산 GPU효율"]
          DC --> DCA["모델 병렬화"]
            DCA --> DCAA["파이프라인 병렬화"]
            DCA --> DCAB["텐서 병렬화"]
          DC --> DCB["데이터 병렬화"]
            DCB --> DCBA["ZeRO방법"]
        D --> DD["일부 학습"]
          DD --> DDA["PEFT"]
            DDA --> DDAA["LoRA"]
            DDA --> DDAB["QLoRA"]
    end
  
  %% 추론성능
  A --> E["추론성능"]
    subgraph 추론성능
      E --> EA["성능 저하"]
          EA --> EAA["KV(Key-Value)캐시"]
              EAA --> EAAA["멀티 쿼리 어텐션"]
              EAA --> EAAB["그룹 쿼리 어텐션"]
          EA --> EAB["데이터 양자화"]
              EAB --> EABA["비츠앤바이츠"]
              EAB --> EABB["GPTQ"]
              EAB --> EABC["AWQ"]
          EA --> EAC["지식증류"]
              EAC --> EACA["선생, 학생 모델"]
      E --> EB["성능 유지"]
          EB --> EBA["배치 전략"]
              EBA --> EBAA["일반 배치"]
              EBA --> EBAB["동적 배치"]
              EBA --> EBAC["연속 배치"]
          EB --> EBB["플래시 어텐션"]
              EBB --> EBBA["SRAM"]
              EBB --> EBBB["HBM"]
          EB --> EBC["상대적 위치 인코딩"]
              EBC --> EBCA["RoPE"]
              EBC --> EBCB["ALiBi"]
          EB --> EBD["효율적인 추론 전략"]
              EBD --> EBDA["커널퓨전"]
              EBD --> EBDB["페이지 어텐션"]
              EBD --> EBDC["추측 디코딩"]
                  EBDC --> EBDCA["드래프트 모델"]
                  EBDC --> EBDCB["타깃 모델"]
          EB --> EBE["vLLM라이브러리"]
    end

  %% RAG
  A --> F["RAG"]
    subgraph RAG
      F --> FA["임베딩 벡터"]
          FA --> FAA["임베딩"]
              FAA --> FAAA["단어 임베딩"]
                  FAAA --> FAAAA["원-핫 인코딩"]
                  FAAA --> FAAAB["백오브워즈"]
                  FAAA --> FAAAC["TF-IDF"]
                  FAAA --> FAAAD["워드투벡"]
                      FAAAD --> FAAADA["밀집임베딩"]
              FAA --> FAAB["문장 임베딩"]
                  FAAB --> FAABA["BERT모델"]
                      FAABA --> FAABAA["언어 모델 -> 임베딩 모델 학습"]
                      FAABA --> FAABAB["바이인코더"]
                      FAABA --> FAABAC["교차인코더"]
          FA --> FAB["검색"]
              FAB --> FABA["키워드 검색"]
                  FABA --> FABAA["BM25"]
              FAB --> FABB["의미 검색"]
              FAB --> FABC["하이브리드 검색"]
                  FABC --> FABCA["키워드 검색 + 의미검색"]
      F --> FB["벡터 데이터베이스"]
          FB --> FBA["KNN"]
          FB --> FBB["ANN"]
              FBB --> FBBA["HNSW"]
      F --> FC["LLM캐시"]
          FC --> FCA["일치캐시"]
          FC --> FCB["유사검색캐시"]
      F --> FD["LLM응답 검증"]
      F --> FE["LLM기록 로깅"]
    end

  %% LLMOps
  A --> G["LLMOps"]
    subgraph LLMOps
      G --> GA["데이터 준비"]
      G --> GB["모델학습"]
      G --> GC["모델저장소"]
      G --> GD["모델 배포"]
      G --> GA["모니터링"]
    end

  %% 멀티모달 LLM
  A --> H["멀티모달 LLM"]

  %% LLM에이전트
  A --> I["LLM에이전트"] 

click B "https://example.com/plan"
click C "https://example.com/dev"
click D "https://example.com/test"


