# Prompting 기법 비교 보고서

## 1. 정답률 비교 (Accuracy Comparison)

| Prompting 기법   | 0-shot | 3-shot | 5-shot |
|------------------|--------|--------|--------|
| Direct Prompting | 0.16   | 0.18   | 0.18   |
| CoT Prompting    | 0.70   | 0.63   | 0.71   |
| My Prompting     | 0.78   | 0.71   | 0.72   |

---

## 2. CoT Prompting이 Direct Prompting보다 더 나은 이유

CoT (Chain-of-Thought) Prompting은 단순히 정답만 요구하는 Direct Prompting과 달리, **문제 해결의 사고 과정을 언어로 드러내는 방식**임.

- **복잡한 추론 필요 문제에 강함**  
  중간 과정을 명시함으로써 모델이 문제를 단계별로 이해하고 해결하도록 유도.

- **추론 오류 감소**  
  Direct Prompting은 답을 단번에 생성하려다 보니 실수가 많지만, CoT는 점진적으로 생각을 전개하기 때문에 오류 가능성이 감소함.

- **모델의 내부 구조 활용 극대화**  
  LLM은 연속적인 언어 패턴에 익숙하기 때문에, 사고의 흐름을 기술하면 더 일관된 출력을 생성할 수 있음.

---

## 3. My Prompting 기법이 CoT보다 더 나을 수 있는 이유

내 My Prompting 기법은 CoT의 장점을 계승하면서도 다음과 같은 측면에서 더 나은 성능을 보여줄 수 있음:


- **문제 유형에 특화된 맞춤 프롬프트**  
  CoT는 일반적인 사고 전개에 그치는 반면, My Prompting은 특정 Role을 부여해서 문제 도메인에 맞는 페르소나를 부여함.

- **노이즈 감소**  
  CoT는 종종 불필요하거나 장황한 출력을 유도할 수 있는데, My Prompting은 좀더 간결하고 목적지향적인 프롬프트를 통해 더 정제된 출력을 도출함.
