# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    # 시 감정 분석 데이터셋에서 텍스트 로드
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    
    # train, validation, test 데이터의 모든 텍스트를 corpus에 추가
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            corpus.append(item["verse_text"])
    
    return corpus