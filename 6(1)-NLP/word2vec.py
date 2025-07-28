import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        self.vocab_size = vocab_size
        self.d_model = d_model

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        self.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_samples = 0.0
            
            for text in corpus:
                # 텍스트를 토큰화
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                if self.method == "cbow":
                    loss = self._train_cbow(tokens, criterion)
                else:  # skipgram
                    loss = self._train_skipgram(tokens, criterion)
                
                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping 추가
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
                    num_samples += 1.0
            
            if num_samples > 0:
                avg_loss = total_loss / num_samples
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def _train_cbow(
        self,
        tokens: list[int],
        criterion: nn.CrossEntropyLoss
    ) -> Tensor | None:
        # 구현하세요!
        if len(tokens) < 2 * self.window_size + 1:
            return None
        
        losses = []
        
        for i in range(self.window_size, len(tokens) - self.window_size):
            # 중심 단어
            target = tokens[i]
            
            # 주변 단어들 (context)
            context = []
            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i:
                    context.append(tokens[j])
            
            # CBOW: 주변 단어들로부터 중심 단어 예측
            context_tensor = torch.tensor(context, dtype=torch.long)
            target_tensor = torch.tensor([target], dtype=torch.long)
            
            # 주변 단어들의 임베딩 평균
            context_embeddings = self.embeddings(context_tensor)
            context_mean = context_embeddings.mean(dim=0, keepdim=True)
            
            # 예측
            output = self.weight(context_mean)
            loss = criterion(output, target_tensor)
            losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        return None

    def _train_skipgram(
        self,
        tokens: list[int],
        criterion: nn.CrossEntropyLoss
    ) -> Tensor | None:
        # 구현하세요!
        if len(tokens) < 2 * self.window_size + 1:
            return None
        
        losses = []
        
        for i in range(self.window_size, len(tokens) - self.window_size):
            # 중심 단어
            center_word = tokens[i]
            
            # 주변 단어들
            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i:
                    context_word = tokens[j]
                    
                    # Skip-gram: 중심 단어로부터 주변 단어 예측
                    center_tensor = torch.tensor([center_word], dtype=torch.long)
                    context_tensor = torch.tensor([context_word], dtype=torch.long)
                    
                    # 중심 단어 임베딩
                    center_embedding = self.embeddings(center_tensor)
                    
                    # 예측
                    output = self.weight(center_embedding)
                    loss = criterion(output, context_tensor)
                    losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        return None

    # 구현하세요!
    def forward(self, x: LongTensor) -> Tensor:
        # 임베딩 후 선형 변환
        embeddings = self.embeddings(x)
        output = self.weight(embeddings)
        return output