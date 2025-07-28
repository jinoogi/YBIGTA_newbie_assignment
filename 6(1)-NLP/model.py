from torch import nn, Tensor, LongTensor
from gru import GRU


class MyGRULanguageModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_classes: int,
        embeddings: int | Tensor
    ) -> None:
        super().__init__()
        if isinstance(embeddings, int):
            self.embeddings = nn.Embedding(embeddings, d_model)
        elif isinstance(embeddings, Tensor):
            self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

        # Bidirectional GRU 사용
        self.gru = GRU(d_model, hidden_size, bidirectional=True)
        
        # 더 복잡한 head 구조
        gru_output_size = hidden_size * 2  # bidirectional이므로 2배
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids: LongTensor) -> Tensor:
        inputs = self.embeddings(input_ids)
        last_hidden_state = self.gru(inputs)  # (batch_size, hidden_size * 2)
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.head(last_hidden_state)  # (batch_size, num_classes)
        return logits