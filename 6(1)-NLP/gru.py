import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        self.hidden_size = hidden_size
        
        # GRU gates: reset gate, update gate, candidate hidden state
        self.r_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.z_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.h_candidate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        # GRU 수식 구현
        # r_t = σ(W_r * [x_t, h_{t-1}] + b_r)
        # z_t = σ(W_z * [x_t, h_{t-1}] + b_z)
        # h̃_t = tanh(W_h * [x_t, r_t ⊙ h_{t-1}] + b_h)
        # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        
        # 입력과 이전 hidden state 결합
        combined = torch.cat([x, h], dim=-1)
        
        # Reset gate
        r = torch.sigmoid(self.r_gate(combined))
        
        # Update gate
        z = torch.sigmoid(self.z_gate(combined))
        
        # Candidate hidden state
        h_candidate_input = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.h_candidate(h_candidate_input))
        
        # New hidden state
        h_new = (1 - z) * h + z * h_tilde
        
        return h_new


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        if bidirectional:
            self.cell_forward = GRUCell(input_size, hidden_size)
            self.cell_backward = GRUCell(input_size, hidden_size)
        else:
            self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        # inputs shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = inputs.shape
        
        if self.bidirectional:
            # Forward pass
            h_forward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            forward_outputs: list[Tensor] = []
            
            for t in range(seq_len):
                x_t = inputs[:, t, :]
                h_forward = self.cell_forward(x_t, h_forward)
                forward_outputs.append(h_forward)
            
            # Backward pass
            h_backward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            backward_outputs: list[Tensor] = []
            
            for t in range(seq_len - 1, -1, -1):
                x_t = inputs[:, t, :]
                h_backward = self.cell_backward(x_t, h_backward)
                backward_outputs.insert(0, h_backward)
            
            # Concatenate forward and backward
            outputs: list[Tensor] = []
            for f, b in zip(forward_outputs, backward_outputs):
                outputs.append(torch.cat([f, b], dim=-1))
            
            # Return last output
            return outputs[-1]
        else:
            # 초기 hidden state
            h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            
            # 각 시퀀스 스텝에 대해 GRU 셀 적용
            for t in range(seq_len):
                x_t = inputs[:, t, :]  # (batch_size, input_size)
                h = self.cell(x_t, h)
            
            # 마지막 hidden state 반환
            return h