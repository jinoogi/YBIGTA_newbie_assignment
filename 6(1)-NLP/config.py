from typing import Literal


device = "cpu"
d_model = 256

# Word2Vec
window_size = 5  # 7에서 5로 줄임
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 5e-04  # 1e-03에서 5e-04로 줄임
num_epochs_word2vec = 10  # 5에서 10으로 증가

# GRU
hidden_size = 256
num_classes = 4
lr = 1e-03  # 5e-03에서 1e-03으로 줄임
num_epochs = 150  # 100에서 150으로 증가
batch_size = 32  # 16에서 32로 증가