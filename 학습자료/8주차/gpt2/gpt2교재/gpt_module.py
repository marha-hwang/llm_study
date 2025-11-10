import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # input + 1 위치로 target를 구성하고, stride만큼 이동하면서 데이터셋 구성
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

class MultiHeadAttention(nn.Module):
    # d_in= 임베딩 차원
    # d_out= 임베딩 차원
    # context_length= 한번에 처리 가능한 문맥크기
    # num_heads= 어텐션 헤드 개수
    # dropout= 드롭아웃 비율
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads

        # 출력차원을 어텐션헤드의 개수로 나눈다.        
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        # Linear도 결국 (d_in, d_out)크기의 행렬
        # 가중치 배열을 초기화하는 부분
        # (임베딩 크기, 임베딩 크기)만큼의 행렬을 생성한다.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 어텐션 출력을 병합하기 위한 레이어
        self.out_proj = nn.Linear(d_out, d_out) 

        self.dropout = nn.Dropout(dropout)

        # 마스킹을 위한 행렬
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    # 입력 : (배치크기 :한번에 병렬적으로 추론할 단어의 수, 현재배치의 토큰 수, 토큰 임베딩 크기)
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 가중치배열과 연산하여 Q, K, V를 만드는 부분
        # (6, 3) * (3, 3) = (6, 3)
        keys = self.W_key(x)  # Shape: (배치크기, 현재배치의 토큰 수, 토큰 임베딩 크기)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 차원을 추가하여 head_dim 행렬을 나눈다.
        # 마지막 차원을 나눈다: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 1번, 2번 인덱스 차원의 행과 열을 바꾼다.
        # 각 헤드의 토큰을 병렬적으로 처리하기 위해 바꾸는 과정
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # queries : (b, num_heads, num_tokens, head_dim)
        # keys.transpose(2,3) : (b, num_heads, head_dim, num_tokens)
        # 출력 : (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # 마스킹 연산 수행
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)


        # attn_weights : (b, num_heads, num_tokens, num_tokens)
        # values : (b, num_heads, num_tokens, head_dim)
        # 출력: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # contiguous() : view()함수를 사용하기 위해 벡터들을 메모리상에서 순서대로 배치시킴
        # view() : 분리된 헤드를 결합시켜 (b, num_tokens, d_out)와 같은 입력벡터와 동일한 차원으로 만든다.
        # 헤드 결합, self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 어텐션 블록이 여러 헤드를 통해 얻은 문맥 정보를 최종적으로 요약하고 압축하여 모델의 다음 단계로 전달하는 역할
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    # 정규화하여 입력과 같은 차원의 출력을 반환
    # scale, shift파라미터를 조정하여 성능향상
    def forward(self, x):
        # 소프트맥스 정규화 : 평균이 0, 분산이 1이 되도록 계산
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 토큰 임베딩의 차원을 입력으로 받아서 4배로 차원을 확장시키고
        # GELU함수 적용
        # 4배 확장된 입력을 받아서 원래 차원으로 다시 복귀시킴
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    # 입력 : (배치크기 :한번에 병렬적으로 추론할 단어의 수, 현재배치의 토큰 수, 토큰 임베딩 크기)
    def forward(self, x):
        ##### 멀티헤드어텐션 연산 부분
        # 잔차연결값 저장
        shortcut = x
        # 정규화
        x = self.norm1(x)
        # 멀티헤드어텐션 연산 수행
        x = self.att(x)   # 출력 : (배치크기 :한번에 병렬적으로 추론할 단어의 수, 현재배치의 토큰 수, 토큰 임베딩 크기)
        # 드롭아웃
        x = self.drop_shortcut(x)
        # 잔차연결
        x = x + shortcut  # Add the original input back

        ##### 피드포워드 연산 부분
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x) # 출력 : (배치크기 :한번에 병렬적으로 추론할 단어의 수, 현재배치의 토큰 수, 토큰 임베딩 크기)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 각 단어들을 임베딩 토큰으로 변경하기 위한 조회 테이블
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Sequential: 배열에 담긴 레이어를 묶어서 한번에 관리하기 위함
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # 모델입력 : (배치크기 :한번에 병렬적으로 추론할 문장의 수, 현재배치의 토큰 수)
        # 예를들어 (2:배치크기, 4: 토큰 수)이 입력인 경우 (2, 4, 임베딩 크기)로 벡터를 변환시킴
        # 토큰 임베딩 이후 포지셔널 임베딩 벡터와 합산
        # 출력 : (배치크기 :한번에 병렬적으로 추론할 단어의 수, 현재배치의 토큰 수, 토큰 임베딩 크기)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # 멀티헤드어텐션 연산 수행
        # 출력 : (배치크기 :한번에 병렬적으로 추론할 단어의 수, 현재배치의 토큰 수, 토큰 임베딩 크기)
        x = self.trf_blocks(x)

        # 마지막 정규화층 연산
        # 출력 : (배치크기 :한번에 병렬적으로 추론할 단어의 수, 현재배치의 토큰 수, 토큰 임베딩 크기)
        x = self.final_norm(x)

        # out_head = Linear(단어임베딩 크기, 단어사전 크기)
        # 각각의 단어를 선형층을 통해 단어 마다의 확률로 변환시키는 과정
        # out_head의 선형층을 거치면서 마지막 차원의 크기가 (단어임베딩 크기 -> 단어사전 크기) 로 확장된다.
        # 모델출력 : (배치크기 :한번에 병렬적으로 추론할 단어의 수, 현재배치의 토큰 수, 각 토큰마다 단어사전의 확률분포)
        logits = self.out_head(x)
        return logits


# idx : 초기 입력
# max_new_tokens : 생성할 최대 토큰
# context_size : 모델이 한 번에 볼 수 있는 문맥크기
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # max_new_tokens개 만큼의 토큰이 생성 될 때까지 반복
    for _ in range(max_new_tokens):

        # 모델이 한번에 처리 가능한 문맥 크기를 넘는 경우, 앞부분을 자르기 위한 코드
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            # 모델입력 : (배치크기 :한번에 병렬적으로 추론할 문장의 수, 현재배치의 토큰 수)
            # 모델출력 : (배치크기 :한번에 병렬적으로 추론할 문장의 수), 현재배치의 토큰 수, 각 토큰마다 단어사전의 확률분포)
            logits = model(idx_cond)

        # 마지막 단어의 확률분포만 남김
        logits = logits[:, -1, :]

        # 가장 높은 확률을 가진 단어만 남김
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 가장 높은 확률을 가진 단어를 기존 문맥에 이어붙임
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    # 차원의 0번 인덱스의 새로운 차원을 추가 ( ,4) -> (1, 4)
    # 모델은 배치로 실행되기 때문에 배치 차원을 추가하는 과정이다.
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )


    # 2차원으로 나온 결과의 0번 인덱스 차원을 제거하여 1차원으로 변환
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()