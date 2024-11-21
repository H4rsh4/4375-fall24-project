import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import math
import copy
import csv
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the clones utility
def clone_module(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Define the Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Define Sublayer Connection
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Define the Multi-Headed Attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clone_module(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # For multi-head attention
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# Define Position-wise Feedforward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# Define Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Define Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_module(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone_module(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Define Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_module(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone_module(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# Define Encoder-Decoder
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# Full Transformer Model Construction
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, attn, ff, dropout), N),
        Decoder(DecoderLayer(d_model, attn, attn, ff, dropout), N),
        nn.Sequential(nn.Embedding(src_vocab, d_model), position),
        nn.Sequential(nn.Embedding(tgt_vocab, d_model), position),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# Logger Class to Save Experiment Details
class ExperimentLogger:
    def __init__(self, filename="experiment_log.csv"):
        self.filename = filename
        # Write header if the file doesn't exist
        try:
            with open(self.filename, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Experiment Number",
                    "Parameters Chosen",
                    "Training Accuracy",
                    "Validation Accuracy",
                    "Training Loss",
                    "Validation Loss",
                    "Run Time"
                ])
        except FileExistsError:
            pass  # File already exists, no need to create a new header

    def log(self, experiment_number, parameters, train_acc, val_acc, train_loss, val_loss, runtime):
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                experiment_number,
                parameters,
                train_acc,
                val_acc,
                train_loss,
                val_loss,
                runtime
            ])
        print(f"Experiment {experiment_number} logged.")

# Transformer Model and Training Code (shortened for simplicity)
def train_model_with_logging(experiment_number, params, logger):
    # Load Dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
    val_dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Corrected Preprocessing Function
    def preprocess_data(batch):
        inputs = tokenizer(batch['article'], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        targets = tokenizer(batch['highlights'], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
        return {"input_ids": inputs['input_ids'], "target_ids": targets['input_ids']}
    
    dataset = dataset.map(preprocess_data, batched=True)
    val_dataset = val_dataset.map(preprocess_data, batched=True)

    # Adjust DataLoader
    train_loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=lambda x: (torch.stack([d['input_ids'] for d in x]), torch.stack([d['target_ids'] for d in x])))
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, collate_fn=lambda x: (torch.stack([d['input_ids'] for d in x]), torch.stack([d['target_ids'] for d in x])))

    # Model
    src_vocab = tokenizer.vocab_size
    tgt_vocab = tokenizer.vocab_size
    model = make_model(src_vocab, tgt_vocab, N=params["num_layers"], d_model=params["d_model"], d_ff=params["d_ff"])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training
    start_time = time.time()
    model.train()
    total_train_loss = 0
    for epoch in range(params["epochs"]):
        for src, tgt in train_loader:
            src, tgt = src.cuda(), tgt.cuda()
            optimizer.zero_grad()
            src_mask = (src != 0).unsqueeze(-2).to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = (tgt_input != 0).unsqueeze(-2).to(device)
            out = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(out.view(-1, out.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
    
    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.cuda(), tgt.cuda()
            src_mask = (src != 0).unsqueeze(-2)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = (tgt_input != 0).unsqueeze(-2).to(device)
            out = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(out.view(-1, out.size(-1)), tgt_output.view(-1))
            total_val_loss += loss.item()

    runtime = time.time() - start_time

    # Calculate Metrics (Dummy Metrics as Placeholders)
    train_acc = 100 - total_train_loss / len(train_loader)
    val_acc = 100 - total_val_loss / len(val_loader)

    # Log Results
    logger.log(
        experiment_number=experiment_number,
        parameters=str(params),
        train_acc=train_acc,
        val_acc=val_acc,
        train_loss=total_train_loss / len(train_loader),
        val_loss=total_val_loss / len(val_loader),
        runtime=runtime
    )

    # Save Model
    model_path = f"transformer_experiment_{experiment_number}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

# Example Usage of Training with Logging
if __name__ == "__main__":
    logger = ExperimentLogger()

    # Define Experiment Parameters
    experiment_params = {
        "num_layers": 6,
        "d_model": 512,
        "d_ff": 2048,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 3
    }

    train_model_with_logging(1, experiment_params, logger)