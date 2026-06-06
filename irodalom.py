import torch
import torch.nn as nn
from torch.nn import functional as F
import datetime
import os, sys
import sentencepiece as spm
from dataclasses import dataclass
import logging

def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # always reset in notebooks (Colab/IPython safe)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # prevents duplicate root logs

    return logger

logger = create_logger()

def is_colab():
    return "COLAB_GPU" in os.environ


@dataclass
class Config:
    root_dir: str
    text_generator_dir: str
    batch_size: int

    tokenizer_name: str = "hu_tokenizer"

    block_size: int = 512
    n_embed: int = 512
    n_head: int = 8
    n_layer: int = 12
    dropout: float = 0.2

    max_lr: float = 1e-4
    warmup_steps: int = 2000

    @property
    def input_txt(self):
        return os.path.join(self.text_generator_dir, "RegenyKorpusz.txt")

    @property
    def model_file(self):
        return os.path.join(self.text_generator_dir, "regeny_gpt.pt")

    @property
    def tokenizer_prefix(self):
        return os.path.join(self.text_generator_dir, self.tokenizer_name)

    @property
    def token_cache_file(self):
        return os.path.join(self.text_generator_dir, "token_cache.pt")


def create_config() -> Config:
    if is_colab():
        if not os.path.ismount("/content/drive"):
            from google.colab import drive

            drive.mount("/content/drive")

        root_dir = "/content/drive/MyDrive/"
        text_generator_dir = root_dir + "TextGenerator/"
        batch_size = 16
    else:
        root_dir = "./"
        text_generator_dir = root_dir
        batch_size = 2

    return Config(
        root_dir=root_dir, text_generator_dir=text_generator_dir, batch_size=batch_size
    )


config = create_config()

sys.path.append(config.root_dir)
import my_common as my


class SentencePieceTokenizer:
    def __init__(self, max_len=32):
        logger.info("Init sentece tokenizer")
        self.vocab_size = 8000
        self.sp = spm.SentencePieceProcessor()
        self.max_len = max_len

        self.model_path = f"{config.tokenizer_prefix}.model"

        if not os.path.exists(self.model_path):
            self.train_tokenizer(config.input_txt, f"{config.tokenizer_prefix}")
        else:
            logger.info(f"Load tokenizer model from {self.model_path}")

        self.sp.load(self.model_path)

        # Standard BERT-style special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.cls_id = self.sp.bos_id()
        self.sep_id = self.sp.eos_id()
        self.mask_id = self.sp.piece_to_id("[MASK]")
        if self.mask_id == self.unk_id:
            raise ValueError("[MASK] not found in vocab")

    def train_tokenizer(self, input_path, model_prefix):
        logger.info("Start tokenizer training")
        spm.SentencePieceTrainer.train(
            input=input_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=["[CLS]", "[SEP]", "[MASK]", "[NL]"],
        )
        logger.info("Tokenizer training finished")

    def preprocess_text(self, text):
        text = text.replace("\n", " [NL] ")
        return text

    def encode(self, text):
        text = self.preprocess_text(text)
        tokens = self.sp.encode(text, out_type=int)
        return tokens

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace("[NL]", "\n")
        return text


class BatchLoader:
    def __init__(self):
        self.tokenizer = SentencePieceTokenizer()
        token_cache_file = config.token_cache_file
        if os.path.exists(token_cache_file):
            logger.info(f"Loading tokens from {token_cache_file}")
            data = torch.load(token_cache_file)
        else:
            chunk_size = 1024 * 1024
            logger.info("Tokenizing text")
            data = []
            with open(config.input_txt, "r", encoding="utf-8") as f:
                while chunk := f.read(chunk_size):
                    data.extend(self.tokenizer.encode(chunk))
            data = torch.tensor(data, dtype=torch.long)
            logger.info(f"Save tokens to {token_cache_file}")
            torch.save(data, token_cache_file)
        # Train and test splits
        logger.info(f"data lenght: {len(data)}")
        n = int(0.95 * len(data))  # first 95% will be train, rest validation
        self.train_data = data[:n]
        self.validation_data = data[n:]

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.validation_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i : i + config.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
        x, y = x.to(my.DEVICE), y.to(my.DEVICE)
        return x, y


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            ),
            persistent=False,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(config.n_embed, n_head=config.n_head)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embed)  # final layer norm
        self.lm_head = nn.Linear(config.n_embed, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=my.DEVICE)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class GPTTrainer:
    def __init__(self):
        self.batch_loader = BatchLoader()
        vocab_size = self.batch_loader.tokenizer.vocab_size
        self.model = GPTLanguageModel(vocab_size).to(my.DEVICE)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), betas=(0.9, 0.95), weight_decay=0.1
        )

        self.start_step = 0

        if os.path.exists(config.model_file):
            checkpoint = torch.load(config.model_file, map_location=my.DEVICE)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_step = checkpoint["step"]

        # print parameters and layers of the model
        my.print_parameter_summary(self.model)

    def get_lr(self, step):
        # warmup
        if step < config.warmup_steps:
            return config.max_lr * step / config.warmup_steps
        return config.max_lr

    def generate_from_model(self):
        context = torch.zeros((1, 1), dtype=torch.long, device=my.DEVICE)
        logger.info(
            self.batch_loader.tokenizer.decode(
                self.model.generate(context, max_new_tokens=config.block_size)[
                    0
                ].tolist()
            )
        )

    @torch.no_grad()
    def estimate_loss(self):
        eval_iters = 10  # multiplied with the batch size
        out = {}
        self.model.eval()
        for split in ["train", "validation"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.batch_loader.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, step):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        torch.save(checkpoint, config.model_file)

    def train(self):
        max_iters = 120000

        logger.info("Start training")
        logger.info(f"Batch size: {config.batch_size}")
        for step in range(self.start_step, max_iters):
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            # sample a batch of data
            xb, yb = self.batch_loader.get_batch("train")
            # evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if not step % (10):
                logger.info(f"current time: {datetime.datetime.now()}")
                logger.info(f"current step: {step}")
                self.generate_from_model()
                losses = self.estimate_loss()
                for k, v in losses.items():
                    logger.info(f"{k + ' loss':<18}: {v:.4f}")
                self.save_checkpoint(step)


def train():
    trainer = GPTTrainer()
    trainer.train()


if __name__ == "__main__":
    train()
