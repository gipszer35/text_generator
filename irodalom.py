import torch
import torch.nn as nn
from torch.nn import functional as F
import datetime
import os, sys
import sentencepiece as spm
from dataclasses import dataclass
import logging
from enum import Enum
import random


def is_colab():
    return "COLAB_GPU" in os.environ


class DataSource(Enum):
    WIKI = "wiki"
    NOVEL = "novel"


@dataclass
class Config:
    root_dir: str
    text_generator_dir: str
    batch_size: int

    block_size: int = 512
    n_embed: int = 512
    n_head: int = 8
    n_layer: int = 14
    dropout: float = 0.0
    # At start the max_lr=5e-4 is good for small models. But with warmup.
    max_lr: float = 1e-4
    warmup_steps: int = 2000

    @property
    def out_dir(self):
        return os.path.join(self.text_generator_dir, "irodalom_out")

    @property
    def novels_txt(self):
        return os.path.join(self.text_generator_dir, "regenykorpusz.txt")

    @property
    def wiki_txt(self):
        return os.path.join(self.text_generator_dir, "wiki_hu.txt")

    @property
    def model_file(self):
        return os.path.join(self.out_dir, "irodalom_gpt_2.pt")

    @property
    def tokenizer_prefix(self):
        return os.path.join(self.out_dir, "hu_tokenizer")

    @property
    def novel_token_cache(self):
        return os.path.join(self.out_dir, "novel_token_cache.pt")

    @property
    def wiki_token_cache(self):
        return os.path.join(self.out_dir, "wiki_token_cache.pt")


def create_config() -> Config:
    if is_colab():
        if not os.path.ismount("/content/drive"):
            from google.colab import drive

            drive.mount("/content/drive")

        root_dir = "/content/drive/MyDrive/"
        text_generator_dir = root_dir + "TextGenerator/"
        batch_size = 40
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

logger = my.create_logger()


class SentencePieceTokenizer:
    def __init__(self, max_len=256):
        logger.info("Initializing SentencePiece tokenizer")
        self.special_tokens = ["[MASK]", "[NL]", "[WIKI]", "[NOVEL]"]

        self.vocab_size = 8000
        self.sp = spm.SentencePieceProcessor()
        self.max_len = max_len

        self.tokenizer_model_path = f"{config.tokenizer_prefix}.model"

        if not os.path.exists(self.tokenizer_model_path):
            self.train_tokenizer(f"{config.tokenizer_prefix}")
        else:
            logger.info(f"Loading tokenizer model from {self.tokenizer_model_path}")

        self.sp.load(self.tokenizer_model_path)

        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        self.special_ids = {}

        for token in self.special_tokens:
            token_id = self.sp.piece_to_id(token)
            if token_id == self.unk_id:
                raise ValueError(
                    f"Critical Error: Special token {token} not found in vocab!"
                )
            self.special_ids[token] = token_id

        self.mask_id = self.special_ids["[MASK]"]
        self.nl_id = self.special_ids["[NL]"]
        self.wiki_id = self.special_ids["[WIKI]"]
        self.novel_id = self.special_ids["[NOVEL]"]

        logger.info(
            f"Tokenizer loaded successfully! Active Vocab Size: {self.sp.get_piece_size()}"
        )
        logger.info(self)

    def __str__(self):
        return (
            "Tokenizer IDs:\n"
            f"  pad_id   = {self.pad_id}\n"
            f"  unk_id   = {self.unk_id}\n"
            f"  bos_id   = {self.bos_id}\n"
            f"  eos_id   = {self.eos_id}\n"
            f"  mask_id  = {self.mask_id}\n"
            f"  nl_id    = {self.nl_id}\n"
            f"  wiki_id  = {self.wiki_id}\n"
            f"  novel_id = {self.novel_id}"
        )

    def train_tokenizer(self, model_prefix):
        logger.info("Starting tokenizer training pipeline")
        spm.SentencePieceTrainer.train(
            input=f"{config.novels_txt},{config.wiki_txt}",
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=self.special_tokens,
        )
        logger.info("Tokenizer training process finished successfully")

    def encode(self, text: str):
        text = text.replace("\n", " [NL] ")
        tokens = self.sp.encode_as_ids(text)

        return tokens

    def decode(self, tokens):
        tokens = list(tokens)

        if tokens and tokens[0] in (self.wiki_id, self.novel_id, self.bos_id):
            tokens = tokens[1:]

        if tokens and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]

        text = self.sp.decode_ids(tokens)
        return text.replace("[NL]", "\n")


class BatchLoader:
    def __init__(self, tokenizer, token_cache_file, text_file):
        self.tokenizer = tokenizer
        if os.path.exists(token_cache_file):
            logger.info(f"Loading tokens from {token_cache_file}")
            data = torch.load(token_cache_file)
        else:
            chunk_size = 1024 * 1024
            logger.info("Tokenizing text")
            data = []
            with open(text_file, "r", encoding="utf-8") as f:
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


class MultiBatchLoader:
    def __init__(self):
        logger.info(
            "Initializing MultiBatchLoader with synchronized X and Y conditioning"
        )
        self.tokenizer = SentencePieceTokenizer()
        self.novel_batch_loader = BatchLoader(
            self.tokenizer, config.novel_token_cache, config.novels_txt
        )
        self.wiki_batch_loader = BatchLoader(
            self.tokenizer, config.wiki_token_cache, config.wiki_txt
        )

    def get_batch(self, split, source: DataSource):
        if source == DataSource.WIKI:
            x, y = self.wiki_batch_loader.get_batch(split)
            style_token = self.tokenizer.wiki_id
        elif source == DataSource.NOVEL:
            x, y = self.novel_batch_loader.get_batch(split)
            style_token = self.tokenizer.novel_id
        else:
            raise ValueError(f"Invalid DataSource: {source}")

        # Clone both tensors to prevent PyTorch in-place autograd errors
        x = x.clone()
        y = y.clone()

        x[:, 0] = style_token

        return x, y


class EfficientAttention(nn.Module):
    def __init__(self, n_embd=512, n_head=8):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head  # 512 // 8 = 64 channels per head

        # Combined projections for all 8 heads into single, highly optimized matrix operations
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.key_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.val_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Final output projection layer
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # B: Batch size (16), T: Sequence length / Block size (512), C: Embedding dimension (512)
        B, T, C = x.shape

        # 1. Project inputs and reshape to isolate the attention heads
        # Tensor transformation flow: [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.val_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 2. Execute PyTorch's native Scaled Dot-Product Attention (SDPA)
        # This triggers FlashAttention kernels under the hood.
        # It is mathematically identical to your old loop but prevents the massive VRAM overhead.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=config.dropout if self.training else 0.0
        )

        # 3. Concatenate all attention heads back into a single feature tensor
        # Tensor transformation flow: [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 4. Apply final linear projection mapping
        out = self.proj(out)
        # 5. And a final dropout
        out = self.dropout(out)
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
        self.sa = EfficientAttention(n_embd, n_head)
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
        self.multi_batch_loader = MultiBatchLoader()
        vocab_size = self.multi_batch_loader.tokenizer.vocab_size
        self.model = GPTLanguageModel(vocab_size).to(my.DEVICE)
        self.set_dropout_called = False

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), betas=(0.9, 0.95), weight_decay=0.1, fused=True
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
        self.model.eval()

        source = self.get_random_source()

        if source == DataSource.WIKI:
            start_token_id = self.multi_batch_loader.tokenizer.wiki_id
            style_label = "[WIKI MODE]"
        else:
            start_token_id = self.multi_batch_loader.tokenizer.novel_id
            style_label = "[NOVEL MODE]"

        print(f"\nGenerating in {style_label}")
        context = torch.tensor([[start_token_id]], dtype=torch.long, device=my.DEVICE)

        generated_tokens = self.model.generate(
            context, max_new_tokens=config.block_size
        )[0].tolist()
        decoded_text = self.multi_batch_loader.tokenizer.decode(generated_tokens)

        print("\n", decoded_text, "\n")
        self.model.train()

    def get_random_source(self, novel_ratio: float = 0.5) -> DataSource:
        if random.random() < novel_ratio:
            return DataSource.NOVEL
        return DataSource.WIKI

    @torch.no_grad()
    def estimate_loss(self):
        eval_iters = 10  # multiplied with the batch size
        out = {}
        self.model.eval()
        for split in ["train", "validation"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.multi_batch_loader.get_batch(
                    split, self.get_random_source(0.5)
                )
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
        temp_model_file = f"{config.model_file}.new"
        torch.save(checkpoint, temp_model_file)
        os.replace(temp_model_file, config.model_file)

    def set_dropout(self):
        if not self.set_dropout_called:
            self.set_dropout_called = True
            for m in self.model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = config.dropout

    def train(self):
        max_iters = 120000

        logger.info("Start training")
        logger.info(f"Batch size: {config.batch_size}")

        accum_steps = 10

        self.optimizer.zero_grad(set_to_none=True)
        for step in range(self.start_step, max_iters):
            self.set_dropout()

            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            xb, yb = self.multi_batch_loader.get_batch(
                "train", self.get_random_source()
            )

            logits, loss = self.model(xb, yb)

            # Scale loss so gradients match a larger batch
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if not step % (50):
                logger.info(f"Time:           {datetime.datetime.now()}")
                logger.info(f"Epoch:          {step}")
                logger.info(f"Max lr:         {config.max_lr}")
                logger.info(f"get_lr:         {self.get_lr(step)}")
                logger.info(f"Batch size:     {config.batch_size}")
                logger.info(f"Backward bsize: {config.batch_size * accum_steps}")
                logger.info(f"Dropout:        {config.dropout}")
                losses = self.estimate_loss()
                for k, v in losses.items():
                    logger.info(f"{k + ' loss':<18}: {v:.4f}")
                self.generate_from_model()
                logger.info(f"Save checkpoint: {config.model_file}")
                self.save_checkpoint(step)
                logger.info(f"Checkpoint saved")


def ensure_dir(path):
    if os.path.exists(path):
        logger.info(f"Directory already exists: {path}")
    else:
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


def train():
    ensure_dir(config.out_dir)
    trainer = GPTTrainer()
    trainer.train()


if __name__ == "__main__":
    train()
