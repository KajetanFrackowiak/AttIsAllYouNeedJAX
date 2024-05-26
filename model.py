import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import math

class LayerNormalization(nn.Module):
    features: int
    eps: float = 1e-6
    
    def setup(self):
        self.alpha = self.param('alpha', nn.initializers.ones, (self.features,))
        self.bias = self.param('bias', nn.initializers.zeros, (self.features,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    d_model: int
    d_ff: int
    dropout: int
    
    def setup(self):
        self.linear_1 = nn.Dense(self.d_ff)
        self.linear_2 = nn.Dense(self.d_model)
        self.dropout_layer = nn.Dense(self.dropout)
        
    def __call__(self, x, deterministic):
        x = nn.relu(self.linear_1(x))
        x = self.dropout(x, deterministic=deterministic)
        return self.linear_2(x)
    
class InputEmbeddings(nn.Module):
    d_model: int
    vocab_size: int
    
    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.d_model)
        
    def __call__(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    d_model: int
    max_seq_len: int
    dropout: float
    
    def setup(self):
        self.dropout_layer = nn.Dropout(self.dropout)
        pe = jnp.zeros((self.max_seq_len, self.d_model))
        position = jnp.arange(0, self.max_seq_len, dtype=jnp.float32).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * - (math.log(10000.0) / self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe
        
    def __call__(self, x, deterministic):
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :]
        return self.dropout_layer(x, deterministic=deterministic)

class ResidualConnecion(nn.Module):
    features: int
    dropout: float
    
    def setup(self):
        self.norm = LayerNormalization(self.features)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def __call__(self, x, sublayer, deterministic):
        return x + self.dropout_layer(sublayer(self.norm(x)), deterministic=deterministic)

class MultiHeadAttentionBlock(nn.Module):
    d_model: int
    num_heads: int
    dropout: float
    
    def setup(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model //  self.num_heads
        self.w_q = nn.Dense(self.d_model)
        self.w_k = nn.Dense(self.d_model)
        self.w_v = nn.Dense(self.d_model)
        self.w_o = nn.Dense(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def __call__(self, q, k, v, mask, deterministic):
        batch_size = q.shape[0]
        q = self.split_heads(self.w_q(q), batch_size)
        k = self.split_heads(self.w_k(k), batch_size)
        v = self.split_heads(self.w_v(v), batch_size)
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask, deterministic)
        return self.w_o(attn_output), attn_weights
    
    def split_heads(self, x, batch_size):
        return x.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
    
    def combine_heads(self, x, batch_size):
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
    
    def scaled_dot_product_attention(self, q, k, v, mask, deterministic):
        d_k = q.shape[-1]
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
        if mask is not None:
            attn_logits = jnp.where(mask == 0, -1e9, attn_logits)
        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn_weights = self.dropout_layer(attn_weights, deterministic=deterministic)
        attn_output = jnp.matmul(attn_weights, v)
        return attn_output, attn_weights
    
class EncoderBlock(nn.Module):
    features: int
    num_heads: int
    d_ff: int
    dropout: float
    
    def setup(self):
        self.self_attention = MultiHeadAttentionBlock(self.features, self.num_heads, self.dropout)
        self.feed_forward = FeedForwardBlock(self.features, self.d_ff, self.dropout)
        self.residual_layers = [ResidualConnecion(self.features, self.dropout) for _ in range(2)]
    
    def __call__(self, x, src_mask, deterministic):
        x, _ = self.residual_layers[0](x, lambda x: self.self_attention(x, x, x, src_mask, deterministic), deterministic)
        x = self.residual_layers[1](x, self.feed_forward, deterministic)
        return x
    
class Encoder(nn.Module):
    features: int
    num_layers: int
    num_heads: int
    d_ff: int
    dropout: float
    
    def setup(self):
        self.layers = [EncoderBlock(self.features, self.num_heads, self.d_ff, self.dropout) for _ in range(self.num_layers)]
        self.norm = LayerNormalization(self.features)
        
    def __call__(self, x, mask, deterministic):
        for layer in self.layers:
            x = layer(x, mask, deterministic)
        return self.norm(x)

class DecoderBlock(nn.Module):
    features: int
    num_heads: int
    d_ff: int
    dropout: float
    
    def setup(self):
        self.self_attention = MultiHeadAttentionBlock(self.features, self.num_heads, self.dropout)
        self.cross_attention = MultiHeadAttentionBlock(self.features, self.num_heads, self.dropout)
        self.feed_forward = FeedForwardBlock(self.features, self.d_ff, self.dropout)
        self.residual_layers = [ResidualConnecion(self.features, self.dropout) for _ in range(3)]
    
    def __call__(self, x, encoder_output, src_mask, tgt_mask, deterministic):
        x, _ = self.residual_layers[0](x, lambda x: self.self_attention(x, x, x, tgt_mask, deterministic), deterministic)
        x, _ = self.residual_layers[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask, deterministic), deterministic)
        x = self.residual_layers[2](x, self.feed_forward, deterministic)
        return x

class Decoder(nn.Module):
    features: int
    num_layers: int
    num_heads: int
    d_ff: int
    dropout: float
    
    def setup(self):
        self.layers = [DecoderBlock(self.features, self.num_heads, self.d_ff, self.dropout)]
        self.norm = LayerNormalization(self.features)
        
    def __call__(self, x, encoder_output, src_mask, tgt_mask, deterministic):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask, deterministic)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    d_model: int
    vocab_size: int
    
    def setup(self):
        self.proj = nn.Dense(self.vocab_size)
        
    def __call__(self, x):
        return self.proj(x)
    
class Transformer(nn.Module):
    src_vocab_size: int
    tgt_vocab_size: int
    src_seq_len: int
    tgt_seq_len: int
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    d_ff: int = 2048
    
    def setup(self):
        self.src_embed = InputEmbeddings(self.d_model, self.src_vocab_size) 
        self.tgt_embed = InputEmbeddings(self.d_model, self.tgt_vocab_size)
        self.src_pos = PositionalEncoding(self.d_model, self.src_seq_len, self.dropout)
        self.tgt_pos = PositionalEncoding(self.d_model, self.tgt_seq_len, self.dropout)
        
        self.encoder = Encoder(self.d_model, self.num_layers, self.num_heads, self.d_ff, self.dropout) 
        self.decoder = Decoder(self.d_model, self.num_layers, self.num_heads, self.d_ff, self.dropout)
        self.projection_layer = ProjectionLayer(self.d_model, self.tgt_vocab_size)
        
    def encode(self, src, src_mask, deterministic):
        src = self.src_embed(src)
        src = self.src_pos(src, deterministic)
        return self.encoder(src, src_mask, deterministic)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask, deterministic):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt, deterministic)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask, deterministic)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def __call__(self, src, tgt, src_mask, tgt_mask, deterministic):
        encoder_output = self.encode(src, src_mask, deterministic)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask, deterministic)
        return self.project(decoder_output)
    
# Helper function to initialize the model
def build_transformer(key, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, num_layers=6, num_heads=8, dropout=0.1, d_ff=2048):
    transformer = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, src_seq_leq=src_seq_len, 
                              tgt_seq_len=tgt_seq_len, d_model=d_model, num_layers=num_layers, num_heads=num_heads, dropout=dropout, d_ff=d_ff)
    variables = transformer.init(key, jnp.ones((1, src_seq_len), dtype=jnp.int32), jnp.ones((1, tgt_seq_len), dtype=jnp.int32), jnp.ones((1, src_seq_len), dtype=jnp.int32), jnp.ones((1, tgt_seq_len), dtype=jnp.int32), deterministic=True)
    return transformer, variables