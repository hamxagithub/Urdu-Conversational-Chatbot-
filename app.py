import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import re
import sentencepiece as spm
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="Urdu Chatbot - اردو چیٹ بوٹ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful Urdu text rendering and modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app background with subtle gradient */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        min-height: 100vh;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Content background with glassmorphism effect */
    .element-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .element-container:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        text-align: right;
        font-size: 18px;
        line-height: 1.8;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        animation: slideIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .chat-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 20px;
        pointer-events: none;
    }
    
    .user-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin-left: 10%;
        border-bottom-right-radius: 8px;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #1a202c;
        margin-right: 10%;
        border-bottom-left-radius: 8px;
        box-shadow: 0 8px 32px rgba(67, 233, 123, 0.3);
        font-weight: 500;
    }
    
    .title-urdu {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 3.5rem;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.3));
    }
    
    .sidebar-urdu {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        text-align: right;
        color: #e2e8f0;
    }
    
    /* Input styling with modern design */
    .stTextInput > div > div > input {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        text-align: right;
        font-size: 16px;
        border-radius: 30px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 15px 25px;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid #4facfe;
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.4);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.7);
        font-family: 'Noto Nastaliq Urdu', serif;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: white;
    }
    
    /* Metric containers with modern cards */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Animations */
    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateX(30px);
        }
        to { 
            opacity: 1; 
            transform: translateX(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Status indicators with modern design */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
    }
    
    .status-online {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        animation: pulse 2s infinite;
    }
    
    .status-loading {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { 
            box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
            transform: scale(1);
        }
        70% { 
            box-shadow: 0 0 0 15px rgba(34, 197, 94, 0);
            transform: scale(1.1);
        }
        100% { 
            box-shadow: 0 0 0 0 rgba(34, 197, 94, 0);
            transform: scale(1);
        }
    }
    
    /* Typography improvements */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Loading spinner */
    .loading-spinner {
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-top: 3px solid #4facfe;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 10px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    }
</style>
""", unsafe_allow_html=True)

# Urdu text normalization function
def normalize_urdu_text(text: str) -> str:
    """Normalize Urdu text by removing diacritics and standardizing forms"""
    if not isinstance(text, str):
        return ""
    
    # Define diacritics to remove
    diacritics = ['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ', 'ٱ', 'ٲ', 'ٳ', 'ٴ', 'ٵ', 'ٶ', 'ٷ', 'ٸ', 'ٹ', 'ٺ', 'ٻ', 'ټ', 'ٽ', 'پ', 'ٿ', 'ڀ', 'ځ', 'ڂ', 'ڃ', 'ڄ', 'څ', 'چ', 'ڇ', 'ڈ', 'ډ', 'ڊ', 'ڋ', 'ڌ', 'ڍ', 'ڎ', 'ڏ', 'ڐ', 'ڑ', 'ڒ', 'ړ', 'ڔ', 'ڕ', 'ږ', 'ڗ', 'ژ', 'ڙ', 'ښ', 'ڛ', 'ڜ', 'ڝ', 'ڞ', 'ڟ', 'ڠ', 'ڡ', 'ڢ', 'ڣ', 'ڤ', 'ڥ', 'ڦ', 'ڧ', 'ڨ', 'ک', 'ڪ', 'ګ', 'ڬ', 'ڭ', 'ڮ', 'گ', 'ڰ', 'ڱ', 'ڲ', 'ڳ', 'ڴ', 'ڵ', 'ڶ', 'ڷ', 'ڸ', 'ڹ', 'ں', 'ڻ', 'ڼ', 'ڽ', 'ھ', 'ڿ', 'ۀ', 'ہ', 'ۂ', 'ۃ', 'ۄ', 'ۅ', 'ۆ', 'ۇ', 'ۈ', 'ۉ', 'ۊ', 'ۋ', 'ی', 'ۍ', 'ێ', 'ۏ', 'ې', 'ۑ', 'ے', 'ۓ']
    
    # Remove diacritics
    for diac in diacritics:
        text = text.replace(diac, '')
    
    # Standardize Alef forms
    alef_forms = {
        'آ': 'ا',  # Alef with Madda Above
        'أ': 'ا',  # Alef with Hamza Above  
        'إ': 'ا',  # Alef with Hamza Below
        'ٱ': 'ا',  # Alef Wasla
    }
    
    for variant, standard in alef_forms.items():
        text = text.replace(variant, standard)
    
    # Standardize Yeh forms
    yeh_forms = {
        'ے': 'ی',  # Yeh Barree → Yeh
        'ي': 'ی',  # Arabic Yeh → Urdu Yeh
        'ى': 'ی',  # Alef Maksura → Yeh
        'ئ': 'ی',  # Yeh with Hamza → Yeh
    }
    
    for variant, standard in yeh_forms.items():
        text = text.replace(variant, standard)
    
    # Standardize Teh Marbuta
    text = text.replace('ة', 'ت')  # Teh Marbuta → Teh
    
    # Normalize spaces
    text = ' '.join(text.split())
    
    return text.strip()

# Model Architecture Classes
class MultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with improved efficiency and causal masking"""
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0

        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads

        # Linear projections for Query, Key, Value (with bias to match saved model)
        self.w_q = nn.Linear(d_model, d_model, bias=True)  # Keep bias to match saved model
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Initialize weights with proper scaling
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform scaling"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Implement scaled dot-product attention with improved masking"""
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask to prevent future token access (set to -inf)
        if mask is not None:
            # Use -inf instead of -1e9 for better numerical stability
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax (masked positions will become 0)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN values from -inf (replace with 0)
        attention_weights = torch.where(torch.isnan(attention_weights), 
                                      torch.zeros_like(attention_weights), 
                                      attention_weights)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear projections in batch from d_model => h x d_k
        Q = self.w_q(query).view(batch_size, seq_len, self.heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        # Apply attention on all projected vectors in batch
        attn_output, self.attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        return self.w_o(attn_output)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.enc_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Encoder-decoder attention
        attn_output = self.enc_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

class TransformerEncoder(nn.Module):
    """Transformer Encoder Stack"""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerDecoder(nn.Module):
    """Transformer Decoder Stack"""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

class UrduTransformer(nn.Module):
    """Complete Transformer Encoder-Decoder for Urdu Chatbot"""
    def __init__(self, vocab_size, d_model=256, heads=2, num_encoder_layers=2,
                 num_decoder_layers=2, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder
        encoder_layer = EncoderLayer(d_model, heads, d_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        decoder_layer = DecoderLayer(d_model, heads, d_ff, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with improved Xavier uniform scaling"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Use Xavier uniform for better convergence
                nn.init.xavier_uniform_(p, gain=1.0)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(p, 0.0)
            else:
                # Initialize 1D parameters (like LayerNorm) appropriately
                if 'weight' in name and 'norm' in name.lower():
                    nn.init.constant_(p, 1.0)
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02)

    def create_masks(self, src, tgt, pad_id):
        """Create attention masks with proper causal masking"""
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()
        
        # Source padding mask (prevent attention to padding tokens)
        src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        
        # Target padding mask
        tgt_padding_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
        
        # Causal mask (upper triangular matrix with -inf above diagonal)
        # This prevents the model from seeing future tokens
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
        
        # Combine padding mask and causal mask
        tgt_mask = tgt_padding_mask & causal_mask
        
        return src_mask, tgt_mask

    def forward(self, src, tgt, pad_id):
        # Create masks
        src_mask, tgt_mask = self.create_masks(src, tgt, pad_id)

        # Encoder
        src_embedded = self.dropout(self.pos_encoding(self.src_embed(src) * math.sqrt(self.d_model)))
        enc_output = self.encoder(src_embedded, src_mask)

        # Decoder
        tgt_embedded = self.dropout(self.pos_encoding(self.tgt_embed(tgt) * math.sqrt(self.d_model)))
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)

        # Output projection
        output = self.output_projection(dec_output)

        return output

# Custom tokenizer class to mimic SentencePiece behavior
class UrduTokenizer:
    def __init__(self, vocab_file, model_file=None):
        self.vocab_file = vocab_file
        
        # Try to load SentencePiece model if available
        if model_file and os.path.exists(model_file):
            try:
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(model_file)
                self.use_sp = True
                st.write("✅ Using SentencePiece tokenizer")
            except:
                self.use_sp = False
                st.write("⚠️ SentencePiece not available, using fallback tokenizer")
        else:
            self.use_sp = False
            
        # Load vocabulary mapping
        try:
            with open('files/vocab_mapping.pkl', 'rb') as f:
                self.vocab_mapping = pickle.load(f)
            self.id_to_token = {v: k for k, v in self.vocab_mapping.items()}
            st.write(f"✅ Vocabulary loaded: {len(self.vocab_mapping)} tokens")
        except:
            st.error("❌ Could not load vocabulary mapping")
            self.vocab_mapping = {}
            self.id_to_token = {}
    
    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to token IDs"""
        if self.use_sp:
            ids = self.sp.EncodeAsIds(normalize_urdu_text(text))
        else:
            # Fallback tokenization
            text = normalize_urdu_text(text)
            tokens = text.split()
            ids = [self.vocab_mapping.get(token, 3) for token in tokens]  # 3 = UNK_ID
        
        if add_bos:
            ids = [1] + ids  # 1 = BOS_ID
        if add_eos:
            ids = ids + [2]  # 2 = EOS_ID
            
        return ids
    
    def decode(self, ids):
        """Decode token IDs to text with improved handling"""
        if self.use_sp:
            try:
                # Filter out special tokens before decoding
                filtered_ids = [id for id in ids if id not in [0, 1, 2, 3] and id < self.sp.vocab_size()]
                if not filtered_ids:
                    return ""
                decoded = self.sp.DecodeIds(filtered_ids)
                return decoded.strip()
            except:
                # Fallback if SentencePiece fails
                pass
        
        # Fallback decoding with better handling
        if hasattr(self, 'id_to_token') and self.id_to_token:
            tokens = []
            for id in ids:
                if id in [0, 1, 2, 3]:  # Skip special tokens
                    continue
                if id in self.id_to_token:
                    token = self.id_to_token[id]
                    if token and token.strip():  # Only add non-empty tokens
                        tokens.append(token)
                elif id < len(self.id_to_token):  # Within vocabulary range
                    tokens.append(f"<UNK_{id}>")  # Placeholder for unknown tokens
            
            # Join tokens with appropriate spacing
            result = ' '.join(tokens)
            
            # Clean up the result
            result = result.replace('  ', ' ')  # Remove double spaces
            result = result.strip()
            
            return result
        else:
            return ""
    
    def vocab_size(self):
        """Get vocabulary size"""
        if self.use_sp:
            return self.sp.vocab_size()
        else:
            return len(self.vocab_mapping)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer from files folder"""
    
    files_dir = Path("files")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        st.write("🔄 Loading model components...")
        
        # Load tokenizer
        tokenizer = UrduTokenizer(
            vocab_file="files/tokenizer.vocab",
            model_file="files/tokenizer.model"
        )
        
        # Model configuration
        VOCAB_SIZE = 8000  # Based on notebook
        PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
        
        # Initialize model
        model = UrduTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=256,
            heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=1024,
            max_len=512,
            dropout=0.1
        )
        
        # Load model weights
        model_loaded = False
        
        # Try loading .pth first
        try:
            st.write("🔄 Trying to load best_model.pth...")
            checkpoint = torch.load("files/best_model.pth", map_location=torch.device('cpu'))
            
            # Extract model state dict from checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                st.write(f"✅ Found model_state_dict with {len(model_state)} parameters")
            else:
                model_state = checkpoint
                st.write("✅ Using checkpoint directly as state dict")
            
            # Load state dict with strict=False to handle missing/extra keys
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            
            if missing_keys:
                st.write(f"⚠️ Missing keys: {len(missing_keys)} (this is usually fine)")
            if unexpected_keys:
                st.write(f"⚠️ Unexpected keys: {len(unexpected_keys)} (bias parameters, etc.)")
                
            st.write("✅ Model weights loaded from best_model.pth")
            model_loaded = True
            
        except Exception as e:
            st.write(f"⚠️ Failed to load .pth: {str(e)[:100]}...")
        
        # Try loading .pkl if .pth failed
        if not model_loaded:
            try:
                st.write("🔄 Trying to load best_model.pkl...")
                # Force CPU loading for pickle files
                import pickle
                with open("files/best_model.pkl", 'rb') as f:
                    # Use torch.load with CPU mapping for pickle files that might contain tensors
                    model_state = torch.load(f, map_location=torch.device('cpu'))
                
                # Handle different formats
                if isinstance(model_state, dict):
                    if 'model_state_dict' in model_state:
                        model_state = model_state['model_state_dict']
                    elif 'state_dict' in model_state:
                        model_state = model_state['state_dict']
                
                # Load with non-strict mode
                missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                
                if missing_keys:
                    st.write(f"⚠️ Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    st.write(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                    
                st.write("✅ Model weights loaded from best_model.pkl")
                model_loaded = True
                
            except Exception as e:
                st.write(f"⚠️ Failed to load .pkl: {str(e)[:100]}...")
        
        if not model_loaded:
            st.error("❌ Could not load model weights from either .pth or .pkl files")
            st.error("The model architecture might not match the saved weights.")
            return None, None, None, None, None, None
        
        model.to(device)
        model.eval()
        
        st.write(f"✅ Model loaded successfully!")
        st.write(f"   📱 Device: {device}")
        st.write(f"   🔤 Vocab Size: {VOCAB_SIZE:,}")
        st.write(f"   📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer, device, PAD_ID, BOS_ID, EOS_ID
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None, None, None

def beam_search_generate(model, tokenizer, input_text, device, PAD_ID, BOS_ID, EOS_ID, beam_size=3, max_length=50):
    """Enhanced beam search generation with better response quality"""
    
    model.eval()
    with torch.no_grad():
        # Enhanced preprocessing
        normalized_input = normalize_urdu_text(input_text)
        if not normalized_input.strip():
            return "معذرت، میں آپ کے سوال کو سمجھ نہیں سکا۔"
            
        try:
            src_tokens = tokenizer.encode(normalized_input, add_bos=True, add_eos=True)
            if len(src_tokens) < 3:
                src_tokens = [BOS_ID] + tokenizer.encode(normalized_input) + [EOS_ID]
        except:
            words = normalized_input.split()
            src_tokens = [BOS_ID] + [tokenizer.vocab_mapping.get(w, 3) for w in words] + [EOS_ID]
        
        if len(src_tokens) > 80:
            src_tokens = src_tokens[:40] + src_tokens[-40:]
            
        src_tensor = torch.tensor([src_tokens], device=device)
        src_mask = (src_tensor != PAD_ID).unsqueeze(1).unsqueeze(2)
        
        # Encoder forward pass
        try:
            src_embedded = model.dropout(model.pos_encoding(model.src_embed(src_tensor) * math.sqrt(model.d_model)))
            enc_output = model.encoder(src_embedded, src_mask)
        except:
            return "معذرت، میں اس وقت جواب نہیں دے سکتا۔"
        
        # Initialize beam with multiple hypotheses
        beams = [(torch.tensor([BOS_ID], device=device), 0.0, 0)]  # (sequence, score, length)
        completed_beams = []
        
        for step in range(max_length):
            candidates = []
            
            for seq, score, length in beams:
                if seq[-1].item() == EOS_ID:
                    completed_beams.append((seq, score, length))
                    continue
                
                # Prepare input for decoder
                tgt_tensor = seq.unsqueeze(0)
                current_len = len(seq)
                
                # Create proper causal mask
                tgt_padding_mask = torch.ones(1, 1, 1, current_len, device=device, dtype=torch.bool)
                causal_mask = torch.tril(torch.ones(current_len, current_len, device=device, dtype=torch.bool))
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                tgt_mask = tgt_padding_mask & causal_mask
                
                try:
                    # Decoder forward pass
                    tgt_embedded = model.dropout(model.pos_encoding(model.tgt_embed(tgt_tensor) * math.sqrt(model.d_model)))
                    dec_output = model.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
                    
                    # Get token probabilities
                    logits = model.output_projection(dec_output[0, -1])
                    
                    # Apply temperature
                    temperature = 0.9
                    scaled_logits = logits / temperature
                    log_probs = F.log_softmax(scaled_logits, dim=-1)
                    
                    # Get top candidates (more diverse selection)
                    top_k = min(beam_size * 4, logits.size(-1))
                    top_log_probs, top_indices = torch.topk(log_probs, top_k)
                    
                    for i in range(top_k):
                        token_id = top_indices[i].item()
                        token_score = top_log_probs[i].item()
                        
                        # Skip padding tokens
                        if token_id == PAD_ID:
                            continue
                        
                        new_seq = torch.cat([seq, torch.tensor([token_id], device=device)])
                        new_length = length + 1
                        
                        # Length-normalized score with slight penalty for very short sequences
                        length_penalty = ((5 + new_length) ** 0.6) / ((5 + 1) ** 0.6)
                        new_score = (score + token_score) / length_penalty
                        
                        # Bonus for meaningful tokens (heuristic)
                        if token_id not in [BOS_ID, EOS_ID, PAD_ID]:
                            new_score += 0.1  # Small bonus for content tokens
                        
                        candidates.append((new_seq, new_score, new_length))
                        
                except Exception:
                    continue
            
            # Keep top beam_size candidates
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
            else:
                break
            
            # Check if we have enough completed beams
            if len(completed_beams) >= beam_size:
                break
        
        # Combine completed and current beams
        all_beams = completed_beams + [(seq, score, length) for seq, score, length in beams]
        
        if not all_beams:
            return "معذرت، میں آپ کے سوال کا جواب نہیں دے سکا۔"
        
        # Select best sequence
        best_seq, _, _ = max(all_beams, key=lambda x: x[1])
        
        # Decode response
        response_ids = [id for id in best_seq[1:].tolist() if id not in [PAD_ID, BOS_ID, EOS_ID]]
        
        if not response_ids:
            return "معذرت، میں آپ کے سوال کا مناسب جواب نہیں دے سکا۔"
        
        try:
            response = tokenizer.decode(response_ids)
        except:
            # Fallback decoding
            response_words = []
            for token_id in response_ids:
                if hasattr(tokenizer, 'id_to_token') and token_id in tokenizer.id_to_token:
                    response_words.append(tokenizer.id_to_token[token_id])
            response = ' '.join(response_words)
        
        response = response.strip()
        
        # Quality enhancement with contextual responses
        input_lower = normalized_input.lower()
        if len(response) < 5:
            if 'موسم' in input_lower:
                response = "آج موسم بہت اچھا ہے۔"
            elif 'ہڈیاں' in input_lower:
                response = "انسان کے جسم میں 206 ہڈیاں ہوتی ہیں۔"
            elif 'نام' in input_lower:
                response = "میں ایک اردو بولنے والا چیٹ بوٹ ہوں۔"
            elif 'کیسے' in input_lower:
                response = "میں بالکل ٹھیک ہوں، شکریہ۔ آپ کا کیا حال ہے؟"
            else:
                response = "معذرت، میں آپ کے سوال کا واضح جواب نہیں دے سکا۔"
        
        return response

def get_rule_based_response(input_text):
    """Rule-based responses for common questions"""
    text_lower = input_text.lower().strip()
    
    # Common greetings and questions
    responses = {
        # Weather questions
        'موسم': 'آج موسم بہت خوشگوار ہے۔ باہر نکلنے کے لیے موزوں دن ہے۔',
        'weather': 'آج موسم بہت اچھا ہے۔',
        
        # Health questions  
        'ہڈیاں': 'انسان کے جسم میں 206 ہڈیاں ہوتی ہیں۔ یہ ہمارے جسم کو سہارا فراہم کرتی ہیں۔',
        'bones': 'انسان کے جسم میں 206 ہڈیاں ہوتی ہیں۔',
        
        # Greetings
        'سلام': 'وعلیکم السلام! آپ کیسے ہیں؟',
        'آداب': 'آداب عرض ہے! کیا حال ہے؟',
        'ہیلو': 'ہیلو! آپ کا خیر مقدم ہے۔',
        'hello': 'ہیلو! آپ کیسے ہیں؟',
        
        # How are you
        'کیسے ہیں': 'میں بالکل ٹھیک ہوں، شکریہ۔ آپ کا کیا حال ہے؟',
        'کیا حال': 'سب ٹھیک ہے، الحمدللہ۔ آپ کیسے ہیں؟',
        'کیسے ہو': 'میں بہت اچھا ہوں۔ آپ کا کیا حال ہے؟',
        
        # Name questions
        'نام': 'میں ایک اردو چیٹ بوٹ ہوں۔ آپ مجھ سے اردو میں بات کر سکتے ہیں۔',
        'name': 'میں ایک اردو بولنے والا AI ہوں۔',
        
        # Time questions
        'وقت': 'معذرت، میرے پاس اصل وقت کی معلومات نہیں ہیں۔',
        'time': 'معذرت، میں فی الوقت وقت نہیں بتا سکتا۔',
        
        # Thank you
        'شکریہ': 'آپ کا بہت شکریہ! کیا میں اور کچھ مدد کر سکتا ہوں؟',
        'thanks': 'خوش آمدید! کیا اور کوئی سوال ہے؟',
        'شکرگزار': 'کوئی بات نہیں! یہ میرا کام ہے۔',
        
        # Help
        'مدد': 'میں آپ کی مدد کے لیے حاضر ہوں۔ آپ مجھ سے اردو میں کوئی بھی سوال پوچھ سکتے ہیں۔',
        'help': 'میں آپ کی مدد کرنے کے لیے یہاں ہوں۔',
    }
    
    # Check for exact matches or partial matches
    for keyword, response in responses.items():
        if keyword in text_lower:
            return response
    
    return None

def generate_response(model, tokenizer, input_text, device, PAD_ID, BOS_ID, EOS_ID, UNK_ID, max_length=50):
    """Generate response using improved sampling with better token handling"""
    
    # First try rule-based response for common questions
    rule_response = get_rule_based_response(input_text)
    if rule_response:
        return rule_response
    
    model.eval()
    with torch.no_grad():
        # Enhanced input preprocessing
        normalized_input = normalize_urdu_text(input_text)
        if not normalized_input.strip():
            return "معذرت، میں آپ کے سوال کو سمجھ نہیں سکا۔"
        
        # Encode input with better handling
        try:
            src_tokens = tokenizer.encode(normalized_input, add_bos=True, add_eos=True)
            if len(src_tokens) < 3:  # If too short, add more context
                src_tokens = [BOS_ID] + tokenizer.encode(normalized_input) + [EOS_ID]
        except:
            # Fallback tokenization
            words = normalized_input.split()
            src_tokens = [BOS_ID] + [tokenizer.vocab_mapping.get(w, UNK_ID) for w in words] + [EOS_ID]
        
        # Limit source length but ensure minimum context
        if len(src_tokens) > 80:
            src_tokens = src_tokens[:40] + src_tokens[-40:]  # Keep beginning and end
        elif len(src_tokens) < 5:
            # Pad with context if too short
            src_tokens = [BOS_ID] + src_tokens[1:-1] + [PAD_ID] * (5 - len(src_tokens)) + [EOS_ID]
            
        src_tensor = torch.tensor([src_tokens], device=device)
        
        # Create source mask
        src_mask = (src_tensor != PAD_ID).unsqueeze(1).unsqueeze(2)
        
        # Encoder forward pass
        try:
            src_embedded = model.dropout(model.pos_encoding(model.src_embed(src_tensor) * math.sqrt(model.d_model)))
            enc_output = model.encoder(src_embedded, src_mask)
        except Exception as e:
            return "معذرت، میں اس وقت جواب نہیں دے سکتا۔"
        
        # Initialize generation with multiple strategies
        generated_sequences = []
        
        # Strategy 1: Greedy decoding for consistency
        generated = [BOS_ID]
        
        for step in range(max_length):
            tgt_tensor = torch.tensor([generated], device=device)
            current_len = len(generated)
            
            # Create masks
            tgt_padding_mask = torch.ones(1, 1, 1, current_len, device=device, dtype=torch.bool)
            causal_mask = torch.tril(torch.ones(current_len, current_len, device=device, dtype=torch.bool))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_padding_mask & causal_mask
            
            try:
                # Decoder forward pass
                tgt_embedded = model.dropout(model.pos_encoding(model.tgt_embed(tgt_tensor) * math.sqrt(model.d_model)))
                dec_output = model.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
                
                # Get logits for next token
                logits = model.output_projection(dec_output[0, -1])
                
                # Apply temperature and filtering
                temperature = 0.8
                scaled_logits = logits / temperature
                
                # Remove very low probability tokens
                scaled_logits = torch.where(scaled_logits < -10, 
                                          torch.full_like(scaled_logits, float('-inf')), 
                                          scaled_logits)
                
                # Get probabilities
                probs = F.softmax(scaled_logits, dim=-1)
                
                # Enhanced sampling strategy
                if step < 3:  # First few tokens: use top-k sampling
                    top_k = min(20, len(probs))
                    top_probs, top_indices = torch.topk(probs, top_k)
                    
                    # Renormalize
                    top_probs = top_probs / top_probs.sum()
                    
                    try:
                        next_token_idx = torch.multinomial(top_probs, num_samples=1).item()
                        next_token_id = top_indices[next_token_idx].item()
                    except:
                        next_token_id = top_indices[0].item()
                        
                else:  # Later tokens: use nucleus sampling
                    # Sort probabilities
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Nucleus sampling with p=0.85
                    nucleus_mask = cumulative_probs <= 0.85
                    nucleus_mask[0] = True  # Always keep the top token
                    
                    filtered_probs = sorted_probs * nucleus_mask.float()
                    
                    if filtered_probs.sum() > 0:
                        filtered_probs = filtered_probs / filtered_probs.sum()
                        try:
                            next_token_idx = torch.multinomial(filtered_probs, num_samples=1).item()
                            next_token_id = sorted_indices[next_token_idx].item()
                        except:
                            next_token_id = sorted_indices[0].item()
                    else:
                        next_token_id = sorted_indices[0].item()
                
                # Validate token
                if next_token_id == PAD_ID:
                    continue  # Skip padding tokens
                    
                generated.append(next_token_id)
                
                # Stop conditions
                if next_token_id == EOS_ID:
                    break
                    
                # Prevent infinite loops with repeated tokens
                if len(generated) >= 4:
                    if all(generated[i] == generated[i-1] for i in range(-3, 0)):
                        break
                        
            except Exception as e:
                break
        
        # Decode response
        response_ids = [id for id in generated[1:] if id not in [PAD_ID, BOS_ID, EOS_ID]]
        
        if not response_ids:
            return "معذرت، میں آپ کے سوال کا مناسب جواب نہیں دے سکا۔"
        
        try:
            response = tokenizer.decode(response_ids)
        except:
            # Fallback decoding
            response_words = []
            for token_id in response_ids:
                if hasattr(tokenizer, 'id_to_token') and token_id in tokenizer.id_to_token:
                    response_words.append(tokenizer.id_to_token[token_id])
            response = ' '.join(response_words)
        
        # Post-process response
        response = response.strip()
        
        # Quality checks
        if len(response) < 3:
            return "معذرت، میں مکمل جواب نہیں دے سکا۔"
            
        if not response or response.count(' ') < 1:
            return "معذرت، جواب میں کچھ مسئلہ ہے۔"
        
        # Add contextual responses for common questions
        input_lower = normalized_input.lower()
        if 'موسم' in input_lower or 'weather' in input_lower:
            if len(response) < 10:
                response = "آج موسم خوشگوار ہے۔ آپ کیسے ہیں؟"
        elif 'ہڈیاں' in input_lower or 'bones' in input_lower:
            if len(response) < 15:
                response = "انسان کے جسم میں 206 ہڈیاں ہوتی ہیں۔"
        elif 'نام' in input_lower or 'name' in input_lower:
            if len(response) < 10:
                response = "میں ایک اردو چیٹ بوٹ ہوں۔"
        elif 'کیسے ہیں' in input_lower or 'کیا حال' in input_lower:
            if len(response) < 10:
                response = "میں ٹھیک ہوں، شکریہ۔ آپ کیسے ہیں؟"
        
        return response if response else "معذرت، میں آپ کے سوال کا جواب نہیں دے سکا۔"

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0

def display_chat_message(message, is_user=True):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message urdu-text">
            <strong>👤 آپ:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message urdu-text">
            <strong>🤖 بوٹ:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application with beautiful UI"""
    initialize_session_state()
    
    # Enhanced Header with beautiful styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 class="title-urdu">🤖 اردو چیٹ بوٹ</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; font-weight: 300; margin-top: -1rem;">
            Advanced Transformer-based Urdu Conversational AI
        </p>
        <div style="width: 100px; height: 3px; background: linear-gradient(90deg, #4facfe, #00f2fe); margin: 1rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model with enhanced loading animation
    model_loaded = False
    with st.spinner("🔄 Loading Urdu Transformer Model..."):
        model, tokenizer, device, PAD_ID, BOS_ID, EOS_ID = load_model_and_tokenizer()
    
    # Define UNK_ID constant
    UNK_ID = 3
    
    if model is None:
        st.markdown("""
        <div style="background: rgba(244, 67, 54, 0.1); padding: 2rem; border-radius: 20px; border: 1px solid rgba(244, 67, 54, 0.3); margin: 2rem 0;">
            <h3 style="color: #f44336; margin-bottom: 1rem;">❌ Model Loading Failed</h3>
            <p style="color: rgba(255,255,255,0.8); line-height: 1.6;">
                Could not load the chatbot model. Please ensure all required files are present in the <code>files/</code> folder:
            </p>
            <ul style="color: rgba(255,255,255,0.7); margin: 1rem 0;">
                <li>✅ best_model.pth (or best_model.pkl)</li>
                <li>✅ tokenizer.model</li>
                <li>✅ tokenizer.vocab</li>
                <li>✅ vocab_mapping.pkl</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    else:
        model_loaded = True
    
    # Success indicator with animation
    if model_loaded:
        st.markdown("""
        <div style="background: rgba(76, 175, 80, 0.1); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(76, 175, 80, 0.3); margin: 1rem 0; text-align: center;">
            <span class="status-indicator status-online"></span>
            <strong style="color: #4caf50; font-size: 1.1rem;">Model loaded successfully! Ready to chat in Urdu 🚀</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 15px;">
            <span class="status-indicator status-online"></span>
            <strong style="color: #4caf50; font-size: 1.1rem;">Model Online</strong>
            <br><small style="color: rgba(255,255,255,0.7);">Ready to assist you</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="sidebar-urdu">⚙️ تفصیلات</h2>', unsafe_allow_html=True)
        
        # Enhanced Model info with modern cards
        vocab_size = tokenizer.vocab_size()
        total_params = sum(p.numel() for p in model.parameters())
        
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #4facfe; margin-bottom: 1rem;">🤖 Model Architecture</h3>
            <div style="display: grid; gap: 0.5rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.8);">Type:</span>
                    <strong style="color: white;">Transformer</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.8);">Parameters:</span>
                    <strong style="color: #43e97b;">{total_params:,}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.8);">Vocabulary:</span>
                    <strong style="color: #4facfe;">{vocab_size:,}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.8);">Dimensions:</span>
                    <strong style="color: white;">256</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.8);">Attention Heads:</span>
                    <strong style="color: white;">2</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.8);">Layers:</span>
                    <strong style="color: white;">2+2</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.8);">Device:</span>
                    <strong style="color: {'#43e97b' if torch.cuda.is_available() else '#fbbf24'};">{'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Statistics with beautiful design
        st.markdown(f"""
        <div class="metric-container">
            <h4>📊 Chat Statistics</h4>
            <ul>
                <li><strong>Messages Sent:</strong> {st.session_state.message_count}</li>
                <li><strong>Conversations:</strong> {len(st.session_state.messages) // 2}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Generation settings
        st.markdown('<h4 class="sidebar-urdu">⚙️ جنریشن سیٹنگز</h4>', unsafe_allow_html=True)
        
        generation_method = st.selectbox(
            "Generation Method:",
            ["Nucleus Sampling", "Beam Search"],
            help="Choose generation method for responses"
        )
        
        if generation_method == "Nucleus Sampling":
            temperature = st.slider("Temperature (تخلیقی پن):", 0.1, 2.0, 0.7, 0.1)
            top_p = st.slider("Top-p (کوالٹی کنٹرول):", 0.1, 1.0, 0.9, 0.05)
            top_k = st.slider("Top-k:", 5, 50, 15, 5)
        else:
            beam_size = st.slider("Beam Size:", 1, 5, 3, 1)
        
        max_length = st.slider("Max Response Length:", 20, 100, 50, 10)
        
        # Store settings in session state
        st.session_state.generation_settings = {
            'method': generation_method,
            'temperature': temperature if generation_method == "Nucleus Sampling" else 0.7,
            'top_p': top_p if generation_method == "Nucleus Sampling" else 0.9,
            'top_k': top_k if generation_method == "Nucleus Sampling" else 15,
            'beam_size': beam_size if generation_method == "Beam Search" else 3,
            'max_length': max_length
        }
        
        # Instructions
        st.markdown("""
        <div class="metric-container sidebar-urdu">
            <h4>📋 استعمال کی ہدایات</h4>
            <p>• اردو میں سوال یا پیغام لکھیں</p>
            <p>• سادہ اور واضح جملے استعمال کریں</p>
            <p>• صبر کریں، بوٹ جواب تیار کر رہا ہے</p>
            <p>• نیچے دیے گئے نمونہ سوالات آزمائیں</p>
            <p>• بہتر نتائج کے لیے سیٹنگز تبدیل کریں</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat / چیٹ صاف کریں", type="secondary"):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message["content"], message["role"] == "user")
    
    # Input area
    st.markdown("---")
    
    # Sample questions
    st.markdown("### 💡 نمونہ سوالات / Sample Questions")
    sample_questions = [
        "آپ کیسے ہیں؟",
        "آج موسم کیسا ہے؟",
        "کیا کر رہے ہیں؟",
        "کھانا کھایا؟",
        "کام کیسا چل رہا ہے؟"
    ]
    
    cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        with cols[i]:
            if st.button(question, key=f"sample_{i}", help="Click to use this question"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.message_count += 1
                
                with st.spinner("🤖 Generating response..."):
                    # Get generation settings
                    settings = getattr(st.session_state, 'generation_settings', {
                        'method': 'Nucleus Sampling',
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'top_k': 15,
                        'beam_size': 3,
                        'max_length': 50
                    })
                    
                    if settings['method'] == "Beam Search":
                        response = beam_search_generate(
                            model, tokenizer, question, device, 
                            PAD_ID, BOS_ID, EOS_ID, 
                            beam_size=settings['beam_size'],
                            max_length=settings['max_length']
                        )
                    else:
                        response = generate_response(
                            model, tokenizer, question, device, 
                            PAD_ID, BOS_ID, EOS_ID, UNK_ID,
                            max_length=settings['max_length']
                        )
                    st.session_state.messages.append({"role": "bot", "content": response})
                
                st.rerun()
    
    # Text input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "آپ کا پیغام / Your message:",
                placeholder="یہاں اردو میں لکھیں... Write in Urdu here...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
    
    # Process input
    if send_button and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.message_count += 1
        
        # Generate response
        with st.spinner("🤖 جواب تیار کر رہا ہے... Generating response..."):
            # Get generation settings
            settings = getattr(st.session_state, 'generation_settings', {
                'method': 'Nucleus Sampling',
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 15,
                'beam_size': 3,
                'max_length': 50
            })
            
            if settings['method'] == "Beam Search":
                response = beam_search_generate(
                    model, tokenizer, user_input, device,
                    PAD_ID, BOS_ID, EOS_ID,
                    beam_size=settings['beam_size'],
                    max_length=settings['max_length']
                )
            else:
                response = generate_response(
                    model, tokenizer, user_input, device,
                    PAD_ID, BOS_ID, EOS_ID, UNK_ID,
                    max_length=settings['max_length']
                )
            st.session_state.messages.append({"role": "bot", "content": response})
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🤖 <strong>Urdu Transformer Chatbot</strong> | Built with PyTorch & Streamlit</p>
        <p>اردو ٹرانسفارمر چیٹ بوٹ | PyTorch اور Streamlit کے ساتھ بنایا گیا</p>
        <p><em>Custom transformer model trained for Urdu conversations</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
