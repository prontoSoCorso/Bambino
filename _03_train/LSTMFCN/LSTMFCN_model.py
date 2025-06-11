import numpy as np
import torch
import torch.nn as nn
import warnings, math


class MultiHeadAttention(nn.Module):
    """Lightweight attention mechanism for feature fusion"""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.orig_d = d_model
        self.n_heads = n_heads

        # compute padded dimension (ceiling to nearest multiple of n_heads, to ensure the divisibility)
        pad_to = math.ceil(d_model / n_heads) * n_heads
        self.pad_extra = pad_to - d_model

        # if padding is needed, project up; otherwise identity
        self.pre_proj = (
            nn.Linear(d_model, pad_to)
            if pad_to != d_model
            else nn.Identity()
        )
        self.d_model = pad_to
        self.d_k = pad_to // n_heads

        # Ensure d_model is divisible by n_heads
        
        # attention projections
        self.w_q = nn.Linear(pad_to, pad_to, bias=False)
        self.w_k = nn.Linear(pad_to, pad_to, bias=False)
        self.w_v = nn.Linear(pad_to, pad_to, bias=False)
        self.w_o = nn.Linear(pad_to, pad_to)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [B, L, orig_d]
        returns: [B, L, orig_d]
        """
        # 1) project up if needed
        x = self.pre_proj(x)                # [B, L, pad_to]
        batch_size, seq_len, _ = x.size()
        
        # Self-attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attenion scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V) # [B, heads, L, d_k]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )       # [B, L, pad_to]

        out = self.w_o(context)  # [B, L, pad_to]

        # Slice off padded dims
        if self.pad_extra:
            out = out[..., : self.orig_d]

        return out  # [B, L, orig_d]

class ResidualConvBlock(nn.Module):
    """Conv1D with residual + squeeze-excitation + batchnorm """
    def __init__(self, in_ch, out_ch, kernel, dropout, se_ratio=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Squeeze-and-Excitation
        se_ch = max(1, int(out_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, se_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(se_ch, out_ch, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Residual connection
        self.res = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.res_norm = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: [B, C, L]
        residual = self.res(x)
        if hasattr(self.res_norm, 'weight'):
            residual = self.res_norm(residual)

        # First conv block
        out = self.conv1(x)              # [B, C', L]
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)

        # Second conv block             
        out = self.conv2(out)           # [B, C_out, L]
        out = self.norm2(out)

        # SE attention
        se_weight = self.se(out)        # [B, C_out, 1]
        out = out * se_weight

        # Residual connection + activation  
        out = self.act(out + residual)  # [B, C_out, L]
        return out

class ModalitySpecificEncoder(nn.Module):
    """Separate encoding for each modality before fusion"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.encoder(x)      # [B, L, hidden_dim]

class LSTMFCN(nn.Module):
    """
    - Multivariate LSTM input (38 dims per timestep).
    - Modality-specific encoding
    - Multi-head attention
    - Residual blocks with SE

    init request:
    - modality_dims: dict { nome_modalità: dimensione_input }
    - enc_hidden_dim: hidden_dim di ciascun encoder
    - dropout_enc: float, dropout negli encoder
    - num_features: dimensione del livello di fusione (proiezione)
    - n_heads: numero di teste nell’attenzione
    - dropout_attn: dropout nell’attenzione
    - lstm_hidden_dim: hidden size dell’LSTM
    - lstm_layers: numero di layer LSTM
    - bidirectional: bool → LSTM bidirezionale
    - dropout_lstm: dropout tra layer LSTM (se layers>1)
    - cnn_filter_sizes: lista di 3 interi, canali di uscita dei 3 blocchi conv
    - cnn_kernel_sizes: lista di 3 interi, lunghezze dei kernel per i conv
    - dropout_cnn: dropout all’interno dei conv-block
    - se_ratio: rapporto per Squeeze-and-Excitation nei conv-block
    - dropout_classifier: dropout prima del classifier finale
    """
    def __init__(self, 
                 modality_dims: dict, 
                 enc_hidden_dim: int, 
                 dropout_enc: float, 
                 num_features: int, 
                 n_heads: int, 
                 dropout_attn: float, 
                 lstm_hidden_dim: int, 
                 lstm_layers: int, 
                 bidirectional: bool, 
                 dropout_lstm: float, 
                 cnn_filter_sizes: list, 
                 cnn_kernel_sizes: list, 
                 dropout_cnn: float, 
                 se_ratio: float, 
                 dropout_classifier: float):
        super().__init__()

        # Modality dimensions (parameterized)
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.num_modalities = len(self.modalities)

        # Modality-specific encoders
        # ciascun encoder mappa [input_dim --> enc_hidden_dim]
        self.encoders = nn.ModuleDict({
            name: ModalitySpecificEncoder(input_dim, enc_hidden_dim, dropout_enc)
            for name, input_dim in modality_dims.items()
        })

        # Fusion layer --> concat hidden of each encoder
        fusion_dim = enc_hidden_dim * self.num_modalities
        self.fusion_proj = nn.Linear(fusion_dim, num_features)
        self.fusion_norm = nn.LayerNorm(num_features)

        # Multi-head attention for temporal modeling
        self.attention = MultiHeadAttention(num_features, n_heads=n_heads, dropout=dropout_attn)

        # LSTM (input_size = num_features)
        lstm_input_size = num_features
        self.lstm = nn.LSTM(lstm_input_size, 
                            lstm_hidden_dim,
                            num_layers=lstm_layers,
                            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
                            bidirectional=bidirectional,
                            batch_first=True)

        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        # CNN branch
        fs = list(map(int, cnn_filter_sizes.split(",")))
        ks = list(map(int, cnn_kernel_sizes.split(",")))
        assert len(fs) == 3 and len(ks) == 3, \
            "Needed exactly 3 filter sizes and 3 kernel sizes for the CNN."
        self.conv_blocks = nn.ModuleList([
            ResidualConvBlock(num_features, fs[0], ks[0], dropout_cnn, se_ratio),
            ResidualConvBlock(fs[0],        fs[1], ks[1], dropout_cnn, se_ratio),
            ResidualConvBlock(fs[1],        fs[2], ks[2], dropout_cnn, se_ratio),
        ])

        # Pooling (global, along temporal dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Feature fusion and classification ([lstm_out + conv_feat] --> fully-connected)
        conv_feat_dim = fs[-1] * 2      # *2 for avg+max pooling
        final_dim = lstm_out_dim + conv_feat_dim  
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_classifier),
            nn.Linear(final_dim, final_dim // 2),
            nn.LayerNorm(final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_classifier * 0.5),
            nn.Linear(final_dim // 2, 1)    # scalar
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(module, 'weight'):
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias'):
                torch.nn.init.zeros_(module.bias)

    def forward(self, X_dict):
        """
        X_dict: dizionario di tensori, chiavi = stesse di modality_dims.
                Per ogni modalità “m”:
                   X_dict[m].shape = [batch_size, seq_len, modality_dims[m]]
        """
        # Separate modality processing (tensor lists like [B, L, enc_hidden_dim])
        encoded_feats = []
        for name in self.modalities:
            x_m = X_dict[name]  # [B, L, input_dim_m]
            feat_m = self.encoders[name](x_m)  # [B, L, enc_hidden_dim]
            encoded_feats.append(feat_m)
        
        # Fusion
        fused = torch.cat(encoded_feats, dim=-1)
        x = self.fusion_proj(fused)                 # [B, L, num_features]
        x = self.fusion_norm(x)                     # [B, L, num_features]

        # Self-attention for temporal modeling
        x_attn = self.attention(x)                  # [B, L, num_features]
        x = x + x_attn                              # [B, L, num_features]
        
        # LSTM branch
        lstm_out, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            # Concatenate forward and backward final states
            h_forward = h_n[-2]     # [B, lstm_hidden_dim]
            h_backward = h_n[-1]    # [B, lstm_hidden_dim]
            h_final = torch.cat([h_forward, h_backward], dim=1)  # [B, lstm_hidden_dim*2]
        else:
            h_final = h_n[-1]       # [B, lstm_hidden_dim]
        
        # CNN branch
        x_conv = x.permute(0, 2, 1)         # [B, num_features, L]
        for conv_block in self.conv_blocks:
            x_conv = conv_block(x_conv)     # [B, ch_out, L]
        
        # Dual pooling
        conv_avg = self.global_avg_pool(x_conv).squeeze(-1) # [B, fs[-1]]
        conv_max = self.global_max_pool(x_conv).squeeze(-1) # [B, fs[-1]]
        conv_feat = torch.cat([conv_avg, conv_max], dim=1)  # [B, fs[-1]*2]
        
        # Final fusion and classification
        final_feat = torch.cat([h_final, conv_feat], dim=1) # [B, lstm_out_dim + cnn_feat_dim]
        logits = self.classifier(final_feat).view(-1)       # [B]
        
        return logits
