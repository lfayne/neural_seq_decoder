import os
import pickle
import torch
import math

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.nn.functional as F

from neural_decoder.augmentations import GaussianSmoothing
from neural_decoder.dataset import SpeechDataset

MAX_TIME = 1000 # Max time series length across every dataset is 919 (round up to 1000)

def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData


'''
RoPE implementation from CS224N A4
'''
def precompute_rotary_emb(dim, max_positions):
    ts = torch.arange(max_positions).repeat(dim//2, 1).T
    ds = torch.arange(dim//2).repeat(max_positions, 1)
    thetas = 10000 ** (-2*ds/dim)

    rope_cache = torch.zeros((max_positions, dim//2, 2))
    rope_cache[:,:,0] = torch.cos(ts * thetas)
    rope_cache[:,:,1] = torch.sin(ts * thetas)

    return rope_cache


def apply_rotary_emb(x, rope_cache):
    temp_cache = rope_cache[:x.shape[2],:,:]
    rotated_x = torch.view_as_complex(temp_cache.view(1,1,temp_cache.shape[0],temp_cache.shape[1],temp_cache.shape[2])) \
              * torch.view_as_complex(x.reshape(x.shape[:-1] + (x.shape[-1]//2, 2)))

    return torch.view_as_real(rotated_x).view(x.shape)

# 10-20 time units per phoneme (50 context width?)
# Time series are padded with 0s
# TODO: verify full implementation of Multi Headed Attention
# TODO: check batch first params
'''
Multiheaded self attention block involving residual connections, layer normalization, and a feed forward layer.
'''
class attentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['output_dim'] % config['n_heads'] == 0
        assert config['output_dim'] % config['embed_dim'] == 0

        self.embed_dim = config['embed_dim']
        self.output_dim = config['output_dim']
        self.n_heads = config['n_heads']
        self.device = config['device']
        self.context = config['context'] # Number of preceding and succeeding time steps that are including (total context window size is 2*self.context+1)
        self.rope = config['rope']
        self.p_drop_att = config['drop_att']
        self.p_drop_final = config['drop_final']

        if self.rope:
            rope_cache = precompute_rotary_emb(self.output_dim//self.n_heads, config['max_time'])
            self.register_buffer("rope_cache", rope_cache)

        self.key = nn.Linear(self.embed_dim, self.output_dim)
        self.query = nn.Linear(self.embed_dim, self.output_dim)
        self.value = nn.Linear(self.embed_dim, self.output_dim)
        self.fc_att = nn.Linear(self.output_dim, self.output_dim)
        self.drop_att = nn.Dropout(self.p_drop_att)
        self.drop_final = nn.Dropout(self.p_drop_final)

        self.layer_norm_att = nn.LayerNorm(self.output_dim, device=self.device)
        self.layer_norm_ff = nn.LayerNorm(self.output_dim, device=self.device)
        self.fc_add_norm = nn.Linear(self.output_dim, self.output_dim, device=self.device)
        self.fc_ff = nn.Linear(self.output_dim, self.output_dim, device=self.device)

        self.apply(self._init_weights)

    def forward(self, x, x_len):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_heads, self.output_dim // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, self.output_dim // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_heads, self.output_dim // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # Apply RoPE
        if self.rope:
            k = apply_rotary_emb(k, self.rope_cache)
            q = apply_rotary_emb(q, self.rope_cache)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Mask out all neural data past end of sequence
        indices = torch.arange(T).repeat(B).reshape((B, T))
        mask = torch.where(indices >= x_len.unsqueeze(1), 0, 1).unsqueeze(1)
        att = att.masked_fill(mask.unsqueeze(-1) == 0, -1e10)
        att = att.masked_fill(mask.unsqueeze(-2) == 0, -1e10)

        # Restrict context window to +- self.context additional time elements worth of neural data
        if self.context:
            ones = torch.ones((T, T))
            context_mask = ones - torch.tril(ones, diagonal=-self.context-1) - torch.tril(ones, diagonal=-self.context-1).T
            att = att.masked_fill(context_mask.unsqueeze(0).unsqueeze(0) == 0, -1e10)

        # Finish attention mechanism post-masking
        att = F.softmax(att, dim=-1)
        att = self.drop_att(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.output_dim) # re-assemble all head outputs side by side

        # output projection
        self_att_out = self.drop_final(self.fc_att(y))

        # Apply residual connections, layernorms, and feed forward network
        if self.embed_dim != self.output_dim:
            adjusted_x = torch.zeros(self_att_out.size())
            for i in range(self.output_dim//self.embed_dim):
                adjusted_x[:,:,i::self.output_dim//self.embed_dim] = (self.embed_dim/self.output_dim) * x.detach().clone()
            add_norm = self.layer_norm_att(adjusted_x + self_att_out)
        else:
            add_norm = self.layer_norm_att(x + self_att_out)
        feed_for = self.fc_ff(F.elu(self.fc_add_norm(add_norm)))
        final_output = self.layer_norm_ff(feed_for + add_norm)

        return final_output
    
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

'''
Full transformer model for translating neural data to phonemes
'''
class transformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['neural_dim'] % config['n_heads'] == 0

        self.neural_dim = config['neural_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_classes = config['n_classes']
        self.context = config['context']
        self.n_days = config['n_days']
        self.context = config['context']
        self.dropout = config['dropout']
        self.n_heads = config['n_heads']
        self.convolve = config['convolve']
        self.kernel_size = config['kernel_size']
        self.stride_len = config['stride']
        self.device = config['device']
        self.gaussianSmoothWidth = config['gaussianSmoothWidth']
        self.inputLayerNonlinearity = torch.nn.Softsign()

        self.gaussianSmoother = GaussianSmoothing(
            self.neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(self.n_days, self.neural_dim, self.neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(self.n_days, 1, self.neural_dim))

        for x in range(self.n_days):
            self.dayWeights.data[x, :, :] = torch.eye(self.neural_dim)

        att_config = {'embed_dim': self.neural_dim, 'output_dim': self.hidden_dim, 'n_heads': self.n_heads, 'device': self.device, \
                      'context': self.context[0], 'rope': True, 'max_time': MAX_TIME, 'drop_att': self.dropout, 'drop_final': self.dropout}
        self.mh_attention = nn.ModuleList()
        self.mh_attention.append(attentionBlock(att_config))
        att_config['rope'] = False # Apply RoPE only to first encoder layer
        att_config['embed_dim'] = self.hidden_dim # Switch to (hidden_dim) x (hidden_dim) attention blocks
        for i in range(len(self.context)-1):
            att_config['context'] = self.context[i+1]
            self.mh_attention.append(attentionBlock(att_config))

        if self.convolve:
            self.conv_layer = torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, self.kernel_size, stride=self.stride_len, groups=self.hidden_dim)
            self.fc_decoder_out = nn.Linear(self.hidden_dim, self.n_classes + 1, device=self.device)
        else:
            self.unfolder = torch.nn.Unfold((self.kernel_size, 1), dilation=1, padding=0, stride=self.stride_len)
            self.fc_decoder_out = nn.Linear(self.hidden_dim * self.kernel_size, self.n_classes + 1, device=self.device)

        self.apply(self._init_weights)
    
    def forward(self, neuralInput, dayIdx, x_len):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # Apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # Apply transformer layers
        out = self.mh_attention[0](transformedNeural, x_len)
        for att_block in self.mh_attention[1:]:
            out = att_block(out, x_len)

        # Apply convolution/unfolding to reduce temporal resolution (to match closer to possible phoneme output)
        if self.convolve: seq_out = torch.permute(self.conv_layer(torch.permute(out, (0, 2, 1))), (0, 2, 1))
        else: seq_out = torch.permute(self.unfolder(torch.unsqueeze(torch.permute(out, (0, 2, 1)), 3)), (0, 2, 1))
        seq_out = F.elu(seq_out)

        # Apply final FC layer to get appropriate output dimension size
        seq_out = self.fc_decoder_out(seq_out)
        return seq_out
    
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
def main():
    # trainLoader, testLoader, loadedData = getDatasetLoaders(
    #     "neural_seq_decoder/src/neural_decoder/ptDecoder_ctc",
    #     64,
    # )
    # X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
    # print(X_len.shape)
    # B, nh, T = (4, 2, 3)
    # x_len = torch.tensor([1, 2, 3, 2])

    # temp = torch.randn((B, nh, T, T))
    # indices = torch.arange(T).repeat(B).reshape((B, T))
    # mask = torch.where(indices >= x_len.unsqueeze(1), 0, 1).unsqueeze(1)
    # temp = temp.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
    # temp = temp.masked_fill(mask.unsqueeze(-2) == 0, float('-inf'))
    # print(temp[0, :, :, :])
    # print(temp[1, :, :, :])
    # print(temp[2, :, :, :])
    # print(temp[3, :, :, :])

    # temp = torch.randn(size=(100,100))
    # ones = torch.ones(100,100)
    # mask = ones - torch.tril(ones, diagonal=-20-1) - torch.tril(ones, diagonal=-20-1).T
    # temp = temp.masked_fill(mask == 0, float('-inf'))
    # print(temp[0,0], temp[0,19], temp[50, 70], temp[50, 30], temp[70, 50], temp[30, 50])
    # print(temp[0,21], temp[-1,-22], temp[50, 71], temp[50, 29])

    return

if __name__ == "__main__":
    main()