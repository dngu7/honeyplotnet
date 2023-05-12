# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F

class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, use_fp16=False, **kwargs):
        super().__init__()

        self.name = 'vq'
        self.dtype_float = torch.float32 #torch.float32 if use_fp16 else torch.float32
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embeddings = torch.nn.Embedding(
            num_embeddings,
            embedding_dim
        )

        torch.nn.init.uniform_(self.embeddings.weight)

    def forward(self, x, is_indices=False, **kwargs):
        if is_indices:
            return self.sample_decoder(x)
        else:
            return self._forward(x)

    def sample_decoder(self, encoding_indices):
        bs, x_dim = encoding_indices.shape
        output_shape = [bs, x_dim, self.embedding_dim]

        flattened = torch.reshape(encoding_indices, [bs, -1])

        encodings = F.one_hot(flattened, self.num_embeddings).to(self.dtype_float).to(encoding_indices.device)
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = torch.reshape(quantized, output_shape)
        return quantized

    def _forward(self, x):

        input_shape = x.shape
        assert input_shape[-1] == self.embedding_dim
        
        flattened = torch.reshape(x, [-1, self.embedding_dim])
        encoding_indices = self.get_code_indices(flattened)

        encodings = F.one_hot(encoding_indices, self.num_embeddings).to(self.dtype_float).to(x.device)

        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = torch.reshape(quantized, input_shape)

        commitment_loss = self.beta * torch.mean(
            (quantized.detach() - x) ** 2
        )

        codebook_loss = torch.mean((quantized - x.detach()) ** 2)

        loss = commitment_loss + codebook_loss

        # Straight-through estimator.
        out = x + (quantized - x).detach()

        return out, quantized, loss

    def get_code_indices(self, q_z):
        flattened_inputs = torch.reshape(q_z, [-1, self.embedding_dim])

        similarity = torch.matmul(flattened_inputs, self.embeddings.weight.t())

        s1 = torch.sum(flattened_inputs ** 2, axis=1, keepdims=True)
        s2 = torch.sum(self.embeddings.weight.t() ** 2, axis=0)
        s3 = - 2 * similarity

        distances = s1 +s2 + s3

        # Derive the indices for minimum distances.
        _, encoding_indices = torch.min(distances, axis=1)
        return encoding_indices
    
    def get_embed(self, **kwargs):
        return self.embeddings.weight