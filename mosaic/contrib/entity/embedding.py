from datetime import datetime
from typing import Callable, Optional

import torch
from torch import Tensor

from mosaic.pipelines.entity import Entity


def default_suffix(column_name, call_num):
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{column_name}_{dt}"


class Embedding(torch.nn.Module):
    def __init__(
        self,
        entity_dp: Entity,
        embedding_col: str,
        save_embs: bool = True,
        save_emb_path: str = None,
        gen_emb_save_suffix: Callable = None,
        freeze=False,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super(Embedding, self).__init__()
        assert isinstance(entity_dp, Entity)
        self.entity_dp = entity_dp
        self.embedding_col = embedding_col
        self.save_embs = save_embs
        self.save_emb_path = save_emb_path
        self.save_call_count = 0
        self.gen_emb_save_suffix = (
            gen_emb_save_suffix if gen_emb_save_suffix is not None else default_suffix
        )
        assert (
            type(self.gen_emb_save_suffix(self.embedding_col, 0)) == str
        ), "gen_emb_save_suffix must return a string"
        assert (
            self.embedding_col in entity_dp.column_names
        ), f"{self.embedding_col} not in column_names"
        embeddings = self.entity_dp[self.embedding_col]._data.float()
        assert (
            embeddings.dim() == 2
        ), "Embeddings parameter is expected to be 2-dimensional"
        self.emb = torch.nn.Embedding.from_pretrained(
            embeddings,
            freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        # Save entity_dp to read in memmap mode if path is provided.
        # If not, keep it all in memory
        self._read_write_entity_dp()

    def forward(self, input: Tensor) -> Tensor:
        return self.emb(input)

    def _read_write_entity_dp(self):
        if self.save_emb_path is not None:
            self.entity_dp.write(self.save_emb_path)
            self.entity_dp = Entity.read(self.save_emb_path, mmap=True)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        res = super(Embedding, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if self.save_embs:
            new_col_name = self.gen_emb_save_suffix(
                self.embedding_col, self.save_call_count
            )
            self.entity_dp.add_embedding_column(
                new_col_name, self.emb.weight.detach().cpu().numpy(), overwrite=True
            )
            self._read_write_entity_dp()
            self.save_call_count += 1
        return res
