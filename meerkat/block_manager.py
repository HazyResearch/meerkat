
from meerkat.columns.abstract import AbstractColumn
import numpy as np
from typing import Dict, Hashable, Tuple, List, Any, Sequence, Union


def infer_column_type(data, names: Union[str, Sequence[str]]=None):
    
    for block_class in registry:
    ## TODO: discuss how this relates to the type inference of AbstractColumn.from_data 
        if block_class.match(data):
            return block_class.get_block_signature(data)

     if names is None:
            names = block.get_names()
        if names is None or len(names) != len(block):
            raise ValueError("Could not determine names")
    
    return column_type, names 


class BlockManager:
    """
    
    This manager manages all blocks.
    """
    def __init__(self) -> None:
        self._data: Dict[str, AbstractColumn] = {}
        self.blocks: Dict[Hashable, AbstractBlock]= {}
        # Mapping from datapanel column to (block idx, local idx in block)
        self.name_to_block_location: Dict[str, Tuple[Hashable, Any]]

    def insert(self, data, names: Union[str, Sequence[str]]=None):
        """
        Loop through all block instances, and check for a match.
        
        If match, then insert into that block.
        If not match, create a new block.

        Args:
            data (): a single blockable object, potentially contains multiple columns
        """
        
        column_type, column_constructor, name_to_data_idx = infer_column_type(data, names)
        is_multiple = len(name_to_idx) > 1

        if column_type.block_type is None:
            # These columns are not stored in a block
            if is_multiple:
                for name, data_idx in name_to_idx.items():
                    self._data[name] = column_constructor(data[data_idx])
            else:
                name = list(name_to_idx.keys())[0]
                self._data[name] = column_constructor(data)
            return 

        # Convert `data` to a block
        block, name_to_block_idx  = column_type.block_type.blockify(data, name_to_data_idx)
        block_sig = block.signature
        if block_sig in self.blocks:
            # indices of the new part of the block
            name_to_block_idx = self.blocks[block_sig].insert(block, name_to_block_idx)
            # sometimes (fast_insert = insert o blockify) will be faster i.e. run local_indices = self.blocks[block_sig].fast_insert(data)
        else:
            self.blocks[block_sig] = block
        

        self.name_to_block_location.update({
            name: (block_sig, block_idx) for name, block_idx in name_to_block_idx.items()
        })

        self._data.update({
            name: column_constructor(block[block_idx]) for name, block_idx in name_to_block_idx.items()
        })
    
    def __getitem__(self, index):
        if isinstance(index, str):
            return self._data[index]
            
        

registry = [NumpyBlock, TensorBlock]

def get_block_signature(data):
    column_type = infer_column_type(data)
    block_type = column_type.block_type
    return block_type.get_block_signature(data)

    
class AbstractBlock:
    @classmethod
    def match(cls, data) -> bool:
        pass

    def insert(self, data):
        """
        Cannot change block_index of 
        """
        pass
    
    def __getitem__(self, index):
        # The index should be something that was returned by self.insert
        pass
    
    def get_names(self):
        return None
    
    def indices(self):
        raise NotImplementedError
    
    @classmethod
    def blockify(cls, data):
        if not cls.match(data):
            raise ValueError
    
    @classmethod
    def get_block_signature(data) -> Hashable:
        pass
    
    @classmethod
    def get_default_column_type(cls):
        raise NotImplementedError

