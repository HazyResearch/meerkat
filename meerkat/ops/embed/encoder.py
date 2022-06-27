from dataclasses import dataclass


@dataclass
class Encoder:
    encode: callable
    preprocess: callable = None
    collate: callable = None
