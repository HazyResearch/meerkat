
from ...html import div



def complete_prompt(row, example_template: mk.Store[str]):
    assert isinstance(row, dict)
    output = example_template.format(**row)
    return output


class FlashFill(div):

    def __init__(

    ):
    
    