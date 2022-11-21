from meerkat.dataframe import DataFrame
from meerkat.gui.graph import reactive

@dataclass
class PromptRequest:
    """Arguments for a prompt request.
    
    Args:
        client (str): client to use for generation
        engine (str): engine to use for generation
        n (int): number of generations per input
    """
    client: str
    engine: str
    n: int
    

@capture_provenance(capture_args=["template"])
@reactive
def prompt(
    df: DataFrame,
    template,
    client: Manifest,
    batch_size: int = 16,
    num_workers: int = 0,
    
):  
    """
    Run a prompt over a DataFrame.


    """
    from manifest import Manifest
    
    def _run_batch():
        Manifest(
            client_name=request.client,
            client_connection=,
        )

    return df.map(_run_batch, batch_size=batch_size)