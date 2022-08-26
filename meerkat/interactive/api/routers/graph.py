import weakref

class Edge:
    """Edges represent out-of-place operations performed w"""
class Node():
    def __init__(self, obj: object) -> None:
        super().__init__()
        self.obj_ref = weakref.ref(obj)
    
    def get(self):
        return self.obj_ref
    
    def get_edges(self):
        return 

class Pivot(Node):
    pass

    
def add_to_graph