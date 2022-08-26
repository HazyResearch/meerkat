import weakref

_GRAPH 
class Node():
    def __init__(self, obj: object) -> None:
        super().__init__()
        self.obj_ref = weakref.ref(obj)
    
    def get(self):
        return self.obj_ref

class Pivot(Node):
    pass

    
def add_to_graph