class DAGNode:
    def __init__(self, parents: list["DAGNode"] = [], id: str = None) -> None:
        self.parents = parents

    def add_node(self, node: "DAGNode"):
        self.parents.append(node)

    def remove_node(self, id: str):
        self.parents = [node for node in self.parents if node.id != id]
