class node:
    __slots__ = 'attributes', 'seen', 'results', 'total_results', 'depth', 'curPrediction', 'bool', 'value', \
        'left', 'right'

    def __init__(self, attributes, seen, results, total_results, depth, curPrediction, boolean):
        self.attributes = attributes
        self.seen = seen
        self.results = results
        self.total_results = total_results
        self.depth = depth
        self.curPrediction = curPrediction
        self.bool = boolean
        self.value = None
        self.left = None
        self.right = None
