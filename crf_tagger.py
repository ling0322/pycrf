import re

class _TaggerNode:
    def __init__(self):
        self.cost = None
        self.previous_node = None

    def __repr__(self):
        return "[cost={}, prev={}]".format(self.cost, self.previous_node)

class CRFTagger:
    def __init__(self, crf_model):
        self._model = crf_model
        self._re_x = re.compile(r'%x\[(-?\d+),(\d+)\]')
        self._uni_templates = self._model.get_uni_templates()
        self._bi_templates = self._model.get_bi_templates()

        # number of tags in model file
        self._y_size = self._model.get_tag_num()

        # Initialize the decode buckets
        self._buckets = []

        self._bos = ["_B-1", "_B-2", "_B-3", "_B-4", "_B-5", "_B-6", "_B-7", "_B-8"]
        self._eos = ["_B+1", "_B+2", "_B+3", "_B+4", "_B+5", "_B+6", "_B+7", "_B+8"]

    def _get_uni_cost(self, x, xpos, tag_id):
        features = list(map(lambda tmpl: self.get_feature_str(tmpl, x, xpos), self._uni_templates))
        cost = sum(map(lambda f: self._model.get_uni_feature_cost(f, tag_id), features))
        return cost

    def _get_bi_cost(self, x, xpos, left_tag_id, right_tag_id):
        features = list(map(lambda tmpl: self.get_feature_str(tmpl, x, xpos), self._bi_templates))
        cost = sum(map(lambda f: self._model.get_bi_feature_cost(f, left_tag_id, right_tag_id), features))
        return cost 
    
    def _check_and_extend_buckets(self, need_size):
        if len(self._buckets) < need_size:
            for _ in range(len(self._buckets), need_size):
                bucket = [_TaggerNode() for _ in range(self._y_size)]
                self._buckets.append(bucket)

    def _get_bucket_cost(self, x, xpos):
        for tag_id, node in enumerate(self._buckets[xpos]):
            node.cost += self._get_uni_cost(x, xpos, tag_id)

    def _get_arc_cost(self, x, xpos):
        possible_arc = []
        for right_tag_id, node_right in enumerate(self._buckets[xpos]):
            possible_arc.clear()
            for left_tag_id, node_left in enumerate(self._buckets[xpos - 1]):
                cost = node_left.cost + self._get_bi_cost(x, xpos, left_tag_id, right_tag_id)
                possible_arc.append((cost, left_tag_id))
            left = max(possible_arc)
            node_right.previous_node = left[1]
            node_right.cost = left[0]

    def _clear_bucket(self, xpos):
        for node in self._buckets[xpos]:
            node.cost = 0
            node.previous_node = 0

    def _viterbi(self, x):
        x_size = len(x)
        for xpos in range(x_size):
            if xpos == 0:
                self._clear_bucket(0)
            else:
                self._get_arc_cost(x, xpos)
            self._get_bucket_cost(x, xpos)

    def _find_best_result(self, last_bucket):
        best_tup = max(enumerate(self._buckets[last_bucket]), key = lambda x: x[1].cost)
        result = [None for _ in range(last_bucket + 1)]
        tag_id = best_tup[0]
        bucket_id = last_bucket
        while bucket_id >= 0:
            result[bucket_id] = tag_id
            tag_id = self._buckets[bucket_id][tag_id].previous_node
            bucket_id -= 1

        return result

    def tag(self, x):
        self._check_and_extend_buckets(len(x))
        self._viterbi(x)
        result_tag = self._find_best_result(len(x) - 1)
        return list(map(lambda y: self._model.get_tag_str(y), result_tag))

    def get_feature_str(self, template_str, x, xpos):
        def _repl(matchobj):
            row = int(matchobj.group(1)) + xpos
            if row < 0:
                return self._bos[-row - 1]
            elif row >= len(x): 
                return self._eos[row - len(x)]
            column = int(matchobj.group(2))
            return x[row][column]

        return self._re_x.sub(_repl, template_str)

