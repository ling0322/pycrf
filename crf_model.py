import pickle
import gzip

class CRFModel:
    def __init__(self, model_path):
        with gzip.open(model_path, 'rb') as fd:
            model = pickle.load(fd)
        self._metadata = model['metadata']
        self._tagset = model['tagset']
        self._feature_index = model['feature_index']
        self._cost_data = model['cost_data']

        self._tag_num = len(self._tagset)
        templates = model['templates']
        self._uni_templates = set(filter(lambda tmpl: tmpl[0] == 'U', templates))
        self._bi_templates = set(filter(lambda tmpl: tmpl[0] == 'B', templates))

        self._tag_index = {}
        for tag_id, tag_str in enumerate(self._tagset):
            self._tag_index[tag_str] = tag_id

    def get_uni_feature_cost(self, feature_str, tag_id):
        feature_id = self._feature_index.get(feature_str, -1)
        if feature_id == -1:
            return 0
        return self._cost_data[feature_id + tag_id]

    def get_bi_feature_cost(self, feature_str, left_tag_id, right_tag_id):
        feature_id = self._feature_index.get(feature_str, -1)
        if feature_id == -1:
            return 0

        return self._cost_data[feature_id + left_tag_id * self._tag_num + right_tag_id]

    def get_uni_templates(self):
        return self._uni_templates

    def get_bi_templates(self):
        return self._bi_templates

    def get_tag_num(self):
        return self._tag_num

    def get_tag_id(self, tag_str):
        return self._tag_index[tag_str]

    def get_tag_str(self, tag_id):
        return self._tagset[tag_id]

