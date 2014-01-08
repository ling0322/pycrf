import crf_tagger
import crf_model
import sys


if len(sys.argv) != 3:
    print('Usage python3 crf_seg.py MODEL_FILE SEG_FILE')

model = crf_model.CRFModel(sys.argv[1])
tagger = crf_tagger.CRFTagger(model)

def _tag_to_seg(text, tag):
    result = []
    word = ''
    for i, ch in enumerate(text):
        word += ch
        if tag[i] in ('E', 'S'):
            result.append(word)
            word = ''
    result.append(word) 
    return result

with open(sys.argv[2]) as fp:
    for line in fp:
        line = line.strip()
        x = list(map(lambda x: [x], line))
        print('  '.join(_tag_to_seg(line, tagger.tag(x))))