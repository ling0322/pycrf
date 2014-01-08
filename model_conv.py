#!/usr/bin/env python3

import pickle
import sys
import gzip

def load_metadata(fd):
    '''
    Load metadate (like version, max_id ...) from model file
    '''
    metadata = {}
    for line in fd:
        line = line.strip()
        if line == '': break

        fields = line.split(': ')
        key = fields[0]
        value = fields[1]
        metadata[key] = value

    return metadata

def load_tagset(fd):
    '''
    Load the tag set from model file
    '''
    tagset = []
    for line in fd:
        line = line.strip()
        if line == '': break

        tagset.append(line)

    return tagset


def load_template(fd):
    '''
    Load template from model file
    '''
    templates = []
    for line in fd:
        line = line.strip()
        if line == '': break

        templates.append(line)

    return templates    

def load_feature_index(fd):
    '''
    Load index for each feature string from model file
    '''
    feature_index = {}
    for line in fd:
        line = line.strip()
        if line == '': break

        fields = line.split()
        index = int(fields[0])
        feature_str = fields[1]
        feature_index[feature_str] = index

    return feature_index

def load_data(fd):
    '''
    Load crf cost data from model file 
    '''
    crf_data = []
    for line in fd:
        line = line.strip()
        if line == '': break

        cost = float(line)
        crf_data.append(cost)

    return crf_data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Usage ./model_conv.py CRF++_Text_File Pickle_File\n')
        quit()

    with open(sys.argv[1]) as fd:
        metadata = load_metadata(fd)
        if metadata['version'] != '100':
            raise 'Invalid model format!'

        tagset = load_tagset(fd)
        templates = load_template(fd)
        feature_index = load_feature_index(fd)
        cost_data = load_data(fd)

        if len(cost_data) != int(metadata['maxid']):
            raise 'Invalid model format!'

        crf_model = dict(
            metadata = metadata,
            tagset = tagset,
            templates = templates,
            feature_index = feature_index,
            cost_data = cost_data)
        
        with gzip.open(sys.argv[2], 'wb') as fd_pickle:
            pickle.dump(crf_model, fd_pickle)




