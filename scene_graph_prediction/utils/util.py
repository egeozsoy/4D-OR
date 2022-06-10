import os, json


def check_file_exist(path):
    if not os.path.exists(path):
        raise RuntimeError('Cannot open file. (', path, ')')


def read_txt_to_list(file):
    output = []
    with open(file, 'r') as f:
        for line in f:
            entry = line.rstrip()
            output.append(entry)
    return output


def check_file_exist(path):
    if not os.path.exists(path):
        raise RuntimeError('Cannot open file. (', path, ')')


def read_classes(read_file):
    obj_classes = []
    with open(read_file, 'r') as f:
        for line in f:
            obj_class = line.rstrip()
            obj_classes.append(obj_class)
    return obj_classes


def read_relationships(read_file):
    relationships = []
    with open(read_file, 'r') as f:
        for line in f:
            relationship = line.rstrip()
            relationships.append(relationship)
    return relationships
