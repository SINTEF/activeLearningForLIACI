from dict2obj import Dict2Obj


def get_dict_from_file(f_path):
    values = {}
    with open(f_path, 'r') as f:
        for line in f:
            if not '=' in line:
                continue
            (k,v) = line.replace(' ', '').replace('\n','').split('=')
            values[k] = v
    return Dict2Obj(values)