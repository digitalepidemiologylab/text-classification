from itertools import chain, starmap


def flatten_dict(dict_):
    """Flatten a nested dictionary structure"""
    # https://codereview.stackexchange.com/questions/173439/pythonic-way-to-flatten-nested-dictionarys

    def unpack(parent_key, parent_value):
        """Unpack one level of nesting in a dictionary"""
        try:
            items = parent_value.items()
        except AttributeError:
            # parent_value was not a dict, no need to flatten
            yield (parent_key, parent_value)
        else:
            for key, value in items:
                yield (parent_key + (key,), value)

    # Put each key into a tuple to initiate building a tuple of subkeys
    dict_ = {(key,): value for key, value in dict_.items()}

    while True:
        # Keep unpacking the dictionary until all value's are not dictionary's
        dict_ = dict(
            chain.from_iterable(starmap(unpack, dict_.items())))
        if not any(isinstance(value, dict) for value in dict_.values()):
            break

    return dict_


def set_nested_value(dict_, keys, value):
    # https://stackoverflow.com/questions/13687924/setting-a-value-in-a-nested-python-dictionary-given-a-list-of-indices-and-value
    for key in keys[:-1]:
        dict_ = dict_.setdefault(key, {})
    dict_[keys[-1]] = value


def merge_dicts(a, b, path=None):
    """Merges b into a.

    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
        else:
            a[key] = b[key]
    return a
