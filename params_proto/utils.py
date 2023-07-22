import re

from typing import MutableMapping

ANSI_CLEANER = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]")


def clean_ansi(string: str):
    return ANSI_CLEANER.sub("", string)


def dot_to_deps(dot_dict, *prefixes):
    # Note: regularize for prefixes that contain dots
    for p in prefixes:
        assert p, f"prefix {[p]} can not be empty."
    if prefixes == (".",):
        prefixes = []
    else:
        full_prefix = ".".join(prefixes)
        assert ".." not in full_prefix
        prefixes: list = full_prefix.split('.')
    l = len(prefixes)
    child_deps = {}
    for k, v in dot_dict.items():
        if k == ".":
            child_deps['.'] = v
            continue

        path: list = k.split('.')
        if path[:l] == prefixes:
            rest = path[l:]
            if len(rest) == 1:
                child_deps[rest[0]] = v
            elif rest[0] in child_deps:
                child_deps[rest[0]]['.'.join(rest[1:])] = v
            else:
                child_deps[rest[0]] = {'.'.join(rest[1:]): v}

    return child_deps


def read_deps(*path: str, should_flatten: bool = True) -> dict:
    """
    Read YAML configuration file with additional functionality for Fast NeRF.

    Parameters
    ----------
    path: path to the config file
    should_flatten: whether to flatten dicts within the config file

    Returns
    -------
    Dict where the config is flattened and base configuration loaded,
    if applicable.
    """
    import os
    import yaml

    path = os.path.join(*path)
    dirname = os.path.dirname(path)

    with open(path, "r") as f:
        deps: dict = yaml.load(f, yaml.SafeLoader)
        print(f"Loaded config file {path}")
        if should_flatten:
            deps = flatten(deps)

        if "_base" in deps:
            # Load base config template if required
            base = deps.pop("_base")
            base_deps = read_deps(dirname, base)
            deps = {**base_deps, **deps}
            print(f"Loaded and merged base config template {base}")

        return deps


def flatten(nested_dict: dict, parent_key: str = "", sep: str = ".", generic_key: str = "_flatten", ) -> dict:
    """
    Flatten configuration dict. Use the "_flatten: False" key and value
    to flatten the dict in the YAML configuration files.
    """
    items = []
    for k, v in nested_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            if generic_key in v:
                assert v[generic_key] is False, "_flatten key needs to be False, currently True"
                items.append(
                    (new_key, {key: value for key, value in v.items() if key != generic_key})
                )
            else:
                items.extend(flatten(v, new_key, sep, generic_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    empty = {}
    _ = dot_to_deps(empty)
    assert _ == {}, "result should be empty"

    set_self = {".": 10}  # should preserve
    _ = dot_to_deps(set_self)
    assert _ == {'.': 10}, "should contain dot"

    set_bunch = {"resources.teacher.replica_hint": 10, }
    _ = dot_to_deps(set_bunch, 'resources')
    assert _ == {'teacher': {'replica_hint': 10}}, "should contain nested dict"

    _ = dot_to_deps({'root.launch_type': 'local'}, "root")
    assert _ == {"launch_type": 'local'}

    _ = dot_to_deps({'resources.teacher.replicas_hint': 10}, "resources.teacher")
    assert _ == {"replicas_hint": 10}
