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


if __name__ == "__main__":
    empty = {}
    _ = dot_to_deps(empty)
    assert _ == {}, "result should be empty"

    set_self = {".": 10}  # should preserve
    _ = dot_to_deps(set_self)
    assert _ == {'.': 10}, "should contain dot"

    set_bunch = {"resources.teacher.replica_hint": 10, }
    _ = dot_to_deps(set_bunch, 'resources')
    print(_)
    assert _ == {'teacher': {'replica_hint': 10}}, "should contain nested dict"

    _ = dot_to_deps({'root.launch_type': 'local'}, "root")
    assert _ == {"launch_type": 'local'}

    _ = dot_to_deps({'resources.teacher.replicas_hint': 10}, "resources.teacher")
    assert _ == {"replicas_hint": 10}
