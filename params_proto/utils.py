def dot_to_deps(dot_dict, *prefixes):
    for p in prefixes:
        assert p, f"prefix {[p]} can not be empty."
    prefix = ".".join(prefixes)
    l = len(prefixes)
    child_deps = {}
    for k, v in dot_dict.items():
        path = tuple(k.split('.'))
        if k == ".":
            child_deps['.'] = v
            continue
        elif path[:l] == prefixes:
            rest = path[l:]
            if len(rest) == 1:
                child_deps[rest[0]] = v
            elif rest[0] in child_deps:
                child_deps[rest[0]]['.'.join(rest[1:])] = v
            else:
                child_deps[rest[0]] = {'.'.join(rest[1:]): v}

    return child_deps


class Examples:
    empty = {}
    set_self = {".": 10}  # should preserve
    set_bunch = { "resources.teacher.replica_hint": 10, }


if __name__ == "__main__":
    _ = dot_to_deps(Examples.empty)
    assert _ == {}, "result should be empty"
    print(_)
    _ = dot_to_deps(Examples.set_self)
    assert _ == {'.': 10}, "should contain dot"
    print(_)
    _ = dot_to_deps(Examples.set_bunch, 'resources')
    assert _ == {'teacher': {'replica_hint': 10}}, "should contain nested dict"
    print(_)

    _ = dot_to_deps({'root.launch_type': 'local'}, "root")
    assert _ == {"launch_type": 'local'}
    print(_)
