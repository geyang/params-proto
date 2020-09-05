#!python3
from params_proto.neo_proto import ParamsProto


def get_indent(text):
    for line in text.split("\n"):
        if line:
            return len(line) - len(line.lstrip())
    return 0


def dedent(text):
    """Copied from the `cmx` package."""
    lines = text.split('\n')
    for l in lines:
        if l:
            break
    indent = get_indent(l)
    return '\n'.join([l[indent:] for l in lines])


class Root(ParamsProto, cli_parse=False):
    """Root config object

    When multiple config objects are used, the first few
    being initiated need to have the `cli_parse` flag
    set to `False`.
    """
    launch_type = "borg"


class Config(ParamsProto):
    """Your ICLR best paper project"""
    seed = 10

    def __init__(self, _dep=None):
        r = Root(_dep)

        if r.launch_type == "borg":
            self.replica_hint = 26
        else:
            self.replica_hint = 1


if __name__ == "__main__":
    from params_proto.neo_proto import ARGS

    help_str = ARGS.parser.format_help()
    print(help_str)

    # print(f"seed is {Config().seed}")
    # print(f"seed is {Config().replica_hint}")
    #
    # for launch_type in ['borg', 'local']:
    #     config = Config({"Root.launch_type": launch_type})
    #     assert config.replica_hint == 26 if launch_type == "borg" else 1
    #     print(config.replica_hint)
