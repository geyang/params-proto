from params_proto import cli_parse, Proto, ParamsProto

@cli_parse
class G:
    some_arg = Proto(0, aliases=['-s'])
