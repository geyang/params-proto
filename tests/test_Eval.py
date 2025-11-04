from params_proto.proto import Eval
from params_proto.utils import clean_ansi


def test_fn():
    proto = Eval @ "lambda: 10"
    assert proto.value() is 10


def test_fn_help_str():
    thunk = Eval("'hey'", help="the learning schedule")
    assert thunk.value is 'hey'
    assert clean_ansi(thunk.help).endswith("the learning schedule")
