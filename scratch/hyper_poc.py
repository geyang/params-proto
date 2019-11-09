"""
First Pattern: use `params_proto.hypa.sweep` to create a proxy object.
"""
from typing import TypeVar

T = TypeVar("ParamsProto")

def sweep(config: T) -> T:
    pass
    # from waterbear import DefaultBear
    #
    # def factory():
    #     return DefaultBear(factory)
    #
    # proxy_sweep_object = factory()
    # return proxy_sweep_object

_ = sweep(G)
_.

with multiple(G) as _:
    # _.start_seed  = ['some ', 'other', 'value']
    _.





"""
Second Pattern
"""

@proto_partial(G)
with hyper.sweep as _ :
    def sweep(_: G):
        with hyper.product:
            _.
            # _.start_seed = ['language_select_described_object',]
            # with hyper.zip:
            #   _.teacher_checkpoint = ['/cns/li-d/home/geyang/r2d2_experts/8805121/1/',]
            #   _.level_name = ['language_select_described_object',]


