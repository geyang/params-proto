#!/Users/geyang/anaconda3/bin/python
from params_proto.v1 import Proto, cli_parse


@cli_parse
class Params:  """
  Your ICLR best paper project

  -- Ge
  """

  seed = Proto(10, help="random seed for the environment")


if __name__ == "__main__":
  print(f"seed is {Params.seed}")
