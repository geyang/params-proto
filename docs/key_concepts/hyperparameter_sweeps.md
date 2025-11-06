# Hyperparameter Sweeps

Hyperparameter sweeps are essential for finding optimal configurations in machine learning and experimentation. This guide shows you how to perform systematic parameter searches using params-proto.


```python
# V2 style (legacy)
from params_proto import Sweep

class Config(ParamsProto):
    lr = Proto(default=0.001)
    batch_size = Proto(default=32)

sweep = Sweep(Config).product([
    Config.lr << [0.001, 0.01, 0.1],
    Config.batch_size << [32, 64],
])

for deps in sweep:
    train()

```

