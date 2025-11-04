# Parameter Overrides

> **Note**: This page is under construction. Check back soon for complete documentation.

There are multiple ways to override parameters in params-proto v3.

## Override Methods

1. **Command Line**: `python script.py --lr 0.01`
2. **Direct Assignment**: `Config.lr = 0.01`
3. **Function kwargs**: `train(lr=0.01)`
4. **proto.bind()**: `proto.bind(lr=0.01)`

See [Quick Start](../quick_start.md) for examples.
