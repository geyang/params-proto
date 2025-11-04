# Prefixed Configurations

> **Note**: This page is under construction. Check back soon for complete documentation.

The `@proto.prefix` decorator creates singleton configuration groups with automatic CLI prefixes.

## Example

```python
@proto.prefix
class Model:
    name: str = "resnet50"

@proto.prefix
class Training:
    lr: float = 0.001

@proto.cli
def main():
    print(f"Training {Model.name} with lr={Training.lr}")
```

CLI: `python main.py --Model.name vit --Training.lr 0.0001`

See [Decorators](decorators.md) for more details.
