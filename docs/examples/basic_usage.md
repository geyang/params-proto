# Basic Usage Examples

> **Note**: This page is under construction. Check back soon for complete documentation.

## Simple Function Example

```python
from params_proto import proto

@proto.cli
def greet(
    name: str = "World",  # Name to greet
    times: int = 1,  # Number of times to greet
):
    """Print a greeting."""
    for _ in range(times):
        print(f"Hello, {name}!")

if __name__ == "__main__":
    greet()
```

## Simple Class Example

```python
@proto
class Person:
    """Person configuration."""
    name: str = "Alice"
    age: int = 30
    city: str = "New York"

person = Person()
print(f"{person.name} is {person.age} years old")
```

See [Quick Start](../quick_start.md) for more examples.
