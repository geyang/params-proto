# List[T] CLI Tests - Examples and Debugging Guide

## Quick Test Examples

### Running Individual Tests

```bash
# Test basic List[str] parsing
pytest tests/test_v3/test_cli_parsing.py::test_list_str_cli_parsing -xvs

# Test List[int] parsing with type conversion
pytest tests/test_v3/test_cli_parsing.py::test_list_int_cli_parsing -xvs

# Test help generation (this one passes!)
pytest tests/test_v3/test_cli_parsing.py::test_list_help_strings -xvs
```

## Manual Testing

You can manually test the current behavior with these Python scripts:

### Example 1: List[str] Current Behavior

```python
from typing import List
from params_proto import proto

@proto.cli
def train(tags: List[str] = None, batch_size: int = 32):
    print(f"tags={tags},batch_size={batch_size}")

if __name__ == "__main__":
    train()
```

**Run with**: `python script.py --tags experiment`

**Current output**: `tags=experiment,batch_size=32` ❌
**Expected output**: `tags=['experiment'],batch_size=32` ✓

**Issue**: Single value not wrapped in list

---

### Example 2: List[int] Multiple Values

```python
from typing import List
from params_proto import proto

@proto.cli
def train(gpu_ids: List[int] = None, seed: int = 42):
    print(f"gpu_ids={gpu_ids},seed={seed}")

if __name__ == "__main__":
    train()
```

**Run with**: `python script.py --gpu-ids 0 1 2 --seed 99`

**Current output**: `gpu_ids=0,seed=99` ❌
  (Note: `1` and `2` disappear, become positional args)

**Expected output**: `gpu_ids=[0, 1, 2],seed=99` ✓

**Issues**:
1. Values not converted to integers
2. Only first value consumed
3. Remaining values lost as positional args

---

### Example 3: List[float] Type Conversion

```python
from typing import List
from params_proto import proto

@proto.cli
def train(learning_rates: List[float] = None):
    print(f"learning_rates={learning_rates}")

if __name__ == "__main__":
    train()
```

**Run with**: `python script.py --learning-rates 0.001 0.0001`

**Current output**: `learning_rates=0.001` ❌
  (String "0.001", second value lost)

**Expected output**: `learning_rates=[0.001, 0.0001]` ✓

---

### Example 4: List with Prefix Class

```python
from typing import List
from params_proto import proto

@proto.prefix
class Model:
    layer_sizes: List[int] = [256, 128]

@proto.cli
def train():
    print(f"Model.layer_sizes={Model.layer_sizes}")

if __name__ == "__main__":
    train()
```

**Run with**: `python script.py --model.layer-sizes 512 256 128`

**Current output**: `Model.layer_sizes=512` ❌
  (Only first value, not a list)

**Expected output**: `Model.layer_sizes=[512, 256, 128]` ✓

---

### Example 5: Working - Help Text Generation

```python
from typing import List
from params_proto import proto

@proto.cli
def train(gpu_ids: List[int] = [0], tags: List[str] = ["default"]):
    """Train model with GPUs and tags."""
    pass

if __name__ == "__main__":
    train()
```

**Run with**: `python script.py --help`

**Output**: ✓ WORKS
```
usage: script.py [-h] [--gpu-ids VALUE] [--tags VALUE]

Train model with GPUs and tags.

options:
  -h, --help      show this help message and exit
  --gpu-ids VALUE
                  (default: [0])
  --tags VALUE    (default: ['default'])
```

**This is the only part that works correctly!**

---

## Debug Output Reference

### Debugging CLI Parsing

To debug what's happening in CLI parsing, you can add debug prints:

```python
from typing import List, get_origin, get_args
from params_proto import proto
import sys

@proto.cli
def train(gpu_ids: List[int] = None):
    print(f"gpu_ids={gpu_ids}, type={type(gpu_ids)}")

# Add debug to see what's happening
import params_proto.cli.cli_parse as cli_parse
original_parse = cli_parse.parse_cli_args

def debug_parse(wrapper):
    print(f"DEBUG: sys.argv = {sys.argv}")
    result = original_parse(wrapper)
    print(f"DEBUG: parse result = {result}")
    return result

cli_parse.parse_cli_args = debug_parse

if __name__ == "__main__":
    train()
```

**Run with**: `python script.py --gpu-ids 0 1 2`

---

## Type Conversion Testing

### Current _convert_type Behavior

```python
from params_proto.type_utils import _convert_type
from typing import List

# Works for basic types
print(_convert_type("42", int))  # → 42 ✓
print(_convert_type("3.14", float))  # → 3.14 ✓
print(_convert_type("hello", str))  # → "hello" ✓

# Broken for List types
print(_convert_type("42", List[int]))  # → "42" ❌ (should be [42])
print(_convert_type("3.14", List[float]))  # → "3.14" ❌ (should be [3.14])
```

---

## Expected Fix Locations

### Fix 1: type_utils.py

**Location**: Line 26-27 (after `origin = get_origin(annotation)`)

```python
# CURRENT (doesn't handle List):
return value

# NEEDED (detect List and convert):
if origin is list:
    inner_type = get_args(annotation)[0] if get_args(annotation) else str
    # Ensure value is iterable
    if isinstance(value, str):
        values = [value]  # Single value becomes 1-element list
    else:
        values = value
    return [_convert_type(v, inner_type) for v in values]
```

### Fix 2: cli_parse.py

**Location**: Line 253-260 (the value consumption loop)

```python
# CURRENT (always consumes 1 value):
if i + 1 >= len(args):
    raise SystemExit(f"error: argument --{key} requires a value")
value_str = args[i + 1]
value = _convert_type(value_str, annotation)
result[orig_name] = value
i += 2

# NEEDED (detect List and consume multiple):
# Check if this is a List type
origin = get_origin(annotation)
if origin is list:
    # Collect all following non-flag arguments
    values = []
    j = i + 1
    while j < len(args) and not args[j].startswith('--') and not args[j].startswith('-'):
        values.append(args[j])
        j += 1
    if not values:
        raise SystemExit(f"error: argument --{key} requires at least one value")
    value = _convert_type(values, annotation)
    result[orig_name] = value
    i = j
else:
    # Original single-value handling
    if i + 1 >= len(args):
        raise SystemExit(f"error: argument --{key} requires a value")
    value_str = args[i + 1]
    value = _convert_type(value_str, annotation)
    result[orig_name] = value
    i += 2
```

---

## Test Matrix

This matrix shows which combinations of List parameters are tested:

| Test Name | List[str] | List[int] | List[float] | Defaults | Prefix | Multiple | Single |
|-----------|-----------|-----------|-------------|----------|--------|----------|--------|
| test_list_str_cli_parsing | ✓ | - | - | None | No | ✓ | ✓ |
| test_list_int_cli_parsing | - | ✓ | - | None | No | ✓ | ✓ |
| test_list_float_cli_parsing | - | - | ✓ | None | No | ✓ | ✓ |
| test_list_with_defaults | ✓ | ✓ | ✓ | Non-None | No | ✓ | ✓ |
| test_list_with_prefix_class | ✓ | ✓ | - | Non-None | Yes | ✓ | ✓ |
| test_list_empty_initialization | ✓ | ✓ | - | Empty[] | No | ✓ | ✓ |
| test_list_single_vs_multiple_values | - | ✓ | - | None | No | ✓ | ✓ |
| test_list_help_strings | ✓ | ✓ | - | Non-None | No | No | No |
| test_list_str_whitespace_handling | ✓ | - | - | None | No | ✓ | ✓ |

---

## Common Issues and Solutions

### Issue: "My List parameter is not being set"

**Possible causes**:
1. Value not captured due to single-value consumption
2. Value not converted to list type

**Debug**:
```python
# Add print statement in your CLI function
@proto.cli
def train(gpu_ids: List[int] = None):
    print(f"DEBUG: gpu_ids = {gpu_ids}")
    print(f"DEBUG: type = {type(gpu_ids)}")
```

---

### Issue: "Only first value is captured"

**Root cause**: CLI parser `i += 2` skips over subsequent values

**Example**: With `--gpu-ids 0 1 2`:
1. Parser sees `--gpu-ids`
2. Consumes `0` as value
3. Increments to next position, which is `1`
4. `1` is not a flag (no `--`), so it's treated as positional
5. `2` also becomes positional

**Solution**: Fix must check for List type and consume until next flag

---

### Issue: "Values are strings, not integers"

**Root cause**: `_convert_type()` returns value unmodified for List types

**Example**:
```python
gpu_ids = ["0", "1", "2"]  # ← strings, not integers!
```

**Solution**: Update `_convert_type()` to detect List origin and convert inner types

---

## Performance Notes

The fix should maintain performance:
- Type checking using `get_origin()` is O(1)
- Iterating to next flag is O(n) where n = number of values for this param
- Type conversion is already O(n) for list elements
- No additional memory overhead

---

## Related Tests

These tests verify List[T] works in other contexts (not CLI):

- `tests/test_v3/test_proto_core.py` - Has List[T] in proto class
- `tests/test_v3/test_help_strings.py` - Tests help generation (lines 224-251)

Both of these demonstrate that List[T] works fine when not parsing from CLI.

