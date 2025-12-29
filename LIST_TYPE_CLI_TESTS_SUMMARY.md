# List[T] Type CLI Parsing Tests - Summary

## Overview
Comprehensive test suite for `List[T]` type CLI parsing in params-proto. These tests document the current behavior and provide assertions for when the bug is fixed.

## Test Location
- File: `/Users/ge/fortyfive/params-proto/tests/test_v3/test_cli_parsing.py`
- Tests: Lines 864-1188 (10 new test functions)

## Current Behavior vs Expected

### What Works Currently ✓
1. **Default values with List types** - List parameters with default values work correctly
   - `gpu_ids: List[int] = [0, 1]` → correctly parsed as `[0, 1]`
   - `tags: List[str] = ["default"]` → correctly parsed as `["default"]`
   - Empty lists: `ids: List[int] = []` → correctly parsed as `[]`

2. **Help text generation** - List parameters appear in help output
   - `--gpu-ids VALUE` displays correctly with default shown
   - Documentation strings are preserved

### What Fails Currently ✗
1. **CLI parsing of List[str]**
   - **Issue**: Single value not wrapped in list
   - **Example**: `--tags experiment` → `tags=experiment` (should be `tags=['experiment']`)
   - **Test**: `test_list_str_cli_parsing` (Test 1)

2. **CLI parsing of List[int]**
   - **Issue**: Single value not converted to integer and wrapped in list
   - **Example**: `--gpu-ids 0` → `gpu_ids=0` (should be `gpu_ids=[0]`)
   - **Test**: `test_list_int_cli_parsing` (Test 1)

3. **CLI parsing of List[float]**
   - **Issue**: Single value not converted to float and wrapped in list
   - **Example**: `--learning-rates 0.001` → returns string (should be `learning_rates=[0.001]`)
   - **Test**: `test_list_float_cli_parsing` (Test 1)

4. **Multiple values in List**
   - **Issue**: When passing multiple values like `--gpu-ids 0 1 2`, only the first value is captured
   - **Expected**: `gpu_ids=[0, 1, 2]`
   - **Actual**: `gpu_ids=0` (second and subsequent values treated as positional args)
   - **Test**: `test_list_int_cli_parsing` (Test 2), `test_list_str_cli_parsing` (Test 2)

5. **Type conversion in lists**
   - **Issue**: `_convert_type()` doesn't handle `List[int]`, `List[float]`, etc.
   - **Location**: `/Users/ge/fortyfive/params-proto/src/params_proto/type_utils.py`
   - **Current code**: Only handles basic types (int, float, str, bool)
   - **Test**: All List parsing tests (Test 1-4 for each type)

6. **CLI argument parsing for multiple values**
   - **Issue**: CLI parser in `cli_parse.py` consumes only one value after `--param`
   - **Location**: `/Users/ge/fortyfive/params-proto/src/params_proto/cli/cli_parse.py` (lines 253-260)
   - **Current behavior**: Gets next arg as single value, increments counter by 2
   - **Needed change**: Should check if annotation is `List[T]` and consume multiple following args
   - **Test**: All "multiple values" test cases

## Test Cases (10 functions)

### 1. test_list_str_cli_parsing
Tests `List[str]` parameter parsing with various scenarios.
- Single value parsing
- Multiple values parsing
- Default None behavior
- Mixed with other parameters

### 2. test_list_int_cli_parsing
Tests `List[int]` parameter parsing with type conversion.
- Single integer value
- Multiple integer values
- Error handling for non-integer values
- Mixed with other parameters

### 3. test_list_float_cli_parsing
Tests `List[float]` parameter parsing with float conversion.
- Single float value
- Multiple float values (including scientific notation)
- Mixed with other parameters

### 4. test_list_with_defaults
Tests List parameters with non-None default values.
- Using default values
- Overriding individual list types
- Overriding all list parameters

### 5. test_list_with_prefix_class
Tests List parameters in `@proto.prefix` classes.
- Default values in prefix classes
- Overriding single list in prefix (e.g., `--model.layer-sizes 512 256`)
- Overriding multiple lists in prefix

### 6. test_list_empty_initialization
Tests List parameters initialized as empty lists.
- Default empty list behavior
- Overriding empty lists with values

### 7. test_list_single_vs_multiple_values
Tests the distinction between single and multiple values.
- Single value should create 1-element list
- Multiple values should create n-element list

### 8. test_list_help_strings
Tests that List parameters display correctly in help output.
- Parameter names appear with correct format
- Default values are shown

### 9. test_list_str_whitespace_handling
Tests that List[str] handles special characters and paths.
- File paths with slashes
- Multiple paths
- Single paths

## Root Cause Analysis

### Issue 1: Type Conversion (`type_utils.py`)
The `_convert_type()` function (line 12-43) only handles basic types:
- `int`, `float`, `str`, `bool`
- Doesn't check `get_origin()` for generic types like `List[T]`
- Returns value as-is for unhandled types (Line 42)

**Fix needed**: Detect `List[T]` origin and recursively convert inner types

### Issue 2: CLI Argument Parser (`cli_parse.py`)
The argument parsing logic (lines 241-260) for regular parameters:
- Checks if parameter is in `param_types` dict
- Gets next single argument: `value_str = args[i + 1]`
- Converts using `_convert_type(value_str, annotation)`
- Increments by 2: `i += 2`

**Problem**: Doesn't distinguish between `List[T]` and scalar types
**Fix needed**:
- Check if annotation is `List[T]` type
- If so, consume all following arguments until next flag (starts with `--` or `-`)
- Collect into list before type conversion

## Files Modified
- `/Users/ge/fortyfive/params-proto/tests/test_v3/test_cli_parsing.py`
  - Added 10 new test functions (lines 864-1188)
  - Follows existing test pattern with `run_cli` fixture

## Test Status
All 10 tests are designed to **FAIL** initially, documenting current broken behavior. They will pass once the CLI parsing and type conversion code is fixed.

## Implementation Strategy
When fixing, follow this order:
1. Update `_convert_type()` to handle `List[T]` origins
2. Update `cli_parse.py` to detect List parameters and consume multiple args
3. Handle edge cases:
   - Empty lists
   - Single values (should become 1-element lists)
   - Type errors in list elements
   - Mixing with other CLI features (prefix, unions, etc.)

## Notes
- Tests use subprocess to run actual CLI commands (realistic testing)
- Each test has comments explaining the expected vs current behavior
- Tests are compatible with existing test infrastructure
- Pattern matches Optional[T] tests for consistency
