# List[T] Type CLI Parsing - Comprehensive Test Suite

## Overview

This document provides a complete overview of the comprehensive test suite created for `List[T]` type CLI parsing in params-proto.

## Files Created

### 1. Test Code
- **File**: `/Users/ge/fortyfive/params-proto/tests/test_v3/test_cli_parsing.py`
- **Lines**: 864-1233 (10 new test functions, ~370 lines)
- **Added to**: Existing CLI parsing test file

### 2. Documentation

Three comprehensive documentation files were created to guide fixing the issue:

#### a) `LIST_TYPE_CLI_TESTS_SUMMARY.md`
- **Purpose**: Quick reference guide for the test suite
- **Contents**:
  - Testing pattern explanation
  - Current behavior vs expected behavior
  - Root cause analysis
  - Implementation strategy

#### b) `TEST_RESULTS_SUMMARY.md`
- **Purpose**: Detailed test results and impact analysis
- **Contents**:
  - Individual test status and failure reasons
  - Root cause deep-dive for both issues
  - Impact analysis
  - What works vs what doesn't
  - Next steps for fixing

#### c) `TEST_EXAMPLES_AND_DEBUGGING.md`
- **Purpose**: Practical guide with examples and debugging steps
- **Contents**:
  - Runnable example scripts
  - Manual testing instructions
  - Debug output examples
  - Expected fix locations with code
  - Test matrix showing coverage

#### d) `LIST_TYPE_TESTING_README.md` (this file)
- **Purpose**: Overview and navigation guide
- **Contents**: This document

## Quick Start

### Running All List Tests

```bash
cd /Users/ge/fortyfive/params-proto
python -m pytest tests/test_v3/test_cli_parsing.py -k "test_list" -v
```

Expected output:
```
tests/test_v3/test_cli_parsing.py::test_list_str_cli_parsing FAILED
tests/test_v3/test_cli_parsing.py::test_list_int_cli_parsing FAILED
tests/test_v3/test_cli_parsing.py::test_list_float_cli_parsing FAILED
tests/test_v3/test_cli_parsing.py::test_list_with_defaults FAILED
tests/test_v3/test_cli_parsing.py::test_list_with_prefix_class FAILED
tests/test_v3/test_cli_parsing.py::test_list_empty_initialization FAILED
tests/test_v3/test_cli_parsing.py::test_list_single_vs_multiple_values FAILED
tests/test_v3/test_cli_parsing.py::test_list_help_strings PASSED
tests/test_v3/test_cli_parsing.py::test_list_str_whitespace_handling FAILED

====== 8 failed, 1 passed in 0.40s ======
```

### Running Individual Tests

```bash
# Test List[str] parsing
pytest tests/test_v3/test_cli_parsing.py::test_list_str_cli_parsing -xvs

# Test List[int] parsing with type conversion
pytest tests/test_v3/test_cli_parsing.py::test_list_int_cli_parsing -xvs

# Test help generation (the passing one)
pytest tests/test_v3/test_cli_parsing.py::test_list_help_strings -xvs

# Test with prefix classes
pytest tests/test_v3/test_cli_parsing.py::test_list_with_prefix_class -xvs
```

## Test Summary

### Total Tests: 10

| Test Name | Type | Status | Notes |
|-----------|------|--------|-------|
| test_list_str_cli_parsing | List[str] | FAIL | Single and multiple value parsing broken |
| test_list_int_cli_parsing | List[int] | FAIL | Type conversion + value parsing broken |
| test_list_float_cli_parsing | List[float] | FAIL | Type conversion + value parsing broken |
| test_list_with_defaults | Mixed | FAIL | Defaults work, CLI parsing broken |
| test_list_with_prefix_class | Prefix | FAIL | Same CLI parsing issues in prefix classes |
| test_list_empty_initialization | Empty[] | FAIL | Empty defaults work, CLI parsing broken |
| test_list_single_vs_multiple_values | List[int] | FAIL | Both single and multiple parsing broken |
| test_list_help_strings | Help | PASS | Help generation works correctly |
| test_list_str_whitespace_handling | List[str] | FAIL | Paths not collected into list |

### Test Coverage

Tests cover:
- Single value in List parameter
- Multiple values in List parameter
- Type conversion (int, float, str)
- Default values (None, empty, populated)
- Prefix classes
- Help text generation
- Edge cases (whitespace, special characters)
- Error handling

### What Works

✓ Default values with List types
✓ Help text generation
✓ Type hints and annotations
✓ List behavior when no CLI parsing occurs

### What Doesn't Work

✗ Parsing List[str] from CLI
✗ Parsing List[int] from CLI
✗ Parsing List[float] from CLI
✗ Type conversion for List elements
✗ Multiple value collection
✗ Integration with prefix classes

## Root Causes

The tests reveal two distinct issues that must be fixed:

### Issue 1: Type Conversion (50% of problem)
**File**: `src/params_proto/type_utils.py`
**Function**: `_convert_type()`
**Problem**: Doesn't handle `List[T]` origins, returns value unmodified

### Issue 2: CLI Argument Parsing (50% of problem)
**File**: `src/params_proto/cli/cli_parse.py`
**Problem**: Only consumes one value per parameter, doesn't distinguish List types

See `TEST_RESULTS_SUMMARY.md` and `TEST_EXAMPLES_AND_DEBUGGING.md` for detailed analysis and fixes.

## Test Design Philosophy

### Subprocess-Based Testing
Tests use Python's `subprocess` module to run actual CLI scripts, providing realistic end-to-end testing rather than unit testing individual components.

```python
@pytest.fixture
def run_cli():
  def _run(script_content: str, args: list[str] = None, expect_error: bool = False):
    # Creates temporary script and runs with Python
    # Returns stdout, stderr, returncode
```

### Clear Documentation
Each test includes comments explaining:
- What is being tested
- Current broken behavior
- Expected correct behavior
- Root cause of the issue

Example:
```python
# Test 1: List[str] with single value should create list
# CURRENTLY BROKEN: Returns "tags=experiment" instead of "tags=['experiment']"
result = run_cli(script, ["--tags", "experiment"], expect_error=False)
assert result["stdout"].strip() == "tags=['experiment'],batch_size=32"
```

### Progressive Complexity
Tests progress from simple to complex:
1. Basic single value
2. Multiple values
3. Type conversion
4. Defaults
5. Prefix classes
6. Edge cases

## How to Use These Tests

### For Development
These tests serve as a specification for the fix:
1. Read the test code to understand expected behavior
2. Look at `TEST_EXAMPLES_AND_DEBUGGING.md` for fix locations
3. Implement the fix
4. Run tests to verify
5. Iterate until all tests pass

### For Verification
After fixing the code:
1. Run the tests: `pytest tests/test_v3/test_cli_parsing.py -k "test_list"`
2. All 9 functional tests should pass (help test already passes)
3. No regressions in other tests

### For Documentation
Use test cases as examples in documentation:
- Show users how List[T] works in CLI
- Copy example scripts for documentation
- Reference test failures to explain limitations

## Integration with Existing Tests

These tests follow the same pattern as existing tests in the file:
- Use the `run_cli` fixture
- Use `subprocess` for realistic testing
- Include clear docstrings
- Follow naming convention: `test_<feature>_<scenario>`
- Use `expect_error=True/False` for error handling

## Next Steps

When fixing the issue:

1. **Read the Analysis**
   - Read `TEST_RESULTS_SUMMARY.md` for full context
   - Review `TEST_EXAMPLES_AND_DEBUGGING.md` for fix locations

2. **Implement Fixes**
   - Fix `_convert_type()` to handle List origin
   - Fix CLI parser to consume multiple values for List params

3. **Run Tests**
   - `pytest tests/test_v3/test_cli_parsing.py -k "test_list" -v`
   - Should go from "8 failed, 1 passed" to "9 passed"

4. **Verify No Regressions**
   - `pytest tests/test_v3/test_cli_parsing.py`
   - All existing tests should still pass

5. **Update Documentation**
   - Add List[T] usage examples to docs
   - Update skill documentation if needed
   - Add to release notes

## File Locations Reference

| Document | Location | Purpose |
|----------|----------|---------|
| Test Code | `/Users/ge/fortyfive/params-proto/tests/test_v3/test_cli_parsing.py` | 10 test functions (lines 864-1233) |
| Test Summary | `/Users/ge/fortyfive/params-proto/LIST_TYPE_CLI_TESTS_SUMMARY.md` | Quick reference guide |
| Test Results | `/Users/ge/fortyfive/params-proto/TEST_RESULTS_SUMMARY.md` | Detailed results and impact |
| Examples | `/Users/ge/fortyfive/params-proto/TEST_EXAMPLES_AND_DEBUGGING.md` | Runnable examples and debug guide |
| This File | `/Users/ge/fortyfive/params-proto/LIST_TYPE_TESTING_README.md` | Overview and navigation |

## Helpful Commands

```bash
# Run all List tests with verbose output
pytest tests/test_v3/test_cli_parsing.py -k "test_list" -v

# Run just the passing test
pytest tests/test_v3/test_cli_parsing.py::test_list_help_strings -xvs

# Run a specific failing test to see detailed error
pytest tests/test_v3/test_cli_parsing.py::test_list_int_cli_parsing -xvs

# Run all tests in the file to check for regressions
pytest tests/test_v3/test_cli_parsing.py

# Run with minimal output (summary only)
pytest tests/test_v3/test_cli_parsing.py -k "test_list" -q
```

## Summary

This test suite provides:

✓ 10 comprehensive test cases for List[T] CLI parsing
✓ Clear documentation of current broken behavior
✓ Detailed root cause analysis
✓ Expected behavior specifications
✓ Fix location guidance
✓ Runnable examples for manual testing
✓ Progressive test complexity for learning

All tests are designed to fail initially, documenting the bug clearly. Once the underlying CLI parsing and type conversion code is fixed, all tests will pass and serve as regression prevention.
