# List[T] Type CLI Parsing Tests - Complete Index

## Overview

Comprehensive test suite for `List[T]` type CLI parsing in params-proto. This index provides quick navigation to all test code and documentation.

**Status**: 8 FAIL, 1 PASS (out of 9 functional tests)

## Files Created

### Test Code

**File**: `/Users/ge/fortyfive/params-proto/tests/test_v3/test_cli_parsing.py`
- Lines: 864-1233
- Functions: 10 test functions
- Approach: Subprocess-based integration testing

### Documentation Files

| File | Size | Purpose | Key Content |
|------|------|---------|-------------|
| `LIST_TYPE_CLI_TESTS_SUMMARY.md` | 6.3K | Quick reference | Testing patterns, root causes, implementation strategy |
| `TEST_RESULTS_SUMMARY.md` | 11K | Detailed results | Individual test breakdown, impact analysis, what works/fails |
| `TEST_EXAMPLES_AND_DEBUGGING.md` | 8.6K | Practical guide | 5 runnable examples, debug techniques, fix locations |
| `LIST_TYPE_TESTING_README.md` | 9.2K | Navigation guide | Quick start, test matrix, integration notes |
| `LIST_TYPE_TESTS_INDEX.md` | This file | Complete index | File locations and content summary |

## Test Functions (10 total)

### 1. test_list_str_cli_parsing
**File**: Line 867
**Status**: FAIL ❌
**Tests**: List[str] with single and multiple values
**Issues**: Values not wrapped in list, only first value captured

### 2. test_list_int_cli_parsing
**File**: Line 918
**Status**: FAIL ❌
**Tests**: List[int] with type conversion
**Issues**: Values not converted to int, only first value captured

### 3. test_list_float_cli_parsing
**File**: Line 974
**Status**: FAIL ❌
**Tests**: List[float] with type conversion
**Issues**: Values not converted to float, only first value captured

### 4. test_list_with_defaults
**File**: Line 1022
**Status**: FAIL ❌
**Tests**: List with non-None default values
**Issues**: Defaults work, but CLI override fails

### 5. test_list_with_prefix_class
**File**: Line 1072
**Status**: FAIL ❌
**Tests**: List in @proto.prefix classes
**Issues**: Same CLI parsing issues in prefix context

### 6. test_list_empty_initialization
**File**: Line 1118
**Status**: FAIL ❌
**Tests**: List initialized as empty []
**Issues**: Empty defaults work, CLI override fails

### 7. test_list_single_vs_multiple_values
**File**: Line 1148
**Status**: FAIL ❌
**Tests**: Single vs multiple value distinction
**Issues**: Both single and multiple values broken

### 8. test_list_help_strings
**File**: Line 1177
**Status**: PASS ✓
**Tests**: Help text generation
**Result**: Help generation works correctly

### 9. test_list_str_whitespace_handling
**File**: Line 1206
**Status**: FAIL ❌
**Tests**: List[str] with special characters/paths
**Issues**: Multiple values not collected, not wrapped in list

## Root Cause Map

### Issue 1: Type Conversion Broken
**Location**: `src/params_proto/type_utils.py` (lines 12-43)
**Function**: `_convert_type(value, annotation)`
**Problem**: Doesn't detect List origin, returns value unmodified
**Impact**: Values stay as strings instead of becoming typed lists

**Detailed in**: `TEST_RESULTS_SUMMARY.md` (lines 290-340)
**Fix shown in**: `TEST_EXAMPLES_AND_DEBUGGING.md` (lines 260-275)

### Issue 2: CLI Argument Parsing Broken
**Location**: `src/params_proto/cli/cli_parse.py` (lines 241-261)
**Function**: Value consumption logic
**Problem**: Always consumes exactly 1 value, doesn't distinguish List types
**Impact**: Multiple values lost, treated as positional arguments

**Detailed in**: `TEST_RESULTS_SUMMARY.md` (lines 341-390)
**Fix shown in**: `TEST_EXAMPLES_AND_DEBUGGING.md` (lines 276-320)

## What Works vs What Doesn't

### Works
- List parameters with default values (no CLI parsing)
- Help text generation for List parameters
- Type annotations and hints
- List behavior when no CLI parsing occurs

### Doesn't Work
- Parsing List[str] values from CLI
- Parsing List[int] values from CLI
- Parsing List[float] values from CLI
- Type conversion for List element types
- Multiple value collection
- Integration with prefix classes

**Summary**: List[T] completely non-functional for CLI input

## Test Execution

### Run All List Tests
```bash
pytest tests/test_v3/test_cli_parsing.py -k "test_list" -v
```
Expected: 8 failed, 1 passed

### Run Individual Test
```bash
pytest tests/test_v3/test_cli_parsing.py::test_list_int_cli_parsing -xvs
```

### Run Only Passing Test
```bash
pytest tests/test_v3/test_cli_parsing.py::test_list_help_strings -xvs
```

## Documentation Navigation

### Start Here
1. Read this file (overall picture)
2. Read `LIST_TYPE_TESTING_README.md` (navigation and quick start)

### Learn the Issue
3. Read `LIST_TYPE_CLI_TESTS_SUMMARY.md` (quick reference)
4. Read `TEST_RESULTS_SUMMARY.md` (detailed analysis)

### Understand the Fix
5. Read `TEST_EXAMPLES_AND_DEBUGGING.md` (examples and fix locations)
6. Look at test code in `test_cli_parsing.py` (lines 864-1233)

## Key Statistics

- **Test functions created**: 10
- **Lines of test code**: ~370
- **Test functions passing**: 1
- **Test functions failing**: 8
- **Root cause issues**: 2
- **Files affected by bug**: 2
- **Documentation files**: 4
- **Example scripts provided**: 5
- **Severity level**: HIGH (feature completely broken)

## Example Usage

### Current Broken Behavior

```python
# Example 1: Single value not wrapped
python -c "
from typing import List
from params_proto import proto

@proto.cli
def main(tags: List[str] = None):
    print(f'tags={tags}')
" --tags experiment

# OUTPUT: tags=experiment
# EXPECT: tags=['experiment']
```

```python
# Example 2: Multiple values not captured
python -c "
from typing import List
from params_proto import proto

@proto.cli
def main(ids: List[int] = None):
    print(f'ids={ids}')
" --ids 0 1 2

# OUTPUT: ids=0
# EXPECT: ids=[0, 1, 2]
```

All examples provided in `TEST_EXAMPLES_AND_DEBUGGING.md`

## Implementation Strategy

1. **Understand the issue**
   - Read `TEST_RESULTS_SUMMARY.md`
   - Review test failures

2. **Locate the fixes needed**
   - `TEST_EXAMPLES_AND_DEBUGGING.md` shows exact line numbers
   - Code snippets provided for both fixes

3. **Implement fixes**
   - Fix `_convert_type()` in `type_utils.py`
   - Fix argument parsing in `cli_parse.py`

4. **Verify with tests**
   - Run tests: `pytest tests/test_v3/test_cli_parsing.py -k "test_list" -v`
   - Should go from "8 failed, 1 passed" to "9 passed"

5. **Check for regressions**
   - Run all CLI tests: `pytest tests/test_v3/test_cli_parsing.py`

## File Relationships

```
tests/test_v3/test_cli_parsing.py (lines 864-1233)
    ├─ test_list_str_cli_parsing
    ├─ test_list_int_cli_parsing
    ├─ test_list_float_cli_parsing
    ├─ test_list_with_defaults
    ├─ test_list_with_prefix_class
    ├─ test_list_empty_initialization
    ├─ test_list_single_vs_multiple_values
    ├─ test_list_help_strings ✓
    └─ test_list_str_whitespace_handling

Documentation/
    ├─ LIST_TYPE_TESTS_INDEX.md (this file)
    ├─ LIST_TYPE_TESTING_README.md (navigation)
    ├─ LIST_TYPE_CLI_TESTS_SUMMARY.md (quick ref)
    ├─ TEST_RESULTS_SUMMARY.md (detailed results)
    └─ TEST_EXAMPLES_AND_DEBUGGING.md (practical guide)

Source Code (needs fixes)
    ├─ src/params_proto/type_utils.py (Issue 1)
    └─ src/params_proto/cli/cli_parse.py (Issue 2)
```

## Quick Commands Reference

| Task | Command |
|------|---------|
| Run all List tests | `pytest tests/test_v3/test_cli_parsing.py -k "test_list" -v` |
| Run specific test | `pytest tests/test_v3/test_cli_parsing.py::test_list_int_cli_parsing -xvs` |
| Run help test (pass) | `pytest tests/test_v3/test_cli_parsing.py::test_list_help_strings -xvs` |
| View test code | `sed -n '864,1233p' tests/test_v3/test_cli_parsing.py` |
| Quick summary | `cat LIST_TYPE_CLI_TESTS_SUMMARY.md` |
| Detailed analysis | `cat TEST_RESULTS_SUMMARY.md` |
| Examples and fixes | `cat TEST_EXAMPLES_AND_DEBUGGING.md` |

## Summary

This comprehensive test suite provides:

1. **10 test functions** covering all aspects of List[T] CLI parsing
2. **Clear documentation** explaining current broken behavior
3. **Detailed root cause analysis** with fix locations
4. **Runnable examples** for manual testing
5. **Implementation guidance** for fixing the issue

All tests are designed to fail initially, documenting the bug clearly. Once the underlying code is fixed, all tests will pass and serve as regression prevention.

The test suite is ready for:
- Development and debugging
- Implementation of the fix
- Verification and validation
- Documentation and communication

---

**Last Updated**: Created as comprehensive test suite for List[T] CLI parsing
**Status**: All tests created and running (8 FAIL, 1 PASS)
**Next Action**: Implement fixes in type_utils.py and cli_parse.py
