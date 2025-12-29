# Release Notes

This page contains the release history and changelog for params-proto.

## Version 3.0.0-rc23 (2025-12-29)

### ✨ Features

- **Tuple[T, ...] CLI Parsing**: Full support for variable-length and fixed-size tuples
  - Variable-length tuples: `Tuple[int, ...]` collects values into tuple with consistent type
  - Fixed-size tuples: `Tuple[int, str, float]` with mixed types at each position
  - CLI collects multiple values: `python script.py --sizes 256 512 1024` → `sizes=(256, 512, 1024)`
  - Automatic element/position type conversion with type safety
  - Works with defaults and `@proto.prefix` classes
  - Help text shows tuple notation: `(INT,...)` or `(INT,STR,FLOAT)`

  **Implementation:**
  - Updated `_convert_type()` in `type_utils.py` to handle both variable and fixed-size tuples
  - Updated CLI parser in `cli_parse.py` to collect multiple values for Tuple parameters
  - Updated `_get_type_name()` to display tuple signatures in help text

  **Test Suite**: All 9 comprehensive test cases now PASSING in `tests/test_v3/test_cli_parsing.py`:
  - ✅ `test_tuple_variable_length_int` - Variable-length integer tuples
  - ✅ `test_tuple_variable_length_float` - Variable-length float tuples
  - ✅ `test_tuple_variable_length_str` - Variable-length string tuples
  - ✅ `test_tuple_fixed_size_mixed` - Fixed-size with mixed types (int, str, float)
  - ✅ `test_tuple_with_defaults` - Overriding tuple defaults
  - ✅ `test_tuple_single_value` - Single value wrapped in tuple
  - ✅ `test_tuple_empty_initialization` - Empty tuple defaults
  - ✅ `test_tuple_with_prefix_class` - Tuples in @proto.prefix classes
  - ✅ `test_tuple_help_strings` - Help text generation

### 📋 Type System Updates

- **Type Support Matrix**: Updated to show `Tuple[T, ...]` as ✅ Full support
- **Documentation**: Comprehensive guide for tuple usage with examples and CLI patterns
- **Total Fully Working Types**: int, float, str, bool, Optional[T], List[T], Tuple[T, ...], Union types

---

## Version 3.0.0-rc22 (2025-12-29)

### 📚 Documentation

- **List[T] CLI Parsing Documentation**: Comprehensive guide for using list types
  - Updated `docs/key_concepts/type-system.md` with practical examples
  - Added section showing CLI usage patterns: `--items a b c` → list of values
  - Help text generation examples showing `[STR]`, `[INT]`, `[FLOAT]` notation
  - Explanation of how multiple values are collected until next flag
  - Type support matrix updated to show List[T] as ✅ Full support

---

## Version 3.0.0-rc21 (2025-12-29)

### ✨ Features

- **List[T] CLI Parsing**: Full support for `List[str]`, `List[int]`, `List[float]` and other list types
  - CLI collects multiple values: `python script.py --items a b c` → `items=['a', 'b', 'c']`
  - Automatic element type conversion: `python script.py --counts 1 2 3` → `counts=[1, 2, 3]`
  - Works with defaults: `--items x y z` overrides `items: List[str] = ["default"]`
  - Help text shows element type: `--items [STR]`, `--counts [INT]`, `--ratios [FLOAT]`
  - Support for List parameters in `@proto.prefix` classes

  **Implementation:**
  - Updated `_convert_type()` in `type_utils.py` to handle generic List[T] types
  - Updated CLI parser in `cli_parse.py` to collect multiple CLI arguments for List parameters
  - Updated `_get_type_name()` to display `[INT]`, `[STR]`, `[FLOAT]` in help text

  **Test Suite**: All 9 comprehensive test cases now PASSING in `tests/test_v3/test_cli_parsing.py`:
  - ✅ `test_list_str_cli_parsing` - Multiple string values
  - ✅ `test_list_int_cli_parsing` - Multiple integers with type conversion
  - ✅ `test_list_float_cli_parsing` - Multiple floats
  - ✅ `test_list_with_defaults` - Overriding list defaults
  - ✅ `test_list_with_prefix_class` - List in @proto.prefix classes
  - ✅ `test_list_empty_initialization` - Empty list defaults
  - ✅ `test_list_single_vs_multiple_values` - Single value wrapped in list
  - ✅ `test_list_help_strings` - Help text generation
  - ✅ `test_list_str_whitespace_handling` - Paths and special characters

### 📋 Documentation & Known Issues

- **Type System Documentation**: Updated type support matrix to accurately reflect CLI parsing status
  - ✅ Fully working: int, float, str, bool, Optional[T], List[T], Union[Class, Class], dataclass unions
  - ⚠️ Partial: Literal[...], Enum (help text works, no runtime conversion)
  - ❌ Broken: Tuple[T, ...], Path, dict (collection types)

- **Path Type Issue**: Documented that Path type annotation is not converted from strings
  - Help text shows correctly, but CLI strings are not wrapped in Path objects

- **Enhanced Union Types Documentation**:
  - Added "Why Union Types Matter" section in union_types.md
  - Reorganized examples to show CLI usage first, then implementation
  - Clarified that Union types are a core feature for multi-way dispatching

**Migration Notes:**
- Workaround for Path parameters: Accept string and convert in function body

---

## Version 3.0.0-rc20 (2025-12-28)

### 🐛 Bug Fixes

- **Optional[T] CLI Parsing**: Fixed `Optional[str]`, `Optional[int]`, and other `Optional[T]` types failing to parse correctly in CLI

  The issue: `Optional[T]` types were incorrectly treated as Union subcommands, requiring special syntax instead of working as simple optional parameters.

  **Before (v3.0.0-rc19):**
  ```python
  from typing import Optional
  from params_proto import proto

  @proto.cli
  def train(checkpoint: Optional[str] = None, learning_rate: float = 0.001):
      print(f"checkpoint={checkpoint}, lr={learning_rate}")
  ```

  ❌ This would fail:
  ```bash
  python train.py --checkpoint model.pt
  # error: unrecognized argument: --checkpoint
  ```

  ✅ **After (v3.0.0-rc20):**
  ```bash
  # Now works correctly with standard syntax
  python train.py --checkpoint model.pt
  # Output: checkpoint=model.pt, lr=0.001

  # Still supports omitting the optional parameter
  python train.py
  # Output: checkpoint=None, lr=0.001

  # Works seamlessly with other parameters
  python train.py --checkpoint model.pt --learning-rate 0.01
  # Output: checkpoint=model.pt, lr=0.01
  ```

  **Improved help output:**
  ```
  Before:  --checkpoint VALUE   Path to checkpoint file
  After:   --checkpoint STR     Path to checkpoint file
  ```

  **Technical fixes:**
  - Fixed `cli_parse.py:_get_union_classes()`: Now filters out `NoneType` from Union type arguments
  - Fixed `cli_parse.py` Union handling: Added detection for `Optional[T]` patterns (Union with single non-None type) and treats them as regular optional parameters
  - Fixed `type_utils.py:_get_type_name()`: Now recognizes `Optional[T]` patterns and recursively extracts the correct inner type name for help text
  - Help text now shows specific types (`STR`, `INT`, `FLOAT`) instead of generic `VALUE` for Optional types

## Version 3.0.0-rc19 (2025-12-28)

### 🧪 Testing

- Added comprehensive test cases for `Optional[str]` and `Optional[int]` CLI parsing
  - Documents current limitation where `Optional[T]` types don't parse correctly with normal `--param value` syntax
  - Tests verify expected behavior once the issue is fixed

### 📚 Documentation

- **New**: Created dedicated `Union Types` documentation page (`docs/key_concepts/union_types.md`)
  - Quick reference with 3 common patterns (Union selection, single class, optional parameters)
  - Clear distinction between `Union[ClassA, ClassB]` and `Optional[T]`
  - Detailed examples and syntax variations
  - Documents `Optional[str]` limitation and workaround

- **Refactored**: Streamlined `cli_guide.md` to reduce verbosity
  - Moved verbose Union/Optional explanation to dedicated `union_types.md`
  - Replaced with concise 3-line reference for quick navigation
  - Maintains clarity while keeping main guide focused

## Version 3.0.0-rc18 (2025-12-26)

### 🐛 Bug Fixes

- **EnvVar Instantiation Fix**: Fixed `@proto.prefix` classes with EnvVar fields failing on instantiation
  with `AttributeError: '_EnvVar' object has no attribute '__get__'`.

  The bug occurred because `_EnvVar` is callable (has `__call__`), so it was incorrectly detected as a
  method during instance creation. The fix uses precise method detection with `inspect.isfunction` and
  `inspect.ismethod` instead of the overly broad `callable()` check.

  ```python
  from params_proto import proto, EnvVar

  @proto.prefix
  class Config:
      host: str = EnvVar @ "HOST" | "localhost"

  # Before fix: AttributeError: '_EnvVar' object has no attribute '__get__'
  # After fix: works correctly
  c = Config()
  ```

### 📚 Documentation

- Added EnvVar + inheritance documentation and tests
- Documented that inherited EnvVar fields are resolved and type-converted correctly

---

## Version 3.0.0-rc17 (2025-12-26)

### ✨ Features

- **Inherited Fields**: Support inherited fields in `@proto` classes. Parent class fields are now
  properly included in `vars()`, CLI args, and work with EnvVar.

  ```python
  class BaseConfig:
      host: str = EnvVar @ "HOST" | "localhost"
      port: int = EnvVar @ "PORT" | 8080

  @proto.prefix
  class AppConfig(BaseConfig):
      debug: bool = EnvVar @ "DEBUG" | False
  ```

---

## Version 3.0.0-rc16 (2025-12-26)

### 🐛 Bug Fixes

- **Python 3.10 Support**: Use `Union[]` syntax instead of `|` operator for type hints to support
  Python 3.10 (#17).

---

## Version 3.0.0-rc15 (2025-12-19)

### 🐛 Bug Fixes

- **EnvVar dtype Conversion**: Fixed `EnvVar.get()` to apply the `dtype` parameter for type conversion.
  Previously, the `dtype` was stored but not used when reading from environment variables, causing
  values to always be returned as strings.

  ```python
  import os
  from params_proto import EnvVar

  os.environ["PORT"] = "9000"

  # Before fix: returned '9000' (str)
  # After fix: returns 9000 (int)
  port = EnvVar("PORT", dtype=int, default=8012).get()
  ```

  All dtypes are now properly applied: `int`, `float`, `bool`, and `str`.

---

## Version 3.0.0-rc10 (2025-12-17)

### 🐛 Bug Fixes

- **Classmethod/Staticmethod Support**: Fixed `@proto`, `@proto.cli`, and `@proto.partial` decorators
  to properly handle `@classmethod` and `@staticmethod` descriptors. Previously, decorating methods
  with `@proto` would incorrectly include `self`/`cls` in CLI parameters or corrupt method calls.

  **Correct decorator order** (proto decorator on the OUTSIDE):
  ```python
  class Trainer:
      @proto.cli      # proto.cli on OUTSIDE receives the descriptor
      @classmethod
      def train(cls, lr: float = 0.01):
          return cls.run_training(lr)

      @proto.cli
      @staticmethod
      def evaluate(model_path: str):
          return load_and_eval(model_path)
  ```

  The decorators now:
  - Detect `classmethod`/`staticmethod` descriptors via `isinstance()`
  - Properly unwrap to get the underlying function signature
  - Exclude `cls` parameter for classmethods automatically
  - Implement descriptor protocol (`__get__`) for proper method binding
  - Re-wrap results in `classmethod()`/`staticmethod()` for `proto.partial`

- **VAR_POSITIONAL/VAR_KEYWORD Handling**: `*args` and `**kwargs` parameters are now properly
  excluded from CLI parameters by checking `inspect.Parameter.kind` instead of just parameter names.

---

## Version 3.0.0-rc7 (2025-12-16)

### ✨ Features

- **Claude Skill**: Added hierarchical Claude skill documentation in `skill/` directory.
  Provides AI assistants with comprehensive params-proto knowledge including:
  - Quick reference and cheat sheet
  - API documentation for all decorators
  - Feature guides (help generation, environment variables, sweeps)
  - Common patterns and examples

---

## Version 3.0.0-rc6 (2025-12-16)

### 🐛 Bug Fixes

- **Boolean Type Display**: Boolean flags now show `BOOL` type in help text for consistency
  with other types (INT, STR, FLOAT).
  ```
  --verbose BOOL       Enable verbose output (default: False)
  --no-cuda BOOL       Use CUDA acceleration (default: True)
  ```

---

## Version 3.0.0-rc5 (2025-12-16)

### 🐛 Bug Fixes

- **Boolean Flag Help Text**: Fixed help text for boolean flags with `default=True`.
  Previously, `--flag` was shown in help even when the flag defaulted to True (making `--flag` a no-op).
  Now shows `--no-flag` for booleans defaulting to True, making it clear how to disable the feature.
  ```python
  @proto.cli
  def train(cuda: bool = True):  # Use CUDA acceleration
      ...
  ```
  Previously: `--cuda            Use CUDA acceleration (default: True)`
  Now:        `--no-cuda         Use CUDA acceleration (default: True)`

- **ANSI Help Colorization**: Fixed regex that incorrectly colored the first word of boolean
  flag descriptions as a type. Now only uppercase type names (INT, STR, FLOAT, BOOL) and enum choices
  (`{A,B,C}`) are colorized as types.

---

## Version 3.0.0-rc4 (2025-12-14)

### 🐛 Bug Fixes

- **EnvVar Class Resolution**: Fixed EnvVar not resolving at import time for class-based configs.
  Previously, accessing `Config.ip` on a `@proto.prefix` class with `ip: str = EnvVar @ "VAR" | "default"`
  would return the `EnvVar` object instead of the resolved value. Now EnvVar values are resolved at
  decoration time for both functions and classes.

---

## Version 3.0.0 (Upcoming)

### 🎉 Major Release: Complete v3 Redesign

params-proto v3 is a complete rewrite focused on simplicity and modern Python type hints.

### ✨ Key Features

#### Documentation & Help Generation
- **Multi-line Documentation**: Inline comments and docstring Args are now concatenated on separate lines for better readability
  ```python
  @proto.cli
  def train(
      batch_size: int = 32,  # Training batch size
  ):
      """Args:
          batch_size: Controls memory usage and gradient noise
      """
  ```
  Generates:
  ```
  --batch-size INT     Training batch size
                       Controls memory usage and gradient noise (default: 32)
  ```
- **Automatic Help Generation**: Parse inline comments (`#`) for parameter documentation
- **Docstring Args Support**: Extract parameter docs from Google/NumPy-style docstrings
- **Smart Deduplication**: Identical inline and docstring descriptions are shown only once
- **Auto-generated Descriptions**: Fallback descriptions generated from parameter names

#### Environment Variables
- **EnvVar Support**: Read configuration from environment variables with type conversion
- **Pipe Operator Syntax**: Clean default value syntax: `EnvVar @ "VAR" | default`
- **Template Expansion**: Support for `$VAR`, `${VAR}`, and multiple variable substitution
- **Type Conversion**: Automatic conversion of env var strings to annotated types

#### CLI Improvements
- **Function CLI**: Decorate functions with `@proto.cli` for instant CLI programs
- **prog Parameter**: Override script name for predictable help output in tests
- **Rich Type Support**: Optional, List, Literal, Tuple, Enum, Path, Union types
- **Multi-namespace Configs**: Use `@proto.prefix` for organized, modular configurations

### 🔄 API Changes
- **Decorator-based**: `@proto` and `@proto.cli` instead of class inheritance
- **Type Hints Required**: Full type hint support for IDE integration
- **Simplified Singleton**: `@proto.prefix` for singleton configs
- **Function Support**: First-class support for function-based configs

### 📚 Documentation
- **Complete Rewrite**: All documentation updated for v3 API
- **Comprehensive Examples**: Real-world ML training and RL agent examples
- **Migration Guide**: Step-by-step guide for upgrading from v2
- **Type System Guide**: Complete documentation of supported type hints

---

## Version 2.13.2 (2025-08-03)

### 📚 Documentation
- **Major**: Added comprehensive Sphinx documentation site with Furo theme
- **Added**: Complete API documentation for proto, hyper, and utils modules
- **Added**: Extensive tutorial collection covering:
  - Basic usage patterns and CLI integration
  - Advanced features including dynamic configs and validation
  - Environment variables for flexible deployment
  - Nested configurations for complex applications
  - Hyperparameter sweeps and experiment management
- **Added**: Read the Docs integration with automatic builds
- **Updated**: Repository moved to `geyang/params-proto` with main branch as default

### 🔧 Configuration Management
- **Added**: Documentation build targets to main Makefile (`make docs`, `make preview`)
- **Added**: `.readthedocs.yaml` configuration for automated documentation builds

---

## Version 2.13.0 (2025-01-15)

### ✨ Features
- **Added**: Environment name checking to ensure all env names are defined in env string
- **Fixed**: Dollar-sign handling in environment variable strings
- **Improved**: Code formatting with ruff configuration
- **Removed**: `typing` dependency (no longer needed for Python 3.8+)

### 🧹 Maintenance
- **Added**: Comprehensive ruff configuration for code formatting
- **Added**: PyCharm/IntelliJ IDE configuration files
- **Updated**: Setup.py dependency management
- **Improved**: Test coverage for environment variable parsing

---

## Version 2.12.1 (2024-04-20)

### 🐛 Bug Fixes
- **Improved**: Error traces for better debugging experience
- **Enhanced**: Exception handling and error reporting

---

## Version 2.12.0 (2023-12-21)

### 🧹 Maintenance
- **Removed**: `textwrap` dependency (was causing import issues)
- **Fixed**: README documentation formatting
- **Updated**: Dependency cleanup

---

## Version 2.11.x Series (2023)

The 2.11.x series focused on advanced parameter management and hierarchical configurations.

### Version 2.11.16 (2023-09-03)
- **Added**: Tree mode support for both Meta instances and ParamsProto instances
- **Improved**: Nested dictionary handling with `_tree` attribute
- **Enhanced**: Parameter attribute management

### Version 2.11.14 (2023-07-24)  
- **Fixed**: `__vars__` property for ParamsProto descendants
- **Improved**: Property descriptor handling for dynamic attributes

### Version 2.11.13 (2023-07-24)
- **Enhanced**: `vars(Args)` functionality for better introspection
- **Added**: Dynamic property access improvements

### Version 2.11.12 (2023-07-24)
- **Changed**: `Args.property` now returns property descriptor instead of value
- **Improved**: Property-based parameter access patterns

### Version 2.11.11 (2023-07-23)
- **Removed**: Debug print statements from production code
- **Cleaned**: Console output for cleaner user experience

### Version 2.11.10 (2023-07-23)
- **Added**: Parameter freezing once namespace is instantiated
- **Improved**: Immutability patterns for configuration safety

### Version 2.11.9 (2023-07-22)
- **Fixed**: Function detection in parameter attributes
- **Improved**: Dynamic attribute handling

### Version 2.11.8 (2023-07-22)
- **Changed**: Child detection moved to initialization time
- **Improved**: Performance of nested configuration detection

### Version 2.11.7 (2023-07-22)
- **Fixed**: `__new__` method super() call issues
- **Improved**: Object instantiation patterns

### Version 2.11.6 (2023-07-22)
- **Fixed**: ParamsProto set as non-recursive Bear to prevent dict→Bear conversion
- **Improved**: Nested attribute handling

### Version 2.11.5 (2023-07-22)
- **Added**: Support for object properties in configurations
- **Enhanced**: Property-based parameter definitions

### Version 2.11.4 (2023-07-22)
- **Improved**: Internal attribute management
- **Enhanced**: Performance optimizations

### Version 2.11.3 (2023-07-21)
- **Fixed**: Critical `__vars__` property bug
- **Improved**: Dictionary representation of configurations

### Version 2.11.2 (2023-07-21)
- **Fixed**: Argument parsing edge cases
- **Improved**: CLI argument handling robustness

### Version 2.11.1 (2023-07-21)
- **Added**: Children attribute support for nested configurations
- **Enhanced**: Hierarchical parameter management

### Version 2.11.0 (2023-07-21)
- **Major**: Added hierarchical hydration support
- **Added**: Nested configuration update mechanisms
- **Enhanced**: Multi-level parameter management

---

## Older Versions

### Version 2.12.0
- **Removed**: `textwrap` dependency
- **Fixed**: Documentation links and formatting

### Version 2.11.x Series
- Multiple bug fixes and improvements
- Enhanced CLI argument parsing
- Better error handling and validation

### Version 2.10.x Series
- Stability improvements
- Performance optimizations
- Bug fixes in parameter handling

### Version 2.9.x Series
- Enhanced hyperparameter sweep functionality
- Improved nested configuration support
- Better environment variable handling

### Version 2.8.x Series (Long-term stable)
- Core functionality stabilization
- Extensive bug fixes and improvements
- Enhanced CLI features

### Version 2.4.0 - 2.8.0
- Major feature additions
- API improvements and stabilization
- Enhanced documentation

### Version 2.0.3 - 2.2.0
- Initial stable releases
- Core parameter management functionality
- Basic CLI integration

---

## Migration Guide

### Upgrading to 2.13.x

The 2.13.x series introduces comprehensive documentation but maintains full backward compatibility. No code changes are required.

### Key Changes Since 2.0.x

1. **Environment Variables**: Enhanced support for environment variable defaults and validation
2. **Nested Configurations**: Improved support for hierarchical parameter structures  
3. **Hyperparameter Sweeps**: Advanced sweep functionality with the `Sweep` class
4. **Type Safety**: Better type hints and validation throughout the codebase
5. **Documentation**: Complete rewrite with examples and API reference

---

## Breaking Changes

### Version 2.13.0
- **Removed**: `typing` dependency - ensure your Python version supports built-in type hints
- **Repository**: Moved from `episodeyang/params_proto` to `geyang/params-proto`

### Earlier Versions
- See individual version notes above for specific breaking changes

---

## Contributors

- **Ge Yang** - Primary author and maintainer
- **Claude Code** - Documentation generation and enhancement

---

## Support

- **Documentation**: https://params-proto.readthedocs.io/
- **Issues**: https://github.com/geyang/params-proto/issues
- **Repository**: https://github.com/geyang/params-proto