# Release Notes

This page contains the release history and changelog for params-proto.

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