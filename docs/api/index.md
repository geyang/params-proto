# API Reference

This section contains the complete API reference for params-proto.

```{toctree}
:maxdepth: 2

proto
hyper
utils
```

## Core Modules

- **{doc}`proto`**: Main module containing ParamsProto, Proto, and Flag classes
- **{doc}`hyper`**: Hyperparameter search and sweep functionality  
- **{doc}`utils`**: Utility functions and helpers

## Quick Reference

### Main Classes

```{eval-rst}
.. autoclass:: params_proto.v2.proto.ParamsProto
   :noindex:

.. autoclass:: params_proto.v2.proto.Proto
   :noindex:

.. autoclass:: params_proto.v2.proto.Flag
   :noindex:
```

### Key Methods

- `ParamsProto.parse()`: Parse command line arguments
- `ParamsProto.to_dict()`: Convert to dictionary
- `ParamsProto.from_dict()`: Create from dictionary