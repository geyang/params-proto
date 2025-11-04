# Building CLI Applications

> **Note**: This page is under construction. Check back soon for complete documentation.

params-proto makes it easy to build command-line applications with rich parameter support.

## Basic CLI Tool

```python
from params_proto import proto

@proto.cli
def process_data(
    input_file: str,  # Input data file path
    output_file: str = "output.txt",  # Output file path
    verbose: bool = False,  # Enable verbose output
):
    """Process data from input file and save to output file."""
    print(f"Processing {input_file} -> {output_file}")
    # ... processing logic ...

if __name__ == "__main__":
    process_data()
```

Run it:
```bash
python process.py input.txt --output-file results.txt --verbose
```

See [Quick Start](../quick_start.md) and [Examples](ml_training.md) for more.
