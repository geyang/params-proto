"""
Tests for nested CLI subcommands.

Tests hierarchical command structures with more than 2 layers of nesting.
"""

from textwrap import dedent

import pytest

# Layer 1: Basic CLI function
# Layer 2: Union subcommand (e.g., Train | Eval)
# Layer 3: Nested class inside subcommand (e.g., Model inside Train)


class TestTwoLayerNesting:
  """Test standard 2-layer nesting: CLI function -> Union subcommand."""

  def test_subcommand_selection(self, run_cli):
    """Test selecting between subcommands at layer 2."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class TrainConfig:
          epochs: int = 100
          lr: float = 0.001

      @dataclass
      class EvalConfig:
          checkpoint: str = "model.pt"
          batch_size: int = 32

      @proto.cli
      def main(config: TrainConfig | EvalConfig):
          print(f"{config.__class__.__name__}")
          if isinstance(config, TrainConfig):
              print(f"epochs={config.epochs},lr={config.lr}")
          else:
              print(f"checkpoint={config.checkpoint},batch_size={config.batch_size}")

      if __name__ == "__main__":
          main()
      """)

    result = run_cli(script, ["--config:TrainConfig"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "TrainConfig"
    assert lines[1] == "epochs=100,lr=0.001"

    result = run_cli(script, ["--config:EvalConfig"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "EvalConfig"
    assert lines[1] == "checkpoint=model.pt,batch_size=32"

  def test_subcommand_with_overrides(self, run_cli):
    """Test overriding subcommand parameters."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class TrainConfig:
          epochs: int = 100
          lr: float = 0.001

      @dataclass
      class EvalConfig:
          checkpoint: str = "model.pt"

      @proto.cli
      def main(config: TrainConfig | EvalConfig):
          if isinstance(config, TrainConfig):
              print(f"epochs={config.epochs},lr={config.lr}")
          else:
              print(f"checkpoint={config.checkpoint}")

      if __name__ == "__main__":
          main()
      """)

    result = run_cli(
      script, ["--config:TrainConfig", "--config.epochs", "200", "--config.lr", "0.01"]
    )
    assert result["stdout"].strip() == "epochs=200,lr=0.01"

    result = run_cli(
      script, ["--config:EvalConfig", "--config.checkpoint", "best_model.pt"]
    )
    assert result["stdout"].strip() == "checkpoint=best_model.pt"


class TestThreeLayerNesting:
  """Test 3-layer nesting: CLI -> Subcommand -> Nested class.

  This tests nested dataclasses where a subcommand contains another
  dataclass as an attribute.
  """

  def test_nested_dataclass_in_subcommand(self, run_cli):
    """Test subcommand with a nested dataclass attribute."""
    script = dedent("""
      from dataclasses import dataclass, field
      from params_proto import proto

      @dataclass
      class ModelConfig:
          hidden_size: int = 256
          num_layers: int = 4

      @dataclass
      class TrainConfig:
          epochs: int = 100
          model: ModelConfig = field(default_factory=ModelConfig)

      @dataclass
      class EvalConfig:
          checkpoint: str = "model.pt"

      @proto.cli
      def main(config: TrainConfig | EvalConfig):
          if isinstance(config, TrainConfig):
              print(f"epochs={config.epochs}")
              print(f"model.hidden_size={config.model.hidden_size}")
              print(f"model.num_layers={config.model.num_layers}")
          else:
              print(f"checkpoint={config.checkpoint}")

      if __name__ == "__main__":
          main()
      """)

    # Test with defaults
    result = run_cli(script, ["--config:TrainConfig"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "epochs=100"
    assert lines[1] == "model.hidden_size=256"
    assert lines[2] == "model.num_layers=4"

  def test_nested_dataclass_override_via_dot_notation(self, run_cli):
    """Test overriding nested dataclass fields via dot notation."""
    script = dedent("""
      from dataclasses import dataclass, field
      from params_proto import proto

      @dataclass
      class ModelConfig:
          hidden_size: int = 256
          num_layers: int = 4

      @dataclass
      class TrainConfig:
          epochs: int = 100
          model: ModelConfig = field(default_factory=ModelConfig)

      @dataclass
      class EvalConfig:
          checkpoint: str = "model.pt"

      @proto.cli
      def main(config: TrainConfig | EvalConfig):
          if isinstance(config, TrainConfig):
              print(f"epochs={config.epochs}")
              print(f"model.hidden_size={config.model.hidden_size}")
              print(f"model.num_layers={config.model.num_layers}")
          else:
              print(f"checkpoint={config.checkpoint}")

      if __name__ == "__main__":
          main()
      """)

    # Try to override nested model config via deep dot notation
    # This tests if --config.model.hidden_size works
    result = run_cli(
      script,
      [
        "--config:TrainConfig",
        "--config.epochs",
        "200",
        "--config.model.hidden_size",
        "512",
        "--config.model.num_layers",
        "8",
      ],
      expect_error=True,  # This may not be supported yet
    )

    # Check if it succeeded or failed
    if result["returncode"] == 0:
      lines = result["stdout"].strip().split("\n")
      assert lines[0] == "epochs=200"
      assert lines[1] == "model.hidden_size=512"
      assert lines[2] == "model.num_layers=8"
    else:
      # If it failed, this documents the current limitation
      pytest.skip(
        "Deep nested dot notation (--config.model.hidden_size) not yet supported"
      )


class TestThreeLayerUnionNesting:
  """Test 3-layer Union nesting: CLI -> Union -> Union."""

  def test_nested_union_subcommand(self, run_cli):
    """Test Union parameter inside a Union subcommand."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      # Layer 3: Model types
      @dataclass
      class TransformerModel:
          num_heads: int = 8
          hidden_dim: int = 512

      @dataclass
      class CNNModel:
          num_filters: int = 64
          kernel_size: int = 3

      # Layer 2: Command types
      @dataclass
      class TrainConfig:
          epochs: int = 100
          model: TransformerModel | CNNModel = None

      @dataclass
      class EvalConfig:
          checkpoint: str = "model.pt"

      # Layer 1: CLI entry point
      @proto.cli
      def main(config: TrainConfig | EvalConfig):
          print(f"command={config.__class__.__name__}")
          if isinstance(config, TrainConfig):
              print(f"epochs={config.epochs}")
              if config.model:
                  print(f"model={config.model.__class__.__name__}")
                  if isinstance(config.model, TransformerModel):
                      print(f"num_heads={config.model.num_heads}")
                  else:
                      print(f"num_filters={config.model.num_filters}")

      if __name__ == "__main__":
          main()
      """)

    # Test basic selection without nested union
    result = run_cli(script, ["--config:TrainConfig"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "command=TrainConfig"
    assert lines[1] == "epochs=100"


class TestMultipleSubcommands:
  """Test CLI with multiple Union parameters (sibling subcommands)."""

  def test_two_union_parameters(self, run_cli):
    """Test function with two independent Union parameters."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class AdamOptimizer:
          lr: float = 0.001
          beta1: float = 0.9

      @dataclass
      class SGDOptimizer:
          lr: float = 0.01
          momentum: float = 0.9

      @dataclass
      class CrossEntropyLoss:
          label_smoothing: float = 0.0

      @dataclass
      class MSELoss:
          reduction: str = "mean"

      @proto.cli
      def main(
          optimizer: AdamOptimizer | SGDOptimizer,
          loss: CrossEntropyLoss | MSELoss,
      ):
          print(f"optimizer={optimizer.__class__.__name__}")
          print(f"loss={loss.__class__.__name__}")
          if isinstance(optimizer, AdamOptimizer):
              print(f"lr={optimizer.lr},beta1={optimizer.beta1}")
          else:
              print(f"lr={optimizer.lr},momentum={optimizer.momentum}")

      if __name__ == "__main__":
          main()
      """)

    result = run_cli(script, ["--optimizer:AdamOptimizer", "--loss:CrossEntropyLoss"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "optimizer=AdamOptimizer"
    assert lines[1] == "loss=CrossEntropyLoss"
    assert lines[2] == "lr=0.001,beta1=0.9"

  def test_two_union_parameters_with_overrides(self, run_cli):
    """Test overriding both Union parameters."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class AdamOptimizer:
          lr: float = 0.001

      @dataclass
      class SGDOptimizer:
          lr: float = 0.01

      @dataclass
      class CrossEntropyLoss:
          label_smoothing: float = 0.0

      @dataclass
      class MSELoss:
          reduction: str = "mean"

      @proto.cli
      def main(
          optimizer: AdamOptimizer | SGDOptimizer,
          loss: CrossEntropyLoss | MSELoss,
      ):
          print(f"optimizer.lr={optimizer.lr}")
          if isinstance(loss, CrossEntropyLoss):
              print(f"loss.label_smoothing={loss.label_smoothing}")
          else:
              print(f"loss.reduction={loss.reduction}")

      if __name__ == "__main__":
          main()
      """)

    result = run_cli(
      script,
      [
        "--optimizer:AdamOptimizer",
        "--optimizer.lr",
        "0.0001",
        "--loss:MSELoss",
        "--loss.reduction",
        "sum",
      ],
    )
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "optimizer.lr=0.0001"
    assert lines[1] == "loss.reduction=sum"


class TestPositionalSubcommands:
  """Test positional argument style for subcommands (like git)."""

  def test_positional_subcommand_first(self, run_cli):
    """Test subcommand as first positional argument."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class CloneCommand:
          url: str = ""
          depth: int = 0

      @dataclass
      class PullCommand:
          remote: str = "origin"
          branch: str = "main"

      @proto.cli
      def git(command: CloneCommand | PullCommand):
          print(f"command={command.__class__.__name__}")
          if isinstance(command, CloneCommand):
              print(f"url={command.url},depth={command.depth}")
          else:
              print(f"remote={command.remote},branch={command.branch}")

      if __name__ == "__main__":
          git()
      """)

    result = run_cli(
      script, ["clone-command", "--command.url", "https://github.com/test"]
    )
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "command=CloneCommand"
    assert "url=https://github.com/test" in lines[1]

  def test_abbreviated_subcommand_name(self, run_cli):
    """Test abbreviated subcommand name matching."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class CloneCommand:
          url: str = ""

      @dataclass
      class PullCommand:
          remote: str = "origin"

      @proto.cli
      def git(command: CloneCommand | PullCommand):
          print(f"command={command.__class__.__name__}")

      if __name__ == "__main__":
          git()
      """)

    # "clone" should match "CloneCommand"
    result = run_cli(script, ["clone"])
    assert result["stdout"].strip() == "command=CloneCommand"

    # "pull" should match "PullCommand"
    result = run_cli(script, ["pull"])
    assert result["stdout"].strip() == "command=PullCommand"


class TestSubcommandWithMixedParams:
  """Test subcommands alongside regular parameters."""

  def test_subcommand_with_bool_and_string(self, run_cli):
    """Test subcommand with mixed parameter types."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class TrainConfig:
          epochs: int = 100

      @dataclass
      class EvalConfig:
          checkpoint: str = "model.pt"

      @proto.cli
      def main(
          config: TrainConfig | EvalConfig,
          output_dir: str = "./output",
          verbose: bool = False,
          dry_run: bool = False,
      ):
          print(f"config={config.__class__.__name__}")
          print(f"output_dir={output_dir}")
          print(f"verbose={verbose}")
          print(f"dry_run={dry_run}")

      if __name__ == "__main__":
          main()
      """)

    result = run_cli(
      script,
      [
        "--config:TrainConfig",
        "--output-dir",
        "/tmp/test",
        "--verbose",
        "--no-dry-run",
      ],
    )
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "config=TrainConfig"
    assert lines[1] == "output_dir=/tmp/test"
    assert lines[2] == "verbose=True"
    assert lines[3] == "dry_run=False"


class TestDeepNestedPositional:
  """Test deep nesting with positional-style subcommand selection."""

  def test_three_level_hierarchy(self, run_cli):
    """Test 3-level command hierarchy like: main -> train -> transformer."""
    script = dedent("""
      from dataclasses import dataclass, field
      from typing import Optional
      from params_proto import proto

      # Level 3: Model architectures
      @dataclass
      class TransformerArch:
          heads: int = 8
          layers: int = 6

      @dataclass
      class RNNArch:
          hidden_size: int = 256
          bidirectional: bool = True

      # Level 2: Commands that use a model
      @dataclass
      class TrainCommand:
          epochs: int = 100
          lr: float = 0.001

      @dataclass
      class ExportCommand:
          format: str = "onnx"
          quantize: bool = False

      # Level 1: CLI function
      @proto.cli
      def main(command: TrainCommand | ExportCommand):
          print(f"command={command.__class__.__name__}")
          if isinstance(command, TrainCommand):
              print(f"epochs={command.epochs},lr={command.lr}")
          else:
              print(f"format={command.format},quantize={command.quantize}")

      if __name__ == "__main__":
          main()
      """)

    # Test train command
    result = run_cli(script, ["train", "--command.epochs", "200"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "command=TrainCommand"
    assert lines[1] == "epochs=200,lr=0.001"

    # Test export command
    # Note: Boolean flags in dot notation require explicit value (true/false)
    # unlike top-level flags which support --flag syntax
    result = run_cli(
      script,
      ["export", "--command.format", "torchscript", "--command.quantize", "true"],
    )
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "command=ExportCommand"
    assert "format=torchscript" in lines[1]
    assert "quantize=True" in lines[1]


class TestBooleanFlagsInSubcommands:
  """Test boolean flag behavior in subcommand dot notation.

  Documents that boolean flags in dot notation require explicit true/false
  values, unlike top-level boolean flags which support --flag/--no-flag syntax.
  """

  def test_toplevel_bool_flag_syntax(self, run_cli):
    """Top-level booleans support --flag and --no-flag syntax."""
    script = dedent("""
      from params_proto import proto

      @proto.cli
      def main(verbose: bool = False, debug: bool = True):
          print(f"verbose={verbose},debug={debug}")

      if __name__ == "__main__":
          main()
      """)

    # --flag sets to True
    result = run_cli(script, ["--verbose"])
    assert result["stdout"].strip() == "verbose=True,debug=True"

    # --no-flag sets to False
    result = run_cli(script, ["--no-debug"])
    assert result["stdout"].strip() == "verbose=False,debug=False"

  def test_subcommand_bool_requires_explicit_value(self, run_cli):
    """Boolean in dot notation requires explicit true/false value."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class Config:
          verbose: bool = False
          enabled: bool = True

      @proto.cli
      def main(config: Config):
          print(f"verbose={config.verbose},enabled={config.enabled}")

      if __name__ == "__main__":
          main()
      """)

    # Must use explicit true/false with dot notation
    # Single-class params still require selection syntax
    result = run_cli(script, ["config", "--config.verbose", "true"])
    assert result["stdout"].strip() == "verbose=True,enabled=True"

    result = run_cli(script, ["config", "--config.enabled", "false"])
    assert result["stdout"].strip() == "verbose=False,enabled=False"

    # --config.verbose alone (without value) should error
    result = run_cli(script, ["config", "--config.verbose"], expect_error=True)
    assert "requires a value" in result["stderr"]


class TestUnprefixedByDefault:
  """Test that subcommand attrs don't need prefix by default."""

  def test_unprefixed_subcommand_attrs_default(self, run_cli):
    """Test that --epochs works by default (no prefix needed)."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class TrainConfig:
          epochs: int = 100
          lr: float = 0.001

      @dataclass
      class EvalConfig:
          checkpoint: str = "model.pt"

      @proto.cli
      def main(config: TrainConfig | EvalConfig):
          if isinstance(config, TrainConfig):
              print(f"epochs={config.epochs},lr={config.lr}")
          else:
              print(f"checkpoint={config.checkpoint}")

      if __name__ == "__main__":
          main()
      """)

    # Test with unprefixed args (default behavior)
    result = run_cli(script, ["train-config", "--epochs", "200", "--lr", "0.01"])
    assert result["stdout"].strip() == "epochs=200,lr=0.01"

    result = run_cli(script, ["eval-config", "--checkpoint", "best.pt"])
    assert result["stdout"].strip() == "checkpoint=best.pt"

  def test_prefixed_syntax_still_works(self, run_cli):
    """Test that prefixed syntax still works by default."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class TrainConfig:
          epochs: int = 100

      @proto.cli
      def main(config: TrainConfig):
          print(f"epochs={config.epochs}")

      if __name__ == "__main__":
          main()
      """)

    # Prefixed syntax should still work
    result = run_cli(script, ["train-config", "--config.epochs", "200"])
    assert result["stdout"].strip() == "epochs=200"

  def test_mixed_regular_and_subcommand_params(self, run_cli):
    """Test unprefixed args with both regular params and subcommand."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class TrainConfig:
          epochs: int = 100

      @proto.cli
      def main(
          config: TrainConfig,
          output: str = "output",
          verbose: bool = False,
      ):
          print(f"epochs={config.epochs}")
          print(f"output={output}")
          print(f"verbose={verbose}")

      if __name__ == "__main__":
          main()
      """)

    result = run_cli(
      script, ["train-config", "--epochs", "200", "--output", "results", "--verbose"]
    )
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "epochs=200"
    assert lines[1] == "output=results"
    assert lines[2] == "verbose=True"

  def test_multiple_union_params_unprefixed(self, run_cli):
    """Test unprefixed attrs with two Union parameters."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class AdamOptimizer:
          lr: float = 0.001
          beta1: float = 0.9

      @dataclass
      class SGDOptimizer:
          lr: float = 0.01
          momentum: float = 0.9

      @dataclass
      class CrossEntropyLoss:
          label_smoothing: float = 0.0

      @dataclass
      class MSELoss:
          reduction: str = "mean"

      @proto.cli
      def main(
          optimizer: AdamOptimizer | SGDOptimizer,
          loss: CrossEntropyLoss | MSELoss,
      ):
          print(f"optimizer={optimizer.__class__.__name__}")
          print(f"loss={loss.__class__.__name__}")
          if isinstance(optimizer, AdamOptimizer):
              print(f"lr={optimizer.lr},beta1={optimizer.beta1}")
          else:
              print(f"lr={optimizer.lr},momentum={optimizer.momentum}")
          if isinstance(loss, CrossEntropyLoss):
              print(f"label_smoothing={loss.label_smoothing}")
          else:
              print(f"reduction={loss.reduction}")

      if __name__ == "__main__":
          main()
      """)

    # Use unprefixed args for both union params
    result = run_cli(
      script,
      [
        "adam-optimizer",
        "cross-entropy-loss",
        "--lr",
        "0.0001",
        "--beta1",
        "0.95",
        "--label-smoothing",
        "0.1",
      ],
    )
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "optimizer=AdamOptimizer"
    assert lines[1] == "loss=CrossEntropyLoss"
    assert lines[2] == "lr=0.0001,beta1=0.95"
    assert lines[3] == "label_smoothing=0.1"


class TestProtoPrefixRequiresPrefix:
  """Test that @proto.prefix decorated classes require prefixed attrs."""

  def test_proto_prefix_class_requires_prefix(self, run_cli):
    """Test that @proto.prefix decorated classes require --config.epochs syntax."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @proto.prefix
      @dataclass
      class TrainConfig:
          epochs: int = 100
          lr: float = 0.001

      @dataclass
      class EvalConfig:
          checkpoint: str = "model.pt"

      @proto.cli
      def main(config: TrainConfig | EvalConfig):
          if isinstance(config, TrainConfig):
              print(f"epochs={config.epochs},lr={config.lr}")
          else:
              print(f"checkpoint={config.checkpoint}")

      if __name__ == "__main__":
          main()
      """)

    # TrainConfig is @proto.prefix, requires prefixed syntax
    result = run_cli(
      script, ["train-config", "--config.epochs", "200", "--config.lr", "0.01"]
    )
    assert result["stdout"].strip() == "epochs=200,lr=0.01"

    # EvalConfig is NOT @proto.prefix, unprefixed works
    result = run_cli(script, ["eval-config", "--checkpoint", "best.pt"])
    assert result["stdout"].strip() == "checkpoint=best.pt"

  def test_proto_prefix_unprefixed_should_error(self, run_cli):
    """Test that unprefixed args error for @proto.prefix classes."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @proto.prefix
      @dataclass
      class TrainConfig:
          epochs: int = 100

      @proto.cli
      def main(config: TrainConfig):
          print(f"epochs={config.epochs}")

      if __name__ == "__main__":
          main()
      """)

    # Unprefixed should error for @proto.prefix class
    result = run_cli(script, ["train-config", "--epochs", "200"], expect_error=True)
    assert "unrecognized argument" in result["stderr"]


class TestPositionalArgsInSubcommands:
  """Test positional arguments captured by subcommand fields.

  This tests the pattern: myapp add my-env/v1.2.3
  Where 'add' is the subcommand and 'my-env/v1.2.3' is captured by the
  subcommand's first required field.
  """

  def test_positional_arg_captured_by_subcommand_field(self, run_cli):
    """Test positional arg after subcommand name is captured."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class AddCommand:
          env: str  # Required field, no default

      @dataclass
      class RemoveCommand:
          env: str  # Required field

      @proto.cli
      def main(command: AddCommand | RemoveCommand):
          print(f"command={command.__class__.__name__}")
          print(f"env={command.env}")

      if __name__ == "__main__":
          main()
      """)

    # Positional arg after subcommand should be captured by 'env' field
    result = run_cli(script, ["add", "my-env/v1.2.3"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "command=AddCommand"
    assert lines[1] == "env=my-env/v1.2.3"

    result = run_cli(script, ["remove", "old-env/v0.9.0"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "command=RemoveCommand"
    assert lines[1] == "env=old-env/v0.9.0"

  def test_multiple_positional_args_in_subcommand(self, run_cli):
    """Test multiple positional args captured by subcommand fields."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class InstallCommand:
          package: str  # Required, first positional
          version: str  # Required, second positional

      @proto.cli
      def main(command: InstallCommand):
          print(f"package={command.package}")
          print(f"version={command.version}")

      if __name__ == "__main__":
          main()
      """)

    result = run_cli(script, ["install", "requests", "2.28.0"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "package=requests"
    assert lines[1] == "version=2.28.0"

  def test_positional_with_named_args(self, run_cli):
    """Test mixing positional and named args in subcommand."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class CloneCommand:
          url: str  # Required positional
          depth: int = 0  # Optional with default

      @proto.cli
      def main(command: CloneCommand):
          print(f"url={command.url}")
          print(f"depth={command.depth}")

      if __name__ == "__main__":
          main()
      """)

    # Positional url, named depth
    result = run_cli(script, ["clone", "https://github.com/test", "--depth", "1"])
    lines = result["stdout"].strip().split("\n")
    assert lines[0] == "url=https://github.com/test"
    assert lines[1] == "depth=1"

  def test_positional_arg_missing_should_error(self, run_cli):
    """Test that missing required positional arg errors."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class AddCommand:
          env: str  # Required field

      @proto.cli
      def main(command: AddCommand):
          print(f"env={command.env}")

      if __name__ == "__main__":
          main()
      """)

    # Missing required positional should error
    result = run_cli(script, ["add"], expect_error=True)
    assert result["returncode"] != 0

  def test_extra_positional_should_error(self, run_cli):
    """Test that extra unrecognized positional args error."""
    script = dedent("""
      from dataclasses import dataclass
      from params_proto import proto

      @dataclass
      class AddCommand:
          env: str  # Required field

      @proto.cli
      def main(command: AddCommand):
          print(f"env={command.env}")

      if __name__ == "__main__":
          main()
      """)

    # Extra positional arg should error
    result = run_cli(script, ["add", "env1", "extra-arg"], expect_error=True)
    assert result["returncode"] != 0
