from dataclasses import dataclass
from typing import Literal

import numpy as np

from params_proto import proto


def test_canonical_usage():
  @dataclass
  class Perspect:
    fov: float = 60.0  # in degrees
    near: float = 0.1  # in meters
    far: float = 100.0  # in meters

  @dataclass
  class Orthographic:
    zoom: float = 1.0
    near: float = 0.1  # in meters
    far: float = 100.0  # in meters

  CUDA_IS_AVAILABLE = False

  @proto
  class Nerf:
    """Add doc string here."""

    seed: int  # always required, this is a position argument

    # this also a position argument, following the seed.
    camera: Perspect | Orthographic
    """The camera can be either perspective or orthographic.

    In the command line, `main.py 42 camera:orthographic --lr 0.001`
    """

    # Various documentation styles and types
    lr: float = 0.01  # learning rate
    batch_size: int = 32  # single line hash comment
    model_name: str = "resnet50"
    """Single line triple-quote docstring"""

    train: bool = True  # This is the training mode flag
    # It controls whether the model is in training or evaluation mode
    # Use --no-train or --train=false to switch to evaluation mode

    optimizer: Literal["adam", "sgd", "rms-prop"] = "adam"
    """Optimizer to use for training. The supported values are: adam,
    sgd, rmsprop. Default is adam with default learning
    rate.
    """

    seed: np.random.rand  # random seed for reproducibility

    cuda: bool = True if CUDA_IS_AVAILABLE else False
    """Enable CUDA acceleration if available, passing the flag always makes it true."""

    # Data augmentation settings
    augment: bool = False

    checkpoint_path: str = "./checkpoints"  # path to save checkpoints

    # we are using this as a singleton.
    @classmethod
    def main(self):
      print(self.camera)
      print(self.seed, self.lr)

  # Local Override of the default values
  Nerf.seed = 100
  Nerf.lr = 0.001

  # main function
  nerf = Nerf().main()
  assert nerf.seed == 100, "should be overriden"
  assert nerf.lr == 0.001, "should be overriden"


def test_function_usage():
  """Test using proto with function-based configs and Union types."""

  @dataclass
  class Adam:
    """Adam optimizer configuration."""

    lr: float = 0.001  # learning rate
    beta1: float = 0.9  # first moment decay
    beta2: float = 0.999  # second moment decay

    eps: float = 1e-8
    """Epsilon for numerical stability"""

    # Weight decay (L2 regularization)
    weight_decay: float = 0.0

  @dataclass
  class SGD:
    """Stochastic Gradient Descent optimizer."""

    lr: float = 0.01
    """Learning rate for SGD"""

    momentum: float = 0.0  # momentum factor

    # Enable Nesterov momentum
    # Provides better convergence in some cases
    nesterov: bool = False

    weight_decay: float = 0.0  # L2 penalty

  @dataclass
  class RMSProp:
    lr: float = 0.01  # learning rate
    alpha: float = 0.99  # smoothing constant

    eps: float = 1e-8
    """Term added to denominator for numerical stability"""

    # Momentum factor
    momentum: float = 0.0

    centered: bool = False
    """If True, compute centered RMSProp"""

    weight_decay: float = 0.0  # weight decay (L2 penalty)

  @proto
  def train(
    seed: int,
    optimizer: Adam | SGD | RMSProp,
    lr: float = 0.01,
    batch_size: int = 32,
    epochs: int = 100,
    log_interval: int = 10,
    **kwargs,
  ):
    r = dict(locals())
    print(r)
    return r

  train.seed = 100
  train.optimizer = "adam"
  train.lr = 0.001
  train.batch_size = 128

  result = train(batch_size=256)  # noqa

  assert result.seed == 100, "should be overriden"
  assert result.lr == 0.001, "should be overriden"
  assert result.batch_size == 256, "direct passage takes precedence over partial"


def test_singleton_override():
  """Test proto.prefix decorator with nested function calls.

  This demonstrates how prefixed functions can be called from within
  a main proto function, with CLI args automatically passed through.
  """

  @proto.prefix
  def train(
    lr: float = 0.001,  # learning rate
    batch_size: int = 32,
    epochs: int = 100,  # number of training epochs
    checkpoint_dir: str = "./checkpoints",  # directory to save training checkpoints
  ):
    """Training configuration and execution."""
    print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}")
    return {
      "mode": "train",
      "lr": lr,
      "batch_size": batch_size,
      "epochs": epochs,
      "checkpoint_dir": checkpoint_dir,
    }

  @proto.prefix
  def eval(
    batch_size: int = 64,  # batch size for evaluation
    checkpoint: str = None,  # checkpoint to evaluate (None = latest)
    num_samples: int = 1000,  # number of samples to evaluate on
  ):
    """Evaluation configuration and execution."""
    print(f"Evaluating with batch_size={batch_size}, checkpoint={checkpoint}")
    return {
      "mode": "eval",
      "batch_size": batch_size,
      "checkpoint": checkpoint,
      "num_samples": num_samples,
    }

  @proto
  def main(
    seed: int = 42,  # random seed for reproducibility
    debug: bool = False,  # enable debug mode with verbose logging
  ):
    """Main entry point that orchestrates training and evaluation.

    The train and eval functions inherit their arguments from the CLI.
    You can override them like: --train.lr 0.01 --eval.batch-size 128
    """
    print(f"Main: seed={seed}, debug={debug}")

    # Call train and eval - they automatically pick up CLI args
    train_result = train()
    eval_result = eval()

    return {
      "seed": seed,
      "debug": debug,
      "train": train_result,
      "eval": eval_result,
    }

  # Example 1: Call main directly, override train and eval params
  result = main(seed=100, train={"lr": 0.01, "epochs": 50}, eval={"batch_size": 128})

  assert result["seed"] == 100
  assert result["train"]["lr"] == 0.01
  assert result["train"]["epochs"] == 50
  assert result["eval"]["batch_size"] == 128

  # Example 2: Set prefixed function values before calling main
  train.lr = 0.002
  train.batch_size = 64
  train.epochs = 75
  eval.batch_size = 256
  eval.num_samples = 2000

  result = main(seed=200)

  assert result["seed"] == 200
  assert result["train"]["lr"] == 0.002
  assert result["train"]["batch_size"] == 64
  assert result["train"]["epochs"] == 75
  assert result["eval"]["batch_size"] == 256
  assert result["eval"]["num_samples"] == 2000

  # Example 3: Use proto.bind() to set overrides, then call normally
  # This mirrors CLI syntax: --seed 300 --train.lr 0.003 --eval.checkpoint ./best.pt
  proto.bind(
    **{
      "seed": 300,
      "train.lr": 0.003,
      "train.epochs": 80,
      "eval.checkpoint": "./best.pt",
    }
  )

  result = main()  # Just call main() - it picks up the bound values

  assert result["seed"] == 300
  assert result["train"]["lr"] == 0.003
  assert result["train"]["epochs"] == 80
  assert result["eval"]["checkpoint"] == "./best.pt"

  # Example 4: Can also use with context manager for scoped overrides
  with proto.bind(seed=400, **{"train.lr": 0.004, "eval.num_samples": 5000}):
    result = main()

    assert result["seed"] == 400
    assert result["train"]["lr"] == 0.004
    assert result["eval"]["num_samples"] == 5000

  # Values reset after context exits
  result = main()
  assert result["seed"] != 400  # Back to defaults or previous values

  """
  Command line examples:

  # Run with default values
  python main.py

  # Override train learning rate and eval batch size
  python main.py --train.lr 0.01 --eval.batch-size 128

  # Set seed and enable debug mode
  python main.py --seed 123 --debug

  # Full customization
  python main.py --seed 99 --debug --train.lr 0.001 --train.epochs 200 --train.batch-size 64 --eval.batch-size 256 --eval.checkpoint "./best.pt"
  """


def test_documentation_styles():
  """Test various documentation styles and type annotations."""
  from typing import Dict, List, Optional

  @proto
  class Params:  # === Basic Types with Different Doc Styles ===
    # Simple single-line hash comment
    learning_rate: float = 0.001

    batch_size: int = 64  # inline hash comment

    model_type: str = "transformer"
    """Single-line triple-quote docstring."""

    # Multi-line hash comment example
    # This parameter controls the dropout rate
    # Valid range: 0.0 to 1.0
    dropout: float = 0.1

    enable_logging: bool = True
    """
    Enable detailed logging during training.
    Set to False in production for better performance.
    """

    # === Collection Types ===

    layer_sizes: List[int] = None  # hidden layer dimensions
    """List of integers defining network architecture."""

    # Hyperparameters dictionary
    # Contains optimizer-specific settings
    hparams: Dict[str, float] = None

    # === Optional Types ===

    checkpoint_dir: Optional[str] = None  # optional checkpoint directory

    resume_from: Optional[str] = None
    """
    Path to checkpoint file to resume training from.
    If None, training starts from scratch.
    """

    # === Numeric Types ===

    max_steps: int = 10000  # maximum training steps

    warmup_steps: int = 1000
    """Number of warmup steps for learning rate scheduler"""

    # Weight decay for regularization
    # Typical values: 0.0, 1e-5, 1e-4
    weight_decay: float = 0.01

    # === String Types ===

    experiment_name: str = "default"  # name for this experiment run

    device: str = "cuda"
    """Device to use: 'cuda', 'cpu', or 'mps'"""

    # === Boolean Flags ===

    use_amp: bool = False  # automatic mixed precision training

    # Enable gradient checkpointing to save memory
    # Trades computation for memory
    gradient_checkpointing: bool = False

    eval_only: bool = False
    """Run evaluation only, skip training"""


def test_union():
  @dataclass
  class Perspect:
    fov: float = 60.0  # in degrees
    near: float = 0.1  # in meters
    far: float = 100.0  # in meters

  @dataclass
  class Orthographic:
    zoom: float = 1.0
    near: float = 0.1  # in meters
    far: float = 100.0  # in meters

  @proto
  class Nerf:
    # this camera object has to be instantiated.
    camera: Perspect | Orthographic
    """The camera can be either perspective or orthographic."""

    def main(self):
      print(self.camera)

  # main function
  proto.cli()


def test_enum():
  """Test enum types for fixed choice parameters."""
  from enum import Enum, auto

  class Activation(Enum):
    """Neural network activation functions."""

    RELU = auto()
    GELU = auto()
    TANH = auto()
    SIGMOID = auto()

  class Optimizer(Enum):
    """Optimizer algorithms."""

    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"

  class Precision(Enum):
    """Training precision modes."""

    FP32 = 32  # Full precision
    FP16 = 16  # Half precision
    BF16 = "bf16"  # Brain float 16

  @proto
  class Train:
    # Enum with auto() values
    activation: Activation = Activation.RELU
    """Activation function for hidden layers"""

    # Enum with string values
    optimizer: Optimizer = Optimizer.ADAM  # optimizer to use

    # Enum with mixed int/string values
    precision: Precision = Precision.FP32
    """Training precision mode for mixed precision training"""

    # Regular parameters
    lr: float = 0.001  # learning rate

  # Test default values
  config = Train()
  assert config.activation == Activation.RELU
  assert config.optimizer == Optimizer.ADAM
  assert config.precision == Precision.FP32

  # Test programmatic override
  config.activation = Activation.GELU
  config.optimizer = Optimizer.ADAMW
  config.precision = Precision.FP16

  assert config.activation == Activation.GELU
  assert config.optimizer == Optimizer.ADAMW
  assert config.precision == Precision.FP16

  # Test value access
  assert config.optimizer.value == "adamw"
  assert config.precision.value == 16

  """
  Command line examples:

  # Use default enum values
  python train.py
  # activation=RELU, optimizer=ADAM, precision=FP32

  # Override with enum member names
  python train.py --activation GELU --optimizer ADAMW

  # Set precision for mixed precision training
  python train.py --precision FP16

  # Full configuration
  python train.py --activation TANH --optimizer SGD --precision BF16 --lr 0.01

  # Invalid enum values will error:
  python train.py --activation SILU
  # Error: invalid choice 'SILU' (choose from RELU, GELU, TANH, SIGMOID)
  """


def test_inheritance():
  class Camera:
    near: float = 0.1  # in meters
    far: float = 100.0  # in meters

    def main(self):
      print("Camera type is", self.__class__.__name__)

  class Perspect(Camera):
    fov: float = 60.0
    aspect: float = 1.0

  class Orthographic(Camera):
    zoom: float = 1.0
    left: float = 0.0
    right: float = 1.0
    top: float = 1.0
    bottom: float = 0.0

  # 这是一个自变量 （literal)｡
  proto.cli(Perspect | Orthographic)
  """
  > python main.py orthographic --zoom 0.5
  Camera type is Orthographic
  """


def test_tuples():
  """Test tuple types with fixed and variable lengths."""
  import pathlib
  from typing import Tuple

  @proto
  class Mlp:
    # Fixed-length tuples for structured data
    input_shape: tuple[int, int, int] = (224, 224, 3)
    """Image input dimensions (height, width, channels)"""

    # RGB color normalization mean
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)

    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    """RGB standard deviation for normalization"""

    # Bounding box format: (x, y, width, height)
    default_bbox: tuple[int, int, int, int] = (0, 0, 100, 100)

    # Variable-length tuples
    checkpoint_paths: tuple[pathlib.Path, ...]
    """Multiple checkpoint files to load from"""

    # Layer dimensions for neural network
    hidden_sizes: tuple[int, ...] = (512, 256, 128)
    """Hidden layer sizes in decreasing order"""

    # Data augmentation scales
    augment_scales: tuple[float, ...] = (0.8, 1.0, 1.2)

    # GPU device IDs to use
    gpu_ids: tuple[int, ...] = (0,)
    """GPU device IDs for multi-GPU training"""

  """
  Command line examples:

  # Basic usage with required variable-length tuple
  python train.py --checkpoint-paths ./ckpt1.pt ./ckpt2.pt

  # Override fixed-length tuples
  python train.py --checkpoint-paths ./model.pt --input-shape 256 256 3

  # Set normalization parameters
  python train.py --checkpoint-paths ./model.pt --norm-mean 0.5 0.5 0.5 --norm-std 0.5 0.5 0.5

  # Configure network architecture
  python train.py --checkpoint-paths ./model.pt --hidden-sizes 1024 512 256 128 64

  # Multi-GPU training
  python train.py --checkpoint-paths ./model.pt --gpu-ids 0 1 2 3

  # Full configuration
  python train.py \\
    --checkpoint-paths ./pretrain.pt ./finetune.pt \\
    --input-shape 384 384 3 \\
    --hidden-sizes 2048 1024 512 \\
    --augment-scales 0.5 0.75 1.0 1.25 1.5 \\
    --gpu-ids 0 1
  """
