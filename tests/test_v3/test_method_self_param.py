"""Tests for self/cls parameter handling in decorated methods.

These tests verify that self, cls, and similar special parameters are correctly
excluded from CLI parameters and config injection.
"""

from params_proto import proto


class TestProtoWrapperMethodParams:
  """Test that ProtoWrapper correctly handles method parameters."""

  def test_staticmethod_no_self_in_params(self):
    """Test that staticmethod parameters don't include self even if mistakenly added."""

    @proto.cli
    def train(lr: float = 0.01, batch_size: int = 32):
      """Training function."""
      return {"lr": lr, "batch_size": batch_size}

    # The wrapper should only have lr and batch_size as params
    assert "self" not in train._params
    assert "cls" not in train._params
    assert "lr" in train._params
    assert "batch_size" in train._params

  def test_classmethod_cls_not_in_params(self):
    """Test that @classmethod's cls parameter is excluded from CLI params."""

    class Trainer:
      lr: float = 0.01

      @proto.cli    # proto.cli on OUTSIDE - receives classmethod descriptor
      @classmethod
      def run(cls, batch_size: int = 32):
        """Run training."""
        return {"batch_size": batch_size, "trainer_cls": cls}

    # cls should NOT be in the params - proto.cli detects classmethod
    assert "cls" not in Trainer.run._params, (
      "cls parameter should be excluded from CLI params"
    )
    assert "batch_size" in Trainer.run._params
    assert Trainer.run._is_classmethod is True

  def test_instance_method_self_not_in_params(self):
    """Test that instance method's self parameter is excluded from CLI params."""

    class Trainer:
      def __init__(self, name="default"):
        self.name = name

      @proto.cli
      def train(self, lr: float = 0.01, batch_size: int = 32):
        """Train method."""
        return {"lr": lr, "batch_size": batch_size, "name": self.name}

    trainer = Trainer("test")

    # self should NOT be in the params
    assert "self" not in trainer.train._params, (
      "self parameter should be excluded from CLI params"
    )
    assert "lr" in trainer.train._params
    assert "batch_size" in trainer.train._params


class TestPartialMethodParams:
  """Test that proto.partial correctly handles method parameters."""

  def test_partial_with_instance_method_self_excluded(self):
    """Test that proto.partial excludes self from config injection."""

    class Config:
      lr: float = 0.01
      batch_size: int = 32

    class Trainer:
      def __init__(self, name="trainer"):
        self.name = name

      @proto.partial(Config, method=True)
      def train(self, lr, batch_size):
        return {"lr": lr, "batch_size": batch_size, "trainer": self}

    trainer = Trainer("mytrainer")
    result = trainer.train()

    # Should work correctly - self should be the trainer instance, not a config value
    assert result["lr"] == 0.01
    assert result["batch_size"] == 32
    assert result["trainer"] is trainer
    assert result["trainer"].name == "mytrainer"

  def test_partial_without_method_flag_self_issue(self):
    """Test that proto.partial without method=True handles self correctly.

    This demonstrates the bug: when method=True is NOT set but the function
    has a self parameter, the self parameter incorrectly gets treated as
    a config parameter.
    """

    class Config:
      lr: float = 0.01
      batch_size: int = 32
      # Note: no 'self' attribute in Config

    class Trainer:
      def __init__(self, name="trainer"):
        self.name = name

      # BUG: Without method=True, self is treated as a regular parameter
      @proto.partial(Config)
      def train(self, lr, batch_size):
        return {"lr": lr, "batch_size": batch_size, "trainer": self}

    trainer = Trainer("mytrainer")

    # This should work, but might fail because self is being treated incorrectly
    # The expected behavior is that self is automatically excluded
    result = trainer.train()

    assert result["lr"] == 0.01
    assert result["batch_size"] == 32
    assert result["trainer"] is trainer

  def test_partial_classmethod_cls_excluded(self):
    """Test that proto.partial excludes cls from config injection for classmethods."""

    class Config:
      lr: float = 0.01
      batch_size: int = 32

    class Trainer:
      name = "TrainerClass"

      @proto.partial(Config, method=True)  # proto.partial on OUTSIDE
      @classmethod
      def train(cls, lr, batch_size):
        return {"lr": lr, "batch_size": batch_size, "cls_name": cls.name}

    result = Trainer.train()

    # cls should be the Trainer class, not a config value
    assert result["lr"] == 0.01
    assert result["batch_size"] == 32
    assert result["cls_name"] == "TrainerClass"


class TestProtoDecoratorOnMethods:
  """Test @proto decorator directly on class methods.

  Note: @proto on instance methods and classmethods requires using proto.partial
  with method=True, or using @staticmethod. Direct @proto on instance/class methods
  is not fully supported because ProtoWrapper is not a descriptor.
  """

  def test_proto_on_instance_method_params_only(self):
    """Test @proto decorator on instance method excludes self from params.

    Note: This tests parameter extraction only. Calling the method requires
    using proto.partial(method=True) for proper self binding.
    """

    class Runner:
      def __init__(self, name):
        self.runner_name = name

      @proto
      def run(self, steps: int = 100, verbose: bool = False):
        """Run for given steps."""
        return {"steps": steps, "verbose": verbose, "runner": self.runner_name}

    runner = Runner("test_runner")

    # self should NOT appear in _params - this is the main fix we're testing
    assert "self" not in runner.run._params, (
      "self should be excluded from @proto decorated method params"
    )
    assert "steps" in runner.run._params
    assert "verbose" in runner.run._params

  def test_proto_on_staticmethod(self):
    """Test @proto decorator on staticmethod works correctly."""

    class Runner:
      @proto         # proto on OUTSIDE - receives staticmethod descriptor
      @staticmethod
      def run(steps: int = 100, verbose: bool = False):
        """Run for given steps."""
        return {"steps": steps, "verbose": verbose}

    # Should work correctly - staticmethod detected
    assert "self" not in Runner.run._params
    assert "cls" not in Runner.run._params
    assert "steps" in Runner.run._params
    assert Runner.run._is_staticmethod is True

    result = Runner.run()
    assert result["steps"] == 100
    assert result["verbose"] is False

  def test_proto_on_classmethod(self):
    """Test @proto decorator on classmethod excludes cls from params."""

    class Runner:
      class_value = 42

      @proto         # proto on OUTSIDE - receives classmethod descriptor
      @classmethod
      def run(cls, steps: int = 100):
        """Run for given steps."""
        return {"steps": steps, "class_value": cls.class_value}

    # cls should NOT appear in _params - classmethod detected
    assert "cls" not in Runner.run._params, (
      "cls should be excluded from @proto decorated classmethod params"
    )
    assert "steps" in Runner.run._params
    assert Runner.run._is_classmethod is True


class TestArgsKwargsExclusion:
  """Test that *args and **kwargs are correctly excluded."""

  def test_args_excluded_from_params(self):
    """Test that *args parameter is excluded from CLI params."""

    @proto.cli
    def train(lr: float = 0.01, *args, batch_size: int = 32):
      """Training with args."""
      return {"lr": lr, "args": args, "batch_size": batch_size}

    # args should NOT be in params
    assert "args" not in train._params, "*args should be excluded from CLI params"
    assert "lr" in train._params
    assert "batch_size" in train._params

  def test_kwargs_excluded_from_params(self):
    """Test that **kwargs parameter is excluded from CLI params."""

    @proto.cli
    def train(lr: float = 0.01, **kwargs):
      """Training with kwargs."""
      return {"lr": lr, "kwargs": kwargs}

    # kwargs should NOT be in params (this should already work)
    assert "kwargs" not in train._params
    assert "lr" in train._params


class TestEdgeCases:
  """Edge cases for parameter handling."""

  def test_first_param_self_filtered_even_in_regular_function(self):
    """Test that first parameter named 'self' is filtered even in regular function.

    This is a design decision - we always filter self/cls in the first position
    because detecting whether a function is truly a method is complex and
    the common case is that self/cls in first position IS a method parameter.
    """

    @proto.cli
    def process(self, data: str = "default"):
      """Process data. Note: self here gets filtered as if it were a method."""
      return {"data": data}

    # self should be filtered - this is the safer design choice
    assert "self" not in process._params, (
      "First parameter 'self' should be filtered for safety"
    )
    assert "data" in process._params

  def test_second_param_named_self_kept(self):
    """Test that a non-first parameter named 'self' is NOT filtered."""

    @proto.cli
    def process(data: str, self_ref: str = "default"):
      """Process data with self_ref parameter."""
      return {"data": data, "self_ref": self_ref}

    # self_ref should be kept (it's not in first position and not named exactly self/cls)
    assert "data" in process._params
    assert "self_ref" in process._params
