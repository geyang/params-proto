"""Tests for @proto decorator on classes that contain classmethod/staticmethod."""

from params_proto import proto


class TestClassLevelProtoWithMethods:
  """Test @proto decorator on class that contains classmethod/staticmethod."""

  def test_proto_class_with_classmethod(self):
    """Test @proto on class with @classmethod inside."""

    @proto
    class Config:
      lr: float = 0.01
      batch_size: int = 32

      @classmethod
      def from_preset(cls, preset: str = "default"):
        """Create config from preset."""
        if preset == "large":
          cls.lr = 0.001
          cls.batch_size = 128
        return cls()

    # Class attributes should work
    assert Config.lr == 0.01
    assert Config.batch_size == 32

    # Classmethod should still be callable
    config = Config.from_preset("large")
    assert config.lr == 0.001
    assert config.batch_size == 128

  def test_proto_class_with_staticmethod(self):
    """Test @proto on class with @staticmethod inside."""

    @proto
    class Config:
      lr: float = 0.01
      batch_size: int = 32

      @staticmethod
      def validate_lr(lr: float) -> bool:
        """Validate learning rate."""
        return 0 < lr < 1.0

    # Class attributes should work
    assert Config.lr == 0.01
    assert Config.batch_size == 32

    # Staticmethod should still work
    assert Config.validate_lr(0.01) is True
    assert Config.validate_lr(1.5) is False

  def test_proto_class_with_instance_method(self):
    """Test @proto on class with regular instance method inside."""

    @proto
    class Config:
      lr: float = 0.01
      batch_size: int = 32

      def summary(self):
        """Return config summary."""
        return f"lr={self.lr}, batch_size={self.batch_size}"

    # Class attributes should work
    assert Config.lr == 0.01

    # Instance method should work
    config = Config()
    assert config.summary() == "lr=0.01, batch_size=32"

  def test_proto_class_classmethod_receives_correct_cls(self):
    """Test that @classmethod in @proto class receives the correct cls."""

    @proto
    class Config:
      name: str = "base"

      @classmethod
      def get_name(cls):
        return cls.name

    assert Config.get_name() == "base"
    Config.name = "modified"
    assert Config.get_name() == "modified"

  def test_proto_class_methods_not_in_annotations(self):
    """Test that methods are not included in __proto_annotations__."""

    @proto
    class Config:
      lr: float = 0.01

      @classmethod
      def factory(cls):
        return cls()

      @staticmethod
      def helper():
        return True

      def instance_method(self):
        return self.lr

    # Only lr should be in annotations, not methods
    annotations = Config.__proto_annotations__
    assert "lr" in annotations
    assert "factory" not in annotations
    assert "helper" not in annotations
    assert "instance_method" not in annotations


class TestProtoPrefixWithMethods:
  """Test @proto.prefix decorator on class that contains methods."""

  def test_proto_prefix_with_staticmethod(self):
    """Test @proto.prefix on class with @staticmethod inside.

    This tests the fix for staticmethod receiving incorrect self argument.
    """

    @proto.prefix
    class Config:
      lr: float = 0.01

      @staticmethod
      def validate_lr(lr: float) -> bool:
        """Validate learning rate."""
        return 0 < lr < 1.0

    obj = Config()

    # Staticmethod should work correctly without receiving self
    assert obj.validate_lr(0.01) is True
    assert obj.validate_lr(1.5) is False
    assert Config.validate_lr(0.01) is True

  def test_proto_prefix_with_classmethod(self):
    """Test @proto.prefix on class with @classmethod inside."""

    @proto.prefix
    class Config:
      lr: float = 0.01

      @classmethod
      def get_lr(cls):
        return cls.lr

    obj = Config()

    # Classmethod should have access to config attributes via instance
    assert obj.get_lr() == 0.01

    # Note: classmethods on instances are bound to the instance,
    # so they see instance attributes (not class-level updates)
    obj.lr = 0.001
    assert obj.get_lr() == 0.001

  def test_proto_prefix_with_instance_method(self):
    """Test @proto.prefix on class with regular instance method."""

    @proto.prefix
    class Config:
      lr: float = 0.01

      def summary(self):
        return f"lr={self.lr}"

    obj = Config()
    assert obj.summary() == "lr=0.01"

  def test_proto_prefix_all_method_types(self):
    """Test @proto.prefix with all method types together."""

    @proto.prefix
    class Config:
      value: str = "default"

      @staticmethod
      def static_method(x):
        return x * 2

      @classmethod
      def class_method(cls):
        return cls.value

      def instance_method(self):
        return self.value.upper()

    obj = Config()

    # All method types should work correctly
    assert obj.static_method(21) == 42
    assert obj.class_method() == "default"
    assert obj.instance_method() == "DEFAULT"

  def test_proto_prefix_inherited_staticmethod(self):
    """Test @proto.prefix with staticmethod inherited from non-proto base."""

    class Base:
      @staticmethod
      def static_method(value):
        return value

    @proto.prefix
    class Child(Base):
      config: str = "default"

    obj = Child()

    # Inherited staticmethod should work correctly
    assert obj.static_method(42) == 42
    assert Child.static_method(42) == 42


class TestPostInit:
  """Test __post_init__ hook for @proto classes."""

  def test_post_init_called(self):
    """Test that __post_init__ is called after instance creation."""
    calls = []

    @proto
    class Config:
      lr: float = 0.01

      def __post_init__(self):
        calls.append("post_init")

    Config()
    assert calls == ["post_init"]

  def test_post_init_has_access_to_attributes(self):
    """Test that __post_init__ can access all config attributes."""

    @proto
    class Config:
      lr: float = 0.01
      batch_size: int = 32
      lr_squared: float = None

      def __post_init__(self):
        self.lr_squared = self.lr ** 2

    c = Config(lr=0.5)
    assert c.lr_squared == 0.25

  def test_post_init_validation(self):
    """Test using __post_init__ for validation."""

    @proto
    class Config:
      lr: float = 0.01

      def __post_init__(self):
        if self.lr > 1:
          raise ValueError("lr must be <= 1")

    # Valid config works
    Config(lr=0.5)

    # Invalid config raises
    try:
      Config(lr=2.0)
      assert False, "Should have raised ValueError"
    except ValueError as e:
      assert "lr must be <= 1" in str(e)

  def test_post_init_with_proto_prefix(self):
    """Test __post_init__ with @proto.prefix classes."""
    calls = []

    @proto.prefix
    class Config:
      lr: float = 0.01

      def __post_init__(self):
        calls.append(f"post_init lr={self.lr}")

    obj = Config()
    assert calls == ["post_init lr=0.01"]

  def test_post_init_computed_attributes(self):
    """Test using __post_init__ for computed attributes."""

    @proto
    class TrainConfig:
      batch_size: int = 32
      num_batches: int = 100
      total_samples: int = None

      def __post_init__(self):
        self.total_samples = self.batch_size * self.num_batches

    c = TrainConfig(batch_size=64, num_batches=50)
    assert c.total_samples == 3200
