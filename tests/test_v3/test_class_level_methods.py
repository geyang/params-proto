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
