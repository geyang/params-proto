"""Test super() calls inside @proto and @proto.prefix decorated classes."""

from params_proto import proto


# Test 1: Basic inheritance with @proto
class BaseClass:
    base_value: int = 10

    def base_method(self):
        return f"base: {self.base_value}"


@proto
class ChildProto(BaseClass):
    child_value: int = 20

    def child_method(self):
        base_result = super().base_method()
        return f"{base_result}, child: {self.child_value}"


# Test 2: Inheritance with @proto.prefix
class BaseConfig:
    lr: float = 0.01

    def describe(self):
        return f"lr={self.lr}"


@proto.prefix
class ChildConfig(BaseConfig):
    batch_size: int = 32

    def describe(self):
        base = super().describe()
        return f"{base}, batch_size={self.batch_size}"


# Test 3: Inheritance between two @proto decorated classes
@proto
class ParentProto:
    parent_attr: str = "parent"

    def get_info(self):
        return f"Parent: {self.parent_attr}"


@proto
class GrandchildProto(ParentProto):
    grandchild_attr: str = "grandchild"

    def get_info(self):
        parent_info = super().get_info()
        return f"{parent_info}, Grandchild: {self.grandchild_attr}"


# Test 4: __post_init__ with super() (undecorated base)
class BaseWithInit:
    value: int = 1
    processed: bool = False

    def __post_init__(self):
        self.processed = True


@proto
class ChildWithInit(BaseWithInit):
    multiplier: int = 2
    child_processed: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.child_processed = True
        self.value *= self.multiplier


def main():
    print("=" * 60)
    print("Testing super() in @proto and @proto.prefix classes")
    print("=" * 60)

    # Test 1
    print("\n--- Test 1: Basic inheritance with @proto ---")
    try:
        child = ChildProto()
        print(f"child class: {child.__class__}")
        print(f"child MRO: {child.__class__.__mro__}")
        print(f"ChildProto: {ChildProto}")
        print(f"ChildProto MRO: {ChildProto.__mro__}")
        print(f"isinstance(child, ChildProto): {isinstance(child, ChildProto)}")
        print(f"isinstance(child, BaseClass): {isinstance(child, BaseClass)}")
        result = child.child_method()
        print(f"Result: {result}")
        assert "base: 10" in result, f"Expected 'base: 10' in result"
        assert "child: 20" in result, f"Expected 'child: 20' in result"
        print("✓ PASSED")
    except Exception as e:
        import traceback
        print(f"✗ FAILED: {e}")
        traceback.print_exc()

    # Test 2
    print("\n--- Test 2: Inheritance with @proto.prefix ---")
    try:
        config = ChildConfig()
        result = config.describe()
        print(f"Result: {result}")
        assert "lr=" in result, f"Expected 'lr=' in result"
        assert "batch_size=" in result, f"Expected 'batch_size=' in result"
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 3
    print("\n--- Test 3: Inheritance between @proto classes ---")
    try:
        grandchild = GrandchildProto()
        result = grandchild.get_info()
        print(f"Result: {result}")
        assert "Parent:" in result, f"Expected 'Parent:' in result"
        assert "Grandchild:" in result, f"Expected 'Grandchild:' in result"
        print("✓ PASSED")
    except Exception as e:
        import traceback
        print(f"✗ FAILED: {e}")
        traceback.print_exc()

    # Test 4
    print("\n--- Test 4: __post_init__ with super() ---")
    try:
        obj = ChildWithInit()
        print(f"value={obj.value}, processed={obj.processed}, child_processed={obj.child_processed}")
        assert obj.processed, "Expected processed=True from parent __post_init__"
        assert obj.child_processed, "Expected child_processed=True from child __post_init__"
        assert obj.value == 2, f"Expected value=2 (1*2), got {obj.value}"
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    main()
