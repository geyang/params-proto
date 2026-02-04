"""Minimal test for super() issue."""

from params_proto import proto


class Base:
    def method(self):
        return "base"


@proto
class Child(Base):
    attr: int = 1

    def method(self):
        # This should call Base.method()
        return f"child + {super().method()}"


# Test without instantiation
print("Child class:", Child)
print("Child id:", id(Child))
print("Child.__mro__:", Child.__mro__)
print("Child.__proto_original_class__:", Child.__proto_original_class__)
print("Original class id:", id(Child.__proto_original_class__))
print("Same class?", Child is Child.__proto_original_class__)

# Test instantiation
print("\nCreating instance...")
child = Child()
print("Instance class:", child.__class__)
print("Instance type:", type(child))

# Check what method is on the instance
print("\nMethod on instance:", child.method)

# Try calling
print("\nCalling method...")
try:
    result = child.method()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

    # Debug: manually call the original method
    print("\nDebug: Trying to call original method directly...")
    original_cls = Child.__proto_original_class__
    print(f"Original class: {original_cls}")
    print(f"Original method: {original_cls.method}")

    # Create instance via original class
    from params_proto.proto import _SINGLETONS
    print(f"\nSINGLETONS: {_SINGLETONS}")
