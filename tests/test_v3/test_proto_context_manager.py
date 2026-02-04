"""Test that @proto and @proto.prefix preserve context manager protocol."""

import pytest
from params_proto import proto


class TestContextManager:
    """Test context manager protocol preservation."""

    def test_proto_prefix_context_manager(self):
        """Test that @proto.prefix preserves __enter__ and __exit__."""

        @proto.prefix
        class RUN:
            prefix: str = "test"
            entered: bool = False
            exited: bool = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.exited = True
                return False

        run = RUN()

        # Check methods exist
        assert hasattr(run, "__enter__"), "Should have __enter__ method"
        assert hasattr(run, "__exit__"), "Should have __exit__ method"

        # Test actual context manager usage
        with run as r:
            assert r.entered, "Should have called __enter__"
            assert not r.exited, "Should not have called __exit__ yet"

        assert run.exited, "Should have called __exit__"

    def test_proto_context_manager(self):
        """Test that @proto preserves __enter__ and __exit__."""

        @proto
        class Session:
            name: str = "session"
            active: bool = False

            def __enter__(self):
                self.active = True
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.active = False
                return False

        session = Session()

        with session as s:
            assert s.active, "Should be active inside context"

        assert not session.active, "Should be inactive after context"

    def test_context_manager_with_exception(self):
        """Test context manager handles exceptions correctly."""

        @proto.prefix
        class RUN:
            prefix: str = "test"
            exc_info: tuple = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.exc_info = (exc_type, exc_val, exc_tb)
                return True  # Suppress exception

        run = RUN()

        with run:
            raise ValueError("test error")

        # Exception should have been captured
        assert run.exc_info[0] is ValueError
        assert str(run.exc_info[1]) == "test error"

    def test_context_manager_inheritance(self):
        """Test context manager works with inheritance."""

        class BaseContext:
            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.exited = True
                return False

        @proto.prefix
        class ChildRUN(BaseContext):
            prefix: str = "child"
            entered: bool = False
            exited: bool = False

        run = ChildRUN()

        with run as r:
            assert r.entered

        assert run.exited


class TestOtherProtocols:
    """Test other Python protocol methods are preserved."""

    def test_callable_protocol(self):
        """Test __call__ is preserved."""

        @proto.prefix
        class Callable:
            value: int = 0
            call_count: int = 0

            def __call__(self, x):
                self.call_count += 1
                return self.value + x

        c = Callable()
        c.value = 10

        result = c(5)
        assert result == 15
        assert c.call_count == 1

    def test_iterator_protocol(self):
        """Test __iter__ and __next__ are preserved."""

        @proto.prefix
        class Counter:
            start: int = 0
            end: int = 3
            _current: int = 0

            def __iter__(self):
                self._current = self.start
                return self

            def __next__(self):
                if self._current >= self.end:
                    raise StopIteration
                value = self._current
                self._current += 1
                return value

        counter = Counter()
        result = list(counter)
        assert result == [0, 1, 2]

    def test_container_protocol(self):
        """Test __getitem__ and __len__ are preserved."""

        @proto.prefix
        class Container:
            items: list = None

            def __post_init__(self):
                if self.items is None:
                    self.items = [1, 2, 3]

            def __getitem__(self, index):
                return self.items[index]

            def __len__(self):
                return len(self.items)

        c = Container()
        assert len(c) == 3
        assert c[0] == 1
        assert c[1] == 2
