"""Test string conversion utilities for params-proto.

Tests conversion from Python naming conventions (PascalCase, camelCase, snake_case)
to CLI naming conventions (kebab-case).
"""

import pytest


# Test cases for different naming patterns
TEST_CASES = [
    # Simple PascalCase
    ("Train", "train"),
    ("Evaluate", "evaluate"),
    ("Export", "export"),

    # camelCase
    ("trainModel", "train-model"),
    ("getUrl", "get-url"),
    ("parseJson", "parse-json"),

    # Consecutive capitals (acronyms at start)
    ("HTTPServer", "http-server"),
    ("XMLParser", "xml-parser"),
    ("URLValidator", "url-validator"),
    ("IOError", "io-error"),

    # Consecutive capitals (acronyms in middle)
    ("parseHTMLDocument", "parse-html-document"),
    ("getURLPath", "get-url-path"),

    # Consecutive capitals (acronyms at end)
    ("getHTTP", "get-http"),
    ("parseXML", "parse-xml"),

    # Mixed patterns
    ("MLModel", "ml-model"),
    ("DeepQNetwork", "deep-q-network"),
    ("GPT4Model", "gpt4-model"),

    # Numbers
    ("Model2D", "model2-d"),
    ("Layer3Conv", "layer3-conv"),
    ("ResNet50", "res-net50"),

    # Edge cases
    ("A", "a"),
    ("AB", "ab"),
    ("ABC", "abc"),
    ("a", "a"),
    ("aB", "a-b"),
    ("aBC", "a-bc"),

    # Already lowercase/kebab
    ("train", "train"),
    ("train-model", "train-model"),

    # snake_case input (should convert underscores to hyphens)
    ("train_model", "train-model"),
    ("deep_q_network", "deep-q-network"),
    ("http_server", "http-server"),
]


def pascal_to_kebab_simple(name: str) -> str:
    """Simple conversion: just lowercase and replace underscores.

    This is the current behavior in params-proto.
    """
    return name.replace("_", "-").lower()


def pascal_to_kebab_smart(name: str) -> str:
    """Smart conversion: handles PascalCase, camelCase, and acronyms.

    This is similar to what tyro does, but implemented independently.
    Splits on:
    - Lowercase/digit followed by uppercase (camelCase boundary)
    - Uppercase followed by lowercase, not at start (PascalCase/acronym boundary)
    """
    import re

    # Insert hyphens before capitals that mark word boundaries
    # Pattern 1: lowercase/digit followed by uppercase (e.g., myHTTP, model2D)
    result = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)

    # Pattern 2: uppercase followed by lowercase, but not at start (e.g., HTTPServer -> HTTP-Server)
    result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", result)

    # Replace underscores with hyphens
    result = result.replace("_", "-")

    # Convert to lowercase
    return result.lower()


class TestSimpleConversion:
    """Test simple lowercase conversion (current params-proto behavior)."""

    def test_simple_pascalcase(self):
        """Simple PascalCase words just get lowercased."""
        assert pascal_to_kebab_simple("Train") == "train"
        assert pascal_to_kebab_simple("Evaluate") == "evaluate"
        assert pascal_to_kebab_simple("Export") == "export"

    def test_simple_with_acronyms(self):
        """Acronyms are NOT split - just lowercased."""
        # Current behavior: no splitting
        assert pascal_to_kebab_simple("HTTPServer") == "httpserver"
        assert pascal_to_kebab_simple("XMLParser") == "xmlparser"
        assert pascal_to_kebab_simple("MLModel") == "mlmodel"

    def test_simple_with_underscores(self):
        """Underscores are converted to hyphens."""
        assert pascal_to_kebab_simple("train_model") == "train-model"
        assert pascal_to_kebab_simple("deep_q_network") == "deep-q-network"

    def test_simple_already_lowercase(self):
        """Already lowercase strings are unchanged."""
        assert pascal_to_kebab_simple("train") == "train"
        assert pascal_to_kebab_simple("evaluate") == "evaluate"


class TestSmartConversion:
    """Test smart PascalCase conversion with acronym handling."""

    @pytest.mark.parametrize("input_name,expected", TEST_CASES)
    def test_conversion_cases(self, input_name, expected):
        """Test various conversion cases."""
        result = pascal_to_kebab_smart(input_name)
        assert result == expected, f"Failed: {input_name} -> {result} (expected {expected})"

    def test_simple_pascalcase(self):
        """Simple PascalCase is converted to kebab-case."""
        assert pascal_to_kebab_smart("Train") == "train"
        assert pascal_to_kebab_smart("Evaluate") == "evaluate"
        assert pascal_to_kebab_smart("TrainModel") == "train-model"

    def test_camelcase(self):
        """camelCase is converted to kebab-case."""
        assert pascal_to_kebab_smart("trainModel") == "train-model"
        assert pascal_to_kebab_smart("getUrl") == "get-url"

    def test_acronyms_at_start(self):
        """Acronyms at the start are handled correctly."""
        assert pascal_to_kebab_smart("HTTPServer") == "http-server"
        assert pascal_to_kebab_smart("XMLParser") == "xml-parser"
        assert pascal_to_kebab_smart("IOError") == "io-error"

    def test_acronyms_in_middle(self):
        """Acronyms in the middle are handled correctly."""
        assert pascal_to_kebab_smart("parseHTMLDocument") == "parse-html-document"
        assert pascal_to_kebab_smart("getURLPath") == "get-url-path"

    def test_acronyms_at_end(self):
        """Acronyms at the end are handled correctly."""
        assert pascal_to_kebab_smart("getHTTP") == "get-http"
        assert pascal_to_kebab_smart("parseXML") == "parse-xml"

    def test_ml_names(self):
        """Machine learning model names are converted correctly."""
        assert pascal_to_kebab_smart("MLModel") == "ml-model"
        assert pascal_to_kebab_smart("DeepQNetwork") == "deep-q-network"
        assert pascal_to_kebab_smart("ResNet50") == "res-net50"

    def test_numbers(self):
        """Numbers are handled correctly."""
        assert pascal_to_kebab_smart("Model2D") == "model2-d"
        assert pascal_to_kebab_smart("Layer3Conv") == "layer3-conv"

    def test_edge_cases(self):
        """Edge cases are handled correctly."""
        assert pascal_to_kebab_smart("A") == "a"
        assert pascal_to_kebab_smart("AB") == "ab"
        assert pascal_to_kebab_smart("ABC") == "abc"

    def test_snake_case_input(self):
        """snake_case is converted to kebab-case."""
        assert pascal_to_kebab_smart("train_model") == "train-model"
        assert pascal_to_kebab_smart("deep_q_network") == "deep-q-network"


class TestNamingBestPractices:
    """Test cases demonstrating best practices for class naming."""

    def test_simple_names_work_everywhere(self):
        """Simple names work well with both simple and smart conversion."""
        simple_names = ["Train", "Evaluate", "Export", "Config", "Model"]

        for name in simple_names:
            # Both converters produce the same result for simple names
            assert pascal_to_kebab_simple(name) == pascal_to_kebab_smart(name)

    def test_acronym_names_differ(self):
        """Names with acronyms produce different results."""
        acronym_names = [
            ("HTTPServer", "httpserver", "http-server"),
            ("MLModel", "mlmodel", "ml-model"),
            ("XMLParser", "xmlparser", "xml-parser"),
        ]

        for name, simple_expected, smart_expected in acronym_names:
            # Simple conversion doesn't split acronyms
            assert pascal_to_kebab_simple(name) == simple_expected
            # Smart conversion does split acronyms
            assert pascal_to_kebab_smart(name) == smart_expected

    def test_recommended_naming(self):
        """Recommended naming patterns for CLI tools."""
        # Good: simple names that work well in both systems
        good_names = ["Train", "Evaluate", "Export", "Serve", "Build"]

        for name in good_names:
            result = pascal_to_kebab_smart(name)
            assert "-" not in result or result.count("-") <= 1

        # Acceptable but may confuse: acronyms
        # Better to use: HttpServer instead of HTTPServer
        # Better to use: MlModel instead of MLModel


def test_integration_with_union_types():
    """Test that conversion works correctly with Union type subcommands."""
    from dataclasses import dataclass

    @dataclass
    class Train:
        lr: float = 0.001

    @dataclass
    class Evaluate:
        model: str = "checkpoint.pt"

    # Class names are converted to CLI commands
    class_names = ["Train", "Evaluate"]
    expected_commands = ["train", "evaluate"]

    for class_name, expected_cmd in zip(class_names, expected_commands):
        # Both converters agree on simple names
        assert pascal_to_kebab_simple(class_name) == expected_cmd
        assert pascal_to_kebab_smart(class_name) == expected_cmd


def test_prefix_conversion():
    """Test that conversion works for @proto.prefix class names."""
    # When using @proto.prefix, class names become CLI prefixes
    prefix_names = [
        ("Model", "model"),
        ("Training", "training"),
        ("Optimizer", "optimizer"),
        ("DataLoader", "data-loader"),  # Only smart conversion handles this
    ]

    for class_name, expected_kebab in prefix_names[:3]:
        # Simple names work with both converters
        assert pascal_to_kebab_simple(class_name) == expected_kebab
        assert pascal_to_kebab_smart(class_name) == expected_kebab

    # DataLoader differs between converters
    assert pascal_to_kebab_simple("DataLoader") == "dataloader"
    assert pascal_to_kebab_smart("DataLoader") == "data-loader"
