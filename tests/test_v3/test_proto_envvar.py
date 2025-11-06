from textwrap import dedent

import pytest


@pytest.fixture(autouse=True)
def clear_proto_state():
  """Clear global proto state before each test."""
  import sys
  import params_proto.proto  # Ensure module is loaded

  # Get the actual proto module (not the decorator function)
  proto_module = sys.modules["params_proto.proto"]
  # Clear the global registries
  proto_module._SINGLETONS.clear()
  proto_module._BIND_CONTEXT.clear()
  if hasattr(proto_module, "_BIND_STACK"):
    proto_module._BIND_STACK.clear()
  yield
  # Clean up after test too
  proto_module._SINGLETONS.clear()
  proto_module._BIND_CONTEXT.clear()
  if hasattr(proto_module, "_BIND_STACK"):
    proto_module._BIND_STACK.clear()


def test_proto_cli_envvar_with_value():
  """Test EnvVar with environment variable set."""
  import os

  from params_proto import EnvVar, proto

  # Set environment variable before decorating
  os.environ["BATCH_SIZE"] = "256"
  os.environ["LEARNING_RATE"] = "0.01"

  try:
    @proto.cli(prog="train")
    def train(
      batch_size: int = EnvVar @ "BATCH_SIZE",  # Read from BATCH_SIZE env var
      learning_rate: float = EnvVar @ 0.001,  # Use 0.001 as default
      epochs: int = 10,  # Regular default
    ):
      """Train model with environment variable configuration."""
      return batch_size, learning_rate, epochs

    # Check that env var was read and converted to correct type
    result = train()
    assert result == (256, 0.001, 10), f"Expected (256, 0.001, 10), got {result}"

    # Check help string
    expected = dedent("""
    usage: train [-h] [--batch-size INT] [--learning-rate FLOAT] [--epochs INT]

    Train model with environment variable configuration.

    options:
      -h, --help           show this help message and exit
      --batch-size INT     Read from BATCH_SIZE env var (default: 256)
      --learning-rate FLOAT
                           Use 0.001 as default (default: 0.001)
      --epochs INT         Regular default (default: 10)
    """)
    assert train.__help_str__ == expected, "help string is not correct"

  finally:
    # Clean up environment variables
    del os.environ["BATCH_SIZE"]
    del os.environ["LEARNING_RATE"]


def test_proto_cli_envvar_without_value():
  """Test EnvVar when environment variable is not set."""
  import os

  from params_proto import EnvVar, proto

  # Make sure env var is not set
  os.environ.pop("MISSING_VAR", None)

  @proto.cli
  def train(
    value: int = EnvVar @ "MISSING_VAR",  # Env var not set
    fallback: float = EnvVar @ 42.0,  # Use default value
  ):
    """Train with missing environment variables."""
    return value, fallback

  # Check that we get None for missing env var, and default for the other
  result = train()
  assert result == (None, 42.0), f"Expected (None, 42.0), got {result}"


def test_proto_cli_envvar_function_syntax():
  """Test EnvVar with function call syntax."""
  import os

  from params_proto import EnvVar, proto

  os.environ["DATABASE_URL"] = "postgres://localhost/mydb"

  try:
    @proto.cli
    def connect(
      db_url: str = EnvVar("DATABASE_URL", default="sqlite:///local.db"),  # Env var with default
      timeout: int = EnvVar("DB_TIMEOUT", default=30),  # Missing env var, use default
    ):
      """Connect to database."""
      return db_url, timeout

    result = connect()
    assert result == ("postgres://localhost/mydb", 30), f"Expected database URL from env and default timeout, got {result}"

  finally:
    del os.environ["DATABASE_URL"]


def test_proto_cli_envvar_simple_name():
  """Test EnvVar with simple environment variable name."""
  import os

  from params_proto import EnvVar, proto

  os.environ["API_KEY"] = "secret-key-123"

  try:
    @proto.cli
    def api_call(
      api_key: str = EnvVar @ "API_KEY",  # Simple var name
    ):
      """Make API call."""
      return api_key

    result = api_call()
    assert result == "secret-key-123", f"Expected 'secret-key-123', got {result}"

  finally:
    del os.environ["API_KEY"]


def test_proto_cli_envvar_dollar_prefix():
  """Test EnvVar with $VAR_NAME syntax."""
  import os

  from params_proto import EnvVar, proto

  os.environ["DATA_DIR"] = "/mnt/data"

  try:
    @proto.cli
    def load_data(
      data_dir: str = EnvVar @ "$DATA_DIR",  # Dollar prefix syntax
    ):
      """Load data from directory."""
      return data_dir

    result = load_data()
    assert result == "/mnt/data", f"Expected '/mnt/data', got {result}"

  finally:
    del os.environ["DATA_DIR"]


def test_proto_cli_envvar_braces_syntax():
  """Test EnvVar with ${VAR_NAME} syntax."""
  import os

  from params_proto import EnvVar, proto

  os.environ["LOG_DIR"] = "/var/log"

  try:
    @proto.cli
    def setup_logging(
      log_dir: str = EnvVar("${LOG_DIR}", default="/tmp/logs"),  # Braces syntax
    ):
      """Setup logging directory."""
      return log_dir

    result = setup_logging()
    assert result == "/var/log", f"Expected '/var/log', got {result}"

  finally:
    del os.environ["LOG_DIR"]


def test_proto_cli_envvar_multiple_vars():
  """Test EnvVar with multiple variables in template."""
  import os

  from params_proto import EnvVar, proto

  os.environ["BASE_DIR"] = "/data"
  os.environ["PROJECT_NAME"] = "myproject"

  try:
    @proto.cli
    def get_path(
      project_path: str = EnvVar @ "$BASE_DIR/$PROJECT_NAME",  # Multiple vars
    ):
      """Get project path."""
      return project_path

    result = get_path()
    assert result == "/data/myproject", f"Expected '/data/myproject', got {result}"

  finally:
    del os.environ["BASE_DIR"]
    del os.environ["PROJECT_NAME"]


def test_proto_cli_envvar_type_conversion():
  """Test EnvVar type conversion for different types."""
  import os

  from params_proto import EnvVar, proto

  os.environ["PORT"] = "8080"
  os.environ["RATIO"] = "0.75"
  os.environ["ENABLED"] = "true"

  try:
    @proto.cli
    def config(
      port: int = EnvVar @ "PORT",  # String to int
      ratio: float = EnvVar @ "RATIO",  # String to float
      enabled: bool = EnvVar @ "ENABLED",  # String to bool
    ):
      """Get configuration."""
      return port, ratio, enabled

    result = config()
    assert result == (8080, 0.75, True), f"Expected (8080, 0.75, True), got {result}"
    assert isinstance(result[0], int), "port should be int"
    assert isinstance(result[1], float), "ratio should be float"
    assert isinstance(result[2], bool), "enabled should be bool"

  finally:
    del os.environ["PORT"]
    del os.environ["RATIO"]
    del os.environ["ENABLED"]
