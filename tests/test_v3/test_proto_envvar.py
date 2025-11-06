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


def test_proto_envvar_with_value():
  """Test EnvVar with environment variable set."""
  import os

  from params_proto import EnvVar, proto

  # Set environment variables that will actually be used
  os.environ["BATCH_SIZE"] = "256"
  os.environ["LR"] = "0.01"  # This one is actually used (not LEARNING_RATE)

  try:

    @proto
    def train(
      batch_size: int = EnvVar @ "BATCH_SIZE",  # Reads "256" from BATCH_SIZE
      learning_rate: float = EnvVar @ "LR" | 0.001,  # Reads "0.01" from LR
      epochs: int = 10,  # Regular default (no env var)
    ):
      """Train model with environment variable configuration."""
      return batch_size, learning_rate, epochs

    # Values should come from env vars (256, 0.01) and default (10)
    result = train()
    assert result == (256, 0.01, 10), f"Expected (256, 0.01, 10), got {result}"

    # Test override with kwargs still works
    result2 = train(batch_size=128)
    assert result2 == (128, 0.01, 10), "kwargs should override env vars"

  finally:
    # Clean up environment variables
    del os.environ["BATCH_SIZE"]
    del os.environ["LR"]


def test_proto_envvar_without_value():
  """Test EnvVar when environment variable is not set (uses fallback or None)."""
  import os

  from params_proto import EnvVar, proto

  # Make sure env vars are not set
  os.environ.pop("MISSING_VAR", None)
  os.environ.pop("ANOTHER_VAR", None)

  @proto
  def train(
    value: int = EnvVar @ "MISSING_VAR" | 100,  # Env var NOT set → fallback to 100
    port: int = EnvVar @ "ANOTHER_VAR",  # Env var NOT set, no fallback → None
  ):
    """Train with missing environment variables."""
    return value, port

  # Check fallback works when env var not set
  result = train()
  assert result == (100, None), f"Expected (100, None) - fallback and None, got {result}"


def test_proto_envvar_function_syntax():
  """Test EnvVar with function call syntax."""
  import os

  from params_proto import EnvVar, proto

  os.environ["DATABASE_URL"] = "postgres://localhost/mydb"

  try:

    @proto
    def connect(
      db_url: str = EnvVar(
        "DATABASE_URL", default="sqlite:///local.db"
      ),  # Env var with default
      timeout: int = EnvVar("DB_TIMEOUT", default=30),  # Missing env var, use default
    ):
      """Connect to database."""
      return db_url, timeout

    result = connect()
    assert result == ("postgres://localhost/mydb", 30), (
      f"Expected database URL from env and default timeout, got {result}"
    )

  finally:
    del os.environ["DATABASE_URL"]


def test_proto_envvar_simple_name():
  """Test EnvVar with simple environment variable name."""
  import os

  from params_proto import EnvVar, proto

  os.environ["API_KEY"] = "secret-key-123"

  try:

    @proto
    def api_call(
      api_key: str = EnvVar @ "API_KEY",  # Simple var name
    ):
      """Make API call."""
      return api_key

    result = api_call()
    assert result == "secret-key-123", f"Expected 'secret-key-123', got {result}"

  finally:
    del os.environ["API_KEY"]


def test_proto_envvar_dollar_prefix():
  """Test EnvVar with $VAR_NAME syntax."""
  import os

  from params_proto import EnvVar, proto

  os.environ["DATA_DIR"] = "/mnt/data"

  try:

    @proto
    def load_data(
      data_dir: str = EnvVar @ "$DATA_DIR",  # Dollar prefix syntax
    ):
      """Load data from directory."""
      return data_dir

    result = load_data()
    assert result == "/mnt/data", f"Expected '/mnt/data', got {result}"

  finally:
    del os.environ["DATA_DIR"]


def test_proto_envvar_braces_syntax():
  """Test EnvVar with ${VAR_NAME} syntax."""
  import os

  from params_proto import EnvVar, proto

  os.environ["LOG_DIR"] = "/var/log"

  try:

    @proto
    def setup_logging(
      log_dir: str = EnvVar("${LOG_DIR}", default="/tmp/logs"),  # Braces syntax
    ):
      """Setup logging directory."""
      return log_dir

    result = setup_logging()
    assert result == "/var/log", f"Expected '/var/log', got {result}"

  finally:
    del os.environ["LOG_DIR"]


def test_proto_envvar_multiple_vars():
  """Test EnvVar with multiple variables in template."""
  import os

  from params_proto import EnvVar, proto

  os.environ["BASE_DIR"] = "/data"
  os.environ["PROJECT_NAME"] = "myproject"

  try:

    @proto
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


def test_proto_envvar_type_conversion():
  """Test EnvVar type conversion for different types."""
  import os

  from params_proto import EnvVar, proto

  os.environ["PORT"] = "8080"
  os.environ["RATIO"] = "0.75"
  os.environ["ENABLED"] = "true"

  try:

    @proto
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


def test_proto_envvar_pipe_operator():
  """Test EnvVar with pipe operator for default values."""
  import os

  from params_proto import EnvVar, proto

  # Set only one env var to test both cases
  os.environ["BATCH_SIZE"] = "256"
  # LR and EPOCHS are NOT set - should use defaults

  try:

    @proto
    def train(
      batch_size: int = EnvVar @ "BATCH_SIZE" | 128,  # Env var set → 256
      learning_rate: float = EnvVar @ "LR" | 0.001,  # Env var NOT set → fallback to 0.001
      epochs: int = EnvVar @ "EPOCHS" | 10,  # Env var NOT set → fallback to 10
    ):
      """Train with pipe operator defaults."""
      return batch_size, learning_rate, epochs

    result = train()
    assert result == (256, 0.001, 10), f"Expected (256, 0.001, 10), got {result}"

    # Verify fallback actually works by checking types
    assert isinstance(result[0], int) and result[0] == 256, "BATCH_SIZE from env"
    assert isinstance(result[1], float) and result[1] == 0.001, "LR fallback to default"
    assert isinstance(result[2], int) and result[2] == 10, "EPOCHS fallback to default"

  finally:
    del os.environ["BATCH_SIZE"]


def test_proto_envvar_complex_templates():
  """Test EnvVar with complex template strings containing multiple variables."""
  import os

  from params_proto import EnvVar, proto

  os.environ["USER"] = "alice"
  os.environ["PROJECT"] = "ml-research"
  os.environ["EXPERIMENT"] = "exp001"
  os.environ["VERSION"] = "v2"

  try:

    @proto
    def configure(
      # Complex path with multiple vars (use ${VAR} syntax for templates not starting with $)
      workspace: str = EnvVar("/home/${USER}/projects/${PROJECT}", default="/tmp"),
      # Even more complex with subdirectories
      checkpoint_dir: str = EnvVar("/data/${PROJECT}/${EXPERIMENT}/checkpoints/${VERSION}", default="/tmp/checkpoints"),
      # Mixed braces and dollar syntax (starts with $ so both syntaxes work)
      log_file: str = EnvVar("${USER}_${PROJECT}_$EXPERIMENT.log", default="default.log"),
    ):
      """Configure paths with complex templates."""
      return workspace, checkpoint_dir, log_file

    result = configure()
    assert result == (
      "/home/alice/projects/ml-research",
      "/data/ml-research/exp001/checkpoints/v2",
      "alice_ml-research_exp001.log",
    ), f"Expected complex template expansion, got {result}"

  finally:
    del os.environ["USER"]
    del os.environ["PROJECT"]
    del os.environ["EXPERIMENT"]
    del os.environ["VERSION"]


def test_proto_envvar_template_with_missing_vars():
  """Test EnvVar template behavior when some variables are missing."""
  import os

  from params_proto import EnvVar, proto

  # Set only some of the required vars
  os.environ["BASE"] = "/data"
  # SUBDIR is NOT set - will be replaced with empty string

  try:

    @proto
    def get_path(
      # Template with missing var - missing vars become empty strings
      data_path: str = EnvVar("$BASE/$SUBDIR/output", default="/tmp/output"),
    ):
      """Get path with potentially missing vars."""
      return data_path

    result = get_path()
    # Missing SUBDIR becomes empty string: "/data//output"
    assert result == "/data//output", f"Expected /data//output (missing var → empty), got {result}"

  finally:
    del os.environ["BASE"]


def test_proto_envvar_explicit_dtype():
  """Test EnvVar with explicit dtype parameter for type control."""
  import os

  from params_proto import EnvVar, proto

  os.environ["WORKERS"] = "4"
  os.environ["THRESHOLD"] = "0.75"

  try:

    @proto
    def configure(
      # Union type with explicit dtype=int (converts "4" → 4)
      workers: int | str = EnvVar("WORKERS", dtype=int, default="auto"),
      # Explicit dtype overrides annotation (str → float)
      threshold: str = EnvVar("THRESHOLD", dtype=float, default=0.5),
    ):
      """Configure with explicit type conversion."""
      return workers, threshold

    result = configure()
    # Workers should be int (not str) because dtype=int
    assert result == (4, 0.75), f"Expected (4, 0.75), got {result}"
    assert isinstance(result[0], int), "workers should be int due to dtype=int"
    assert isinstance(result[1], float), "threshold should be float due to dtype=float"

  finally:
    del os.environ["WORKERS"]
    del os.environ["THRESHOLD"]
