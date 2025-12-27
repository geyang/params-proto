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


def test_proto_envvar_class_based():
  """Test EnvVar resolution works for proto-decorated classes (not just functions)."""
  import os

  from params_proto import EnvVar, proto

  os.environ["AVP_IP"] = "192.168.1.1"
  os.environ["CAMERA_PATH"] = "/dev/video0"

  try:

    @proto.prefix
    class Connection:
      ip: str = EnvVar @ "AVP_IP" | "10.11.106.153"
      port: int = EnvVar @ "PORT" | 5000  # Not set, uses default

    @proto.prefix
    class Camera:
      path: str = EnvVar @ "CAMERA_PATH" | "/dev/video1"
      fps: int = 30  # Regular default (no env var)

    # Test direct class attribute access - should return resolved values
    assert Connection.ip == "192.168.1.1", f"Expected env value, got {Connection.ip}"
    assert Connection.port == 5000, f"Expected default, got {Connection.port}"
    assert Camera.path == "/dev/video0", f"Expected env value, got {Camera.path}"
    assert Camera.fps == 30, f"Expected default, got {Camera.fps}"

    # Verify types
    assert isinstance(Connection.ip, str), "ip should be str"
    assert isinstance(Connection.port, int), "port should be int"
    assert isinstance(Camera.path, str), "path should be str"
    assert isinstance(Camera.fps, int), "fps should be int"

  finally:
    del os.environ["AVP_IP"]
    del os.environ["CAMERA_PATH"]


def test_proto_envvar_class_with_type_conversion():
  """Test EnvVar type conversion works for proto-decorated classes."""
  import os

  from params_proto import EnvVar, proto

  os.environ["PORT"] = "8080"
  os.environ["DEBUG"] = "true"
  os.environ["RATIO"] = "0.75"

  try:

    @proto.prefix
    class ServerConfig:
      port: int = EnvVar @ "PORT" | 3000  # String "8080" → int 8080
      debug: bool = EnvVar @ "DEBUG" | False  # String "true" → bool True
      ratio: float = EnvVar @ "RATIO" | 0.5  # String "0.75" → float 0.75

    assert ServerConfig.port == 8080, f"Expected 8080, got {ServerConfig.port}"
    assert ServerConfig.debug is True, f"Expected True, got {ServerConfig.debug}"
    assert ServerConfig.ratio == 0.75, f"Expected 0.75, got {ServerConfig.ratio}"

    # Verify types
    assert isinstance(ServerConfig.port, int), "port should be int"
    assert isinstance(ServerConfig.debug, bool), "debug should be bool"
    assert isinstance(ServerConfig.ratio, float), "ratio should be float"

  finally:
    del os.environ["PORT"]
    del os.environ["DEBUG"]
    del os.environ["RATIO"]


def test_envvar_get_with_dtype():
  """Test EnvVar.get() applies dtype conversion correctly.

  This tests the direct .get() method, not via @proto decoration.
  Previously, dtype was stored but not applied in .get().
  """
  import os

  from params_proto import EnvVar

  os.environ["PORT"] = "9000"
  os.environ["THRESHOLD"] = "0.75"
  os.environ["DEBUG"] = "true"
  os.environ["ENABLED"] = "1"
  os.environ["DISABLED"] = "false"

  try:
    # Test int conversion
    port = EnvVar("PORT", dtype=int, default=8012).get()
    assert port == 9000, f"Expected 9000, got {port}"
    assert isinstance(port, int), f"Expected int, got {type(port)}"

    # Test float conversion
    threshold = EnvVar("THRESHOLD", dtype=float, default=0.5).get()
    assert threshold == 0.75, f"Expected 0.75, got {threshold}"
    assert isinstance(threshold, float), f"Expected float, got {type(threshold)}"

    # Test bool conversion - "true"
    debug = EnvVar("DEBUG", dtype=bool, default=False).get()
    assert debug is True, f"Expected True, got {debug}"
    assert isinstance(debug, bool), f"Expected bool, got {type(debug)}"

    # Test bool conversion - "1"
    enabled = EnvVar("ENABLED", dtype=bool, default=False).get()
    assert enabled is True, f"Expected True, got {enabled}"

    # Test bool conversion - "false"
    disabled = EnvVar("DISABLED", dtype=bool, default=True).get()
    assert disabled is False, f"Expected False, got {disabled}"

    # Test default value when env var not set
    missing = EnvVar("MISSING_VAR", dtype=int, default=42).get()
    assert missing == 42, f"Expected 42 (default), got {missing}"
    assert isinstance(missing, int), f"Expected int, got {type(missing)}"

    # Test without dtype - should return string
    port_str = EnvVar("PORT", default="8012").get()
    assert port_str == "9000", f"Expected '9000', got {port_str}"
    assert isinstance(port_str, str), f"Expected str, got {type(port_str)}"

  finally:
    del os.environ["PORT"]
    del os.environ["THRESHOLD"]
    del os.environ["DEBUG"]
    del os.environ["ENABLED"]
    del os.environ["DISABLED"]


def test_envvar_get_with_dtype_template():
  """Test EnvVar.get() applies dtype with template syntax."""
  import os

  from params_proto import EnvVar

  os.environ["COUNT"] = "100"

  try:
    # Test with $ prefix template
    count = EnvVar("$COUNT", dtype=int, default=0).get()
    assert count == 100, f"Expected 100, got {count}"
    assert isinstance(count, int), f"Expected int, got {type(count)}"

  finally:
    del os.environ["COUNT"]


def test_proto_envvar_inheritance():
  """Test EnvVar resolution works for inherited fields in proto-decorated classes.

  This tests that:
  1. EnvVar fields from base classes are properly inherited
  2. Type conversion (dtype) works for inherited EnvVar fields
  3. Both inherited and child EnvVar fields coexist correctly
  """
  import os

  from params_proto import EnvVar, proto

  os.environ["BASE_HOST"] = "192.168.1.1"
  os.environ["BASE_PORT"] = "8080"
  os.environ["BASE_DEBUG"] = "true"
  os.environ["BASE_RATIO"] = "0.75"
  os.environ["CHILD_TIMEOUT"] = "30"
  os.environ["CHILD_ENABLED"] = "1"

  try:
    # Base class with EnvVar fields (not decorated)
    class BaseConfig:
      host: str = EnvVar @ "BASE_HOST" | "localhost"
      port: int = EnvVar @ "BASE_PORT" | 3000
      debug: bool = EnvVar @ "BASE_DEBUG" | False
      ratio: float = EnvVar @ "BASE_RATIO" | 0.5

    @proto.prefix
    class AppConfig(BaseConfig):
      timeout: int = EnvVar @ "CHILD_TIMEOUT" | 10
      enabled: bool = EnvVar @ "CHILD_ENABLED" | False

    # Test inherited fields - values
    assert AppConfig.host == "192.168.1.1", f"Expected '192.168.1.1', got {AppConfig.host}"
    assert AppConfig.port == 8080, f"Expected 8080, got {AppConfig.port}"
    assert AppConfig.debug is True, f"Expected True, got {AppConfig.debug}"
    assert AppConfig.ratio == 0.75, f"Expected 0.75, got {AppConfig.ratio}"

    # Test inherited fields - types
    assert isinstance(AppConfig.host, str), f"host should be str, got {type(AppConfig.host)}"
    assert isinstance(AppConfig.port, int), f"port should be int, got {type(AppConfig.port)}"
    assert isinstance(AppConfig.debug, bool), f"debug should be bool, got {type(AppConfig.debug)}"
    assert isinstance(AppConfig.ratio, float), f"ratio should be float, got {type(AppConfig.ratio)}"

    # Test child fields - values
    assert AppConfig.timeout == 30, f"Expected 30, got {AppConfig.timeout}"
    assert AppConfig.enabled is True, f"Expected True, got {AppConfig.enabled}"

    # Test child fields - types
    assert isinstance(AppConfig.timeout, int), f"timeout should be int, got {type(AppConfig.timeout)}"
    assert isinstance(AppConfig.enabled, bool), f"enabled should be bool, got {type(AppConfig.enabled)}"

  finally:
    for key in ["BASE_HOST", "BASE_PORT", "BASE_DEBUG", "BASE_RATIO", "CHILD_TIMEOUT", "CHILD_ENABLED"]:
      os.environ.pop(key, None)


def test_proto_envvar_inheritance_fallback():
  """Test EnvVar fallback values work correctly for inherited fields."""
  import os

  from params_proto import EnvVar, proto

  # Only set some env vars to test fallback behavior
  os.environ["BASE_HOST"] = "10.0.0.1"
  # BASE_PORT not set - should use fallback

  try:
    class BaseConfig:
      host: str = EnvVar @ "BASE_HOST" | "localhost"
      port: int = EnvVar @ "BASE_PORT" | 3000  # Should fallback to 3000

    @proto.prefix
    class AppConfig(BaseConfig):
      timeout: int = EnvVar @ "APP_TIMEOUT" | 60  # Should fallback to 60

    # Inherited field with env var set
    assert AppConfig.host == "10.0.0.1", f"Expected '10.0.0.1', got {AppConfig.host}"

    # Inherited field with fallback
    assert AppConfig.port == 3000, f"Expected 3000 (fallback), got {AppConfig.port}"
    assert isinstance(AppConfig.port, int), f"port should be int, got {type(AppConfig.port)}"

    # Child field with fallback
    assert AppConfig.timeout == 60, f"Expected 60 (fallback), got {AppConfig.timeout}"
    assert isinstance(AppConfig.timeout, int), f"timeout should be int, got {type(AppConfig.timeout)}"

  finally:
    os.environ.pop("BASE_HOST", None)


def test_proto_prefix_envvar_instantiation():
  """Test that @proto.prefix classes with EnvVar can be instantiated.

  Regression test for: AttributeError: '_EnvVar' object has no attribute '__get__'

  The _EnvVar object returned by `EnvVar @ 'HOST' | 'localhost'` must be resolved
  before @proto.prefix stores it, otherwise instantiation fails because _EnvVar
  is not a descriptor (missing __get__).
  """
  import os

  from params_proto import EnvVar, proto

  os.environ["HOST"] = "test-host"

  try:

    @proto.prefix
    class Config:
      host: str = EnvVar @ "HOST" | "localhost"

    # This should not raise: AttributeError: '_EnvVar' object has no attribute '__get__'
    c = Config()

    # Verify the value is resolved correctly
    assert c.host == "test-host", f"Expected 'test-host', got {c.host}"
    assert isinstance(c.host, str), f"host should be str, got {type(c.host)}"

    # Also verify class-level access works
    assert Config.host == "test-host", f"Expected 'test-host', got {Config.host}"

  finally:
    os.environ.pop("HOST", None)


def test_proto_prefix_envvar_instantiation_with_fallback():
  """Test @proto.prefix with EnvVar fallback can be instantiated when env var not set."""
  import os

  from params_proto import EnvVar, proto

  # Ensure env var is NOT set
  os.environ.pop("MISSING_HOST", None)

  @proto.prefix
  class Config:
    host: str = EnvVar @ "MISSING_HOST" | "default-host"

  # This should not raise
  c = Config()

  # Verify the fallback value is used
  assert c.host == "default-host", f"Expected 'default-host', got {c.host}"
  assert Config.host == "default-host", f"Expected 'default-host', got {Config.host}"
