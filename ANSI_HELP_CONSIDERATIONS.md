# ANSI Help Text - Implementation Considerations

## Current State

We've created `ansi_help.py` with:
- Terminal width detection
- ANSI colorization (bold types, red required, cyan defaults)
- Line wrapping for long comments
- `__help_str__` remains plain text for testing

## What Else to Consider

### 1. Integration with proto.py

**Add `__ansi_str__` property:**
```python
class ProtoWrapper:
    @property
    def __ansi_str__(self):
        """Get ANSI-formatted help text."""
        from params_proto.ansi_help import get_ansi_help
        return get_ansi_help(self)
```

**Question:** Should we also add this to `ProtoClass`?

### 2. Actual CLI Execution (--help handling)

When user runs `python script.py --help`, we need to:

**Option A:** Intercept in `__call__` method
```python
def __call__(self, *args, **kwargs):
    # Check for --help before parsing
    if '--help' in sys.argv or '-h' in sys.argv:
        print(self.__ansi_str__ if sys.stdout.isatty() else self.__help_str__)
        sys.exit(0)
    # ... rest of CLI parsing
```

**Option B:** Use argparse's built-in help formatter
- More complex, would need custom formatter class
- Better integration with argparse
- Harder to control when to use ANSI vs plain

**Recommendation:** Option A is simpler and gives us full control

### 3. Environment Detection

**When to use ANSI:**
- `sys.stdout.isatty()` is True (terminal, not pipe/redirect)
- `NO_COLOR` environment variable not set
- `TERM` is not "dumb"
- Not on Windows (unless modern terminal)

**Implementation:**
```python
def should_use_ansi() -> bool:
    """Determine if ANSI codes should be used."""
    # Respect NO_COLOR standard
    if os.environ.get('NO_COLOR'):
        return False

    # Check if stdout is a terminal
    if not sys.stdout.isatty():
        return False

    # Check TERM variable
    term = os.environ.get('TERM', '')
    if term == 'dumb':
        return False

    return True
```

### 4. Testing Strategy

**Current approach:** Test `__help_str__` only (plain text)

**Do we need to test ANSI?**
- No: ANSI codes make tests brittle
- Maybe: Test that `should_use_ansi()` logic works
- Maybe: Test that ANSI codes can be stripped properly

**Recommendation:**
- Keep testing plain `__help_str__`
- Add smoke test that `__ansi_str__` doesn't crash
- Test `should_use_ansi()` with mocked environment

### 5. Configuration Options

**Should users be able to configure?**

```python
@proto.cli(
    prog='train',
    ansi=True,  # Force ANSI on/off
    wrap_width=100,  # Override terminal width
    color_scheme='custom',  # Custom colors?
)
```

**Or environment variables:**
```bash
PARAMS_PROTO_ANSI=1
PARAMS_PROTO_WIDTH=100
PARAMS_PROTO_COLORS="type:blue,required:red,default:green"
```

**Recommendation:** Start simple (auto-detect), add config if users request it

### 6. Color Scheme Customization

**Current colors:**
- Types: Bold Cyan
- (required): Bold Red
- (default: ...): Dim Cyan
- Option names: Bold

**Considerations:**
- Some terminals have different color schemes
- Red on red background is unreadable
- Colorblind users might need different colors

**Recommendation:** Ship with current scheme, document how to customize

### 7. Windows Support

**Challenges:**
- Old Windows terminals don't support ANSI
- Windows 10+ supports ANSI with `colorama` or native
- Need to detect Windows version

**Implementation:**
```python
import platform

def enable_windows_ansi():
    """Enable ANSI on Windows if possible."""
    if platform.system() != 'Windows':
        return True

    # Windows 10+ has native ANSI support
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False
```

**Recommendation:** Add Windows support if users request it

### 8. Performance

**Current approach:**
- Width detection on every call
- Colorization happens at display time

**Optimizations:**
- Cache terminal width (check on SIGWINCH?)
- Pre-compute ANSI string if parameters don't change

**Recommendation:** Current approach is fine (help is shown once per run)

### 9. Accessibility

**Considerations:**
- Screen readers might read ANSI codes
- Some users disable colors
- High contrast mode users

**Implementation:**
```python
# Respect FORCE_COLOR environment variable
if os.environ.get('FORCE_COLOR'):
    return True  # User explicitly wants color

# Respect CLICOLOR standard
if os.environ.get('CLICOLOR') == '0':
    return False
```

**Recommendation:** Follow standard environment variables

### 10. Documentation

**What to document:**
- How to disable ANSI (NO_COLOR=1)
- How __help_str__ vs __ansi_str__ works
- Why tests use __help_str__
- Terminal width detection behavior

**Example docs:**
```markdown
## Help Text Formatting

params-proto generates two versions of help text:

- `__help_str__`: Plain text (for testing, pipes, logs)
- `__ansi_str__`: Colorized with line wrapping (for terminals)

Colors automatically disabled when:
- Output is piped: `python script.py --help | grep param`
- NO_COLOR environment variable set
- TERM=dumb

Customize behavior:
```bash
# Disable colors
NO_COLOR=1 python script.py --help

# Force colors in pipes
FORCE_COLOR=1 python script.py --help | less -R
```
```

## Recommended Next Steps

1. **Add `__ansi_str__` property** to ProtoWrapper and ProtoClass
2. **Implement `should_use_ansi()`** with proper environment detection
3. **Update `__call__` method** to show ANSI help when appropriate
4. **Add smoke test** that ANSI formatting doesn't crash
5. **Document** in user guide
6. **Defer** customization features until requested

## Open Questions

1. Should `proto.cli()` accept an `ansi=` parameter to override auto-detection?
2. Should we support custom color schemes via environment variables?
3. Do we need Windows-specific ANSI enablement?
4. Should wrapped text break on word boundaries or allow mid-word breaks?
5. Should we add `--no-color` flag in addition to NO_COLOR env var?

## Non-Goals (For Now)

- Custom color schemes
- Rich formatting (underline, italics, backgrounds)
- Emoji support
- Markdown rendering in help text
- Dynamic width adjustment during display
- HTML export of help text
