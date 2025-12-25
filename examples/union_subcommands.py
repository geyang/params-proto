"""
Example: Union Types as Subcommands

Union types create subcommand-like behavior where each class becomes
a CLI command with its own parameters.

Usage:
  python union_subcommands.py perspective --fov 45 --aspect 1.77
  python union_subcommands.py orthographic --scale 2.0
  python union_subcommands.py perspective --help
"""

from dataclasses import dataclass
from typing import Union

from params_proto import proto


@dataclass
class PerspectiveCamera:
  """Perspective camera with field of view."""

  fov: float = 60.0  # Field of view in degrees
  aspect: float = 1.33  # Aspect ratio (width/height)
  near: float = 0.1  # Near clipping plane
  far: float = 100.0  # Far clipping plane


@dataclass
class OrthographicCamera:
  """Orthographic camera with uniform scale."""

  scale: float = 1.0  # Orthographic scale
  near: float = 0.1  # Near clipping plane
  far: float = 100.0  # Far clipping plane


@proto.cli
def view(
  camera: Union[PerspectiveCamera, OrthographicCamera],  # Required subcommand
  output: str = "render.png",  # Output file
  verbose: bool = False,  # Verbose logging
):
  """
  Render a 3D scene with the specified camera.

  Args:
      camera: Camera configuration (perspective or orthographic)
      output: Output file path
      verbose: Enable verbose logging
  """
  if verbose:
    print(f"Rendering to: {output}")

  if isinstance(camera, PerspectiveCamera):
    print("Perspective Camera:")
    print(f"  FOV: {camera.fov}°")
    print(f"  Aspect Ratio: {camera.aspect}")
    print(f"  Clipping: {camera.near} to {camera.far}")
  elif isinstance(camera, OrthographicCamera):
    print("Orthographic Camera:")
    print(f"  Scale: {camera.scale}")
    print(f"  Clipping: {camera.near} to {camera.far}")


if __name__ == "__main__":
  view()
