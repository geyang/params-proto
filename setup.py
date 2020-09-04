from os import path

from setuptools import setup, find_packages

with open(path.join(path.abspath(path.dirname(__file__)), 'README'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
with open(path.join(path.abspath(path.dirname(__file__)), 'VERSION'), encoding='utf-8') as f:
    VERSION = f.read()

setup(name="params_proto",
      description="A command line argument parsing utility using python class-based namespace for better IDE static "
                  "auto-completion",
      long_description=LONG_DESCRIPTION,
      version=VERSION,
      url="https://github.com/episodeyang/params_proto",
      author="Ge Yang",
      author_email="yangge1987@gmail.com",
      license=None,
      keywords=["params_proto", "decorator", "argparse",
                "argcomplete", "auto-completion", "autocomplete",
                "shell arguments", "argument parser"],
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3"
      ],
      packages=[p for p in find_packages() if "test" not in p],
      install_requires=["waterbear>=2.6.5", "argparse", "argcomplete", "typing"]
      )
