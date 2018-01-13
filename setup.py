from os import path

from setuptools import setup

with open(path.join(path.abspath(path.dirname(__file__)), 'README'), encoding='utf-8') as f:
    long_description = f.read()

setup(name="params_proto",
      description="A command line argument parsing utility using python class-based namespace for better IDE static "
                  "auto-completion",
      long_description=long_description,
      version="2.0.0",
      url="https://github.com/episodeyang/params_proto",
      author="Ge Yang",
      author_email="yangge1987@gmail.com",
      license=None,
      keywords=["params_proto", "decorator", "argparse", "shell arguments", "argument parser"],
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3"
      ],
      packages=["params_proto"],
      install_requires=["waterbear", "argparse", "typing"]
      )
