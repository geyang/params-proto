from setuptools import setup

setup(name="params_proto",
      description="A command line argument parsing using python namespace",
      long_description="params_proto uses python namespace to make working with schema-based command line arguments "
                       "much easier.",
      version="0.0.1",
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
      install_requires=["munch", "argparse", "typing"]
      )
