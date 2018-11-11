import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyrobdean",
    version="0.0.1",
    author="Example Author",  # todo
    author_email="author@example.com",  # todo
    description="A small example package",  # todo
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/example-project",  # todo
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",  # todo
        "License :: OSI Approved :: MIT License",  # todo
        "Operating System :: OS Independent",
    ),
)