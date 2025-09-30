# Documentation Development

## Build the Documentation

The following sections describe how to set up and build the Megatron Bridge documentation.

Switch to the documentation source folder and generate HTML output.

```sh
cd docs/
uv run --only-group docs sphinx-build . _build/html
```

* The resulting HTML files are generated in a `_build/html` folder that is created under the project `docs/` folder.
* The generated python API docs are placed in `apidocs` under the `docs/` folder.

## Live Building

When writing documentation, it can be helpful to serve the documentation and have it update live while you edit.

To do so, run:

```sh
cd docs/
uv run --only-group docs sphinx-autobuild . _build/html --port 12345 --host 0.0.0.0
```

Open a web browser and go to `http://${HOST_WHERE_SPHINX_COMMAND_RUN}:12345` to view the output.


## Write Tests in Python Docstrings

Any code in triple backtick blocks with the `{doctest}` directive will be tested. The format follows Python's doctest module syntax, where `>>>` indicates Python input and the following line shows the expected output. Here's an example:

```python
def add(x: int, y: int) -> int:
    """
    Adds two integers together.

    Args:
        x (int): The first integer to add.
        y (int): The second integer to add.

    Returns:
        int: The sum of x and y.

    Examples:
    ```{doctest}
    >>> from megatron.bridge.made_up_package import add
    >>> add(1, 2)
    3
    ```

    """
    return x + y
```

## Run Tests in Python Docstrings

You can run tests in our Python docstrings with:

```sh
cd docs/
uv run --only-group docs sphinx-build -b doctest . _build/doctest
```

## Documentation Version

The three files below control the version switcher. Before you attempt to publish a new version of the documentation, update these files to match the latest version numbers.

* docs/versions1.json
* docs/project.json
* docs/conf.py

