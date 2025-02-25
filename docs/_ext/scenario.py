"""Custom format for the scenario config in yaml."""

from __future__ import annotations
import warnings
import shutil
from pathlib import Path
from textwrap import dedent
import yaml
from myst_sphinx_gallery import generate_gallery, GalleryConfig
import nbformat

# from sphinx.application import Sphinx


def _parse_body(body: str) -> dict:
    return yaml.safe_load(body)


def _parse_header(header_string: str) -> dict:
    """Parse the header of the file."""
    header_lines = header_string.splitlines()
    if header_lines[0] != "---":
        raise ValueError("Header must start with '---'")
    if header_lines[-1] != "---":
        raise ValueError("Header must end with '---'")
    header_lines = "\n".join(header_lines[1:-1])
    parsed_header = yaml.safe_load(header_lines)
    header = {"title": None, "authors": None, "description": None}
    header = header | parsed_header
    return header


def _split_header_body(filecontent: list[str]) -> tuple[str, str]:
    """Split a file content into header and body."""
    header = []
    body = []
    in_header = True
    for line in filecontent:
        if line.startswith("#") and in_header:
            header.append(line.strip("# "))
        else:
            in_header = False
            body.append(line)
    return "\n".join(header), "".join(body)


def scenario2nb(source_file, output_file):
    """Convert a scenario file to a Notebook-like file."""
    # split header (comments ) from body (last line of header = last comment line)
    with open(source_file, "r") as f:
        filecontent = f.readlines()
    header, body = _split_header_body(filecontent)

    header_dict = _parse_header(header)
    N = nbformat.v4.new_notebook()
    # Now lets add cells
    N.cells.append(
        nbformat.v4.new_markdown_cell(
            source=dedent(
                f"""\
            # {header_dict.get("title", "Unnamed Scenario")}
            ```{{note}}
            *Authors*: {header_dict.get("authors", "Anonymous")}
            *Description*: {header_dict.get("description", "No description")}
            ```
            """
            )
        )
    )

    N.cells.append(nbformat.v4.new_markdown_cell(source=f"```yaml\n{body}\n```"))
    # TODO Generate a full script from the notebook
    # TODO Add internal referencing from the API

    nbformat.write(N, output_file)


def get_gallery_header(directory, output_directory):
    gallery_header_file = Path(directory) / "GALLERY_HEADER.rst"
    output_dir_file = Path(output_directory) / "GALLERY_HEADER.rst"
    if gallery_header_file.exists():
        shutil.copy2(gallery_header_file, output_dir_file)
    else:
        # Use a Default file.
        shutil.copy2(Path(__file__).parent / "GALLERY_HEADER.rst", output_dir_file)


def main_ext(app):
    root_dir = Path(__file__).parent.parent
    # For the scenarios, we first are going to create the notebooks from a set of yaml files
    # TODO : Use a tmp dir
    tmp_dir = root_dir / Path("auto_scenarios_tmp")
    # Delete the previous auto_scenarios_tmp
    if tmp_dir.exists():
        for file in Path("auto_scenarios_tmp").rglob("*"):
            file.unlink()
    else:
        tmp_dir.mkdir()

        # Create the new notebooks
        for yaml_file in (root_dir / "../src/cli-conf").rglob("scenario*.yaml"):
            nb_file = root_dir / "auto_scenarios_tmp" / (yaml_file.stem + ".ipynb")
            yaml_file = yaml_file.resolve()
            try:
                scenario2nb(yaml_file, nb_file)
            except Exception as e:
                warnings.warn(f"Error in {yaml_file}: {e}")
    get_gallery_header(root_dir / "../src/cli-conf", root_dir / "auto_scenarios_tmp")
    # Generate the gallery
    generate_gallery(
        GalleryConfig(
            examples_dirs="auto_scenarios_tmp",
            gallery_dirs="auto_scenarios",
            root_dir=root_dir,
            notebook_thumbnail_strategy="markdown",
            target_prefix="scenario",
            base_gallery=True,
        )
    )


def setup(app):
    app.connect("builder-inited", main_ext)
