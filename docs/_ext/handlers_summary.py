"""Sphinx extension that write a summary table of all handlers in SNAKE-fMRI."""

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList

from snkf.handlers import H


def setup(app):
    app.add_directive("handlers_table", HandlerTableDirective)
    return {"version": "0.1"}


class HandlerTableDirective(Directive):

    has_content = True

    def run(self) -> list[nodes.table]:
        """Return a table node that contains a row per handler.

        Columns are Handler name, link to Handler class, and  description of the handler
        (typically the first line of the docstring)

        """
        print("in custom run ")
        table = nodes.table()
        tgroup = nodes.tgroup(cols=3)
        table += tgroup
        tgroup += nodes.colspec(colwidth=1)
        tgroup += nodes.colspec(colwidth=1)
        tgroup += nodes.colspec(colwidth=3)

        thead = nodes.thead()
        tgroup += thead
        header_row = nodes.row()
        thead += header_row
        header_row += [
            nodes.entry("", nodes.paragraph(text="name")),
            nodes.entry("", nodes.paragraph(text="Reference")),
            nodes.entry("", nodes.paragraph(text="Description")),
        ]
        tbody = nodes.tbody()
        tgroup += tbody

        print(table)
        for hname, hclass in H.items():
            print(hname, hclass.__module__, hclass.__name__)
            row = nodes.row()
            tbody += row
            row += [
                nodes.entry("", nodes.paragraph(text=hname)),
                nodes.entry("", self.parse_handler_class(hclass)),
                nodes.entry(
                    "",
                    nodes.paragraph(
                        text=(
                            hclass.__doc__.strip().split("\n", 1)[0]
                            if hclass.__doc__
                            else ""
                        )
                    ),
                ),
            ]

        print(table)
        return [table]

    def parse_handler_class(self, hclass) -> list[nodes.Node]:
        """Parse the documentation of the handler class."""
        # Construct the reStructuredText content for the class reference
        content = ViewList()
        content.append(
            f":class:`{hclass.__module__}.{hclass.__name__}`", "__dummy.rst", 10
        )
        node = nodes.container()
        node.document = self.state.document

        self.state.nested_parse(content, 0, node)
        return node.children[0]
