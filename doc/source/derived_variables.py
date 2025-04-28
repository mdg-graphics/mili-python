"""Generates derived_variables.rst."""
from mili import reader

file_path: str = "../tests/data/serial/sstate/d3samp6.plt"

output_text = """
.. code-block:: python

    print( db.supported_derived_variables() )

    {}
"""

with open("derived_variables.rst", mode="w", errors="ignore") as fout:
    db = reader.open_database(file_path)
    derived_variables = db.supported_derived_variables()

    derived_variables_text = ""
    for i, var in enumerate(derived_variables):
        derived_variables_text += var
        derived_variables_text += ", "
        if i % 5 == 4:
            derived_variables_text += "\n    "

    fout.write(output_text.format(derived_variables_text))