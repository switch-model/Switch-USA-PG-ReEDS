#!/usr/bin/env python3
"""
run sql queries on CSV files (or other sources) using duckdb.

This can be installed for greater convenience via `ln -s /path/to/q.py
/usr/local/bin/q`.

Example command:

    q "select * from 'path/to/file.csv'"

Simplest form is "select * from '/path/to/file.csv'", but it is also possible to
do "select * from read_csv_auto('/path/to/file.csv')", or use other forms for
Excel files, .parquet files, sqlite3, etc. See
https://duckdb.org/docs/data/csv/overview.html for options

This also provides a convenience function, round1000(), which divides columns by
1000 and rounds to the nearest int (useful for comparing large numbers).

If you redirect output with | or > this script will generate csv text for the
full query result instead of an abbreviated summary.

It can also be helpful to run these queries to learn about tables:
q "describe select * from 'path/to/file.csv'"
q "select column_name from (describe select * from 'path/to/file.csv')"

"""

import io
import sys
from contextlib import redirect_stdout

import duckdb


def round1000(x: float) -> int:
    return int(round(x / 1000))


duckdb.create_function("round1000", round1000)

if len(sys.argv) == 2:
    query = sys.argv[1]  # monolithic query
else:
    # parse tokens, add quotes to filenames (for convenience and because the
    # shell would have stripped them)
    tokens = []
    for t in sys.argv[1:]:
        # split final comma or semicolon into separate token
        if t.endswith(",") or t.endswith(";"):
            tokens.append(t[:-1])
            tokens.append(t[-1:])
        else:
            tokens.append(t)
    for i, t in enumerate(tokens):
        if i > 0 and tokens[i - 1].lower() in {"from", "join", ","} and not "'" in t:
            # shell parser stripped any quotes from the filename (and it wasn't
            # part of a quoted function call), so add them back
            tokens[i] = f"'{t}'"
    query = " ".join(tokens)
    # This still won't handle quoted strings in the query correctly. Also needs
    # quotes around * in query.


con = duckdb.connect()

try:
    if sys.stdout.isatty():
        # show the output from .show() but suppress the line about variable types
        buf = io.StringIO()
        with redirect_stdout(buf):
            duckdb.sql(query).show(max_rows=50)
        lines = buf.getvalue().splitlines()
        # remove the third line (data type), if present
        if len(lines) > 2:
            lines.pop(2)
        print("\n".join(lines))
    else:
        # Results are being piped or redirected; generate csv for full result set.
        # This seems to be the fastest way to generate a .csv file. The alternative
        # would be to read it in as a dataframe, then save that to sys.stdout somehow.
        duckdb.sql(
            f"COPY ({query.strip(';')}) TO '/dev/stdout' (HEADER, DELIMITER ',');"
        )
except Exception as e:
    raise SystemExit(str(e))
    # print(e, file=sys.stderr)
    # sys.exit(1)
