"""
patch pd.DataFrame.to_csv() to reduce binary-csv artifacts,
e.g., 0.0387 -> 0.038700000000000005

This will produce results identical to pg_to_switch.py.
"""

import pandas as pd


def clean_to_csv(self, *args, float_format="%.15g", **kwargs):
    pd_to_csv(self, *args, float_format=float_format, **kwargs)


pd_to_csv = pd.DataFrame.to_csv
pd.DataFrame.to_csv = clean_to_csv
