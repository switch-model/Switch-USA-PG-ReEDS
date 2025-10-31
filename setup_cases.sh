#!/bin/bash

# setup all myopic cases (don't specify case-id)
python pg_to_switch.py pg/settings switch/in/ --myopic
