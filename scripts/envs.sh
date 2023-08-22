#!/bin/bash
ROOT=$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)
PYTHONPATH="";
export PYTHONPATH="${ROOT}":${PYTHONPATH}
echo "PYTHONPATH includes "${PYTHONPATH}