#!/bin/bash

dir=$(cd "$(dirname "$0")"; pwd)
gdwct_dir="${dir}/../gdwct"

echo -e 'import os\nimport sys\nsys.path.append(os.path.join(os.getcwd(), "gdwct"))' > ${gdwct_dir}/__init__.py
patch -d${gdwct_dir} < ${dir}/gdwct.patch
