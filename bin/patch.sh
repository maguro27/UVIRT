#!/bin/bash

dir=$(cd "$(dirname "$0")"; pwd)
gdwct_dir="${dir}/../gdwct"

patch -d${gdwct_dir} < ${dir}/gdwct.patch
