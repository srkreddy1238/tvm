# isort: skip_file
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Adreno schedule rules.
"""

from .pool import Pool2D
from .convolution import Conv2D, Conv2DTensorization
from .matmul import DequantMatmulTensorization, MatmulTensorization
from .layout_transform import LayoutTransform
from .attention import attention_prefill_ragged_adreno, attention_prefill_paged_adreno

from .fallback import Fallback
