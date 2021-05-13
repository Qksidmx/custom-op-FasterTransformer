# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Use time_two ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

faster_transformer_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_faster_transformer_ops.so'))
print(dir(faster_transformer_ops))
bert_transformer = faster_transformer_ops.bert_transformer
sentence_bert_transformer = faster_transformer_ops.sentence_bert_transformer
open_nmt_transformer = faster_transformer_ops.open_nmt_transformer
attention = faster_transformer_ops.attention
