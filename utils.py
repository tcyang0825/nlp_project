import os
import hnswlib

import paddle
from training.ann_util import build_index
from training.data import (
    convert_example_test,
    create_dataloader,
    gen_id2corpus,
    gen_text_file,
)
from training.model import SimCSE

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.utils.log import logger

params_path = "model/model_state.pdparams"

paddle.set_device("gpu")
rank = paddle.distributed.get_rank()
if paddle.distributed.get_world_size() > 1:
    paddle.distributed.init_parallel_env()
tokenizer = AutoTokenizer.from_pretrained("rocketqa-zh-base-query-encoder")

pretrained_model = AutoModel.from_pretrained("rocketqa-zh-base-query-encoder")

model = SimCSE(pretrained_model, output_emb_size=256)
model = paddle.DataParallel(model)

# Load pretrained semantic model
if params_path and os.path.isfile(params_path):
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    logger.info("Loaded parameters from %s" % params_path)
else:
    raise ValueError("Please set --params_path with correct pretrained model file")

inner_model = model._layers

final_index = hnswlib.Index(space="ip", dim=256)
final_index.load_index("model/my_index.bin")


def get_sentence_index(sentence, inner_model, final_index):
    tokenizer = AutoTokenizer.from_pretrained("rocketqa-zh-base-query-encoder")
    encoded_inputs = tokenizer(text=[sentence], max_seq_len=128)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    input_ids = paddle.to_tensor(input_ids, dtype="int64")
    token_type_ids = paddle.to_tensor(token_type_ids, dtype="int64")
    cls_embedding = inner_model.get_pooled_embedding(
        input_ids=input_ids, token_type_ids=token_type_ids
    )
    # print('提取特征:{}'.format(cls_embedding))
    recalled_idx, cosine_sims = final_index.knn_query(cls_embedding.numpy(), 10)
    return recalled_idx


# results = get_sentence_index("新加坡怎么找房子？", inner_model=inner_model, final_index=final_index)
# print(results)
