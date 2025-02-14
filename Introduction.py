import streamlit as st

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
from utils import get_sentence_index

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

st.header("HR gie gie is **_really_ cool**.:sparkling_heart:")
st.markdown("This text is :red[colored red], and this is **:blue[colored]** and bold.")
st.markdown(":green[the color] is the :mortar_board: of hr giegie")

get_sentence_index(
    sentence="just for test, replace this the develop.",
    inner_model=inner_model,
    final_index=final_index,
)
