import streamlit as st
import pandas as pd
# from utils import get_sentence_index
from Introduction import *
import pandas
# dic = {}
# data_frame = pd.read_csv('../data/qa_pair.csv')
question = st.text_input("Enter some text 👇")
if question:
    ans = get_sentence(question, inner_model, final_index)
    # 将二维数组转换为DataFrame
    # st.checkbox("Use container width", value=False, key="use_container_width")
    df = pd.DataFrame(ans, columns=['Question in Database','Answer', 'possibility'])
    st.dataframe(df, width=800)  # Same as st.write(df)



