import streamlit as st
import pandas as pd
def get_sentence(sentemce):
    # code here

    # set a threshold if > then return the sentences list else return an apology sentence
    dd = [["helloooooooooo sajdashdi ashdaosiu  ashdiuashgd asiud abdas hgdasiud asgiu",0.6],[2,0.8]]
    return dd
#while True:
question = st.text_input("Enter some text 👇")
if question:
    ans = get_sentence(question)
    # 将二维数组转换为DataFrame
    # st.checkbox("Use container width", value=False, key="use_container_width")
    df = pd.DataFrame(ans, columns=['Answer', 'possibility'])
    st.dataframe(df,width=800)  # Same as st.write(df)



