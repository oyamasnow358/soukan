import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
import matplotlib as mpl
import shutil


# サーバー環境用のフォントパス
server_font_path = "/tmp/ipaexg.ttf"

# ローカルのフォントパスをサーバーにコピー
local_font_path = "ipaexg.ttf"  # フォントファイルをアプリフォルダに同梱
if not os.path.exists(server_font_path):
    shutil.copy(local_font_path, server_font_path)


st.title("相関分析 Web アプリ")
df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
# CSVのひな型を作成してダウンロード
st.sidebar.header("CSVひな型のダウンロード")

def create_sample_csv():
    sample_data = {
        "変数1": np.random.randint(1, 100, 10),
        "変数2": np.random.randint(1, 100, 10),
        "変数3": np.random.randint(1, 100, 10)
    }
    sample_df = pd.DataFrame(sample_data)
    return sample_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

sample_csv = create_sample_csv()
st.sidebar.download_button(
    label="CSVひな型をダウンロード",
    data=sample_csv,
    file_name="sample_correlation.csv",
    mime="text/csv")

# CSVファイルのアップロード
st.sidebar.header("データのアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='utf-8-sig', errors='replace')
    st.write("### アップロードされたデータ")
    st.dataframe(df.head())

    # 相関係数の計算
    correlation_matrix = df.corr()
    
    st.write("### 特徴量間の相関係数")
    st.dataframe(correlation_matrix)

    # ヒートマップの描画
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    if font_name:
        ax.set_title("相関行列", fontproperties=fm.FontProperties(fname=font_path))
    else:
        ax.set_title("相関行列")
    st.pyplot(fig)
    plt.close(fig)

    # 相関の説明
    st.write("### 相関の解釈")
    explanation = ""  # 説明文用の変数
    
    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2:
                corr_value = correlation_matrix.loc[col1, col2]
                if abs(corr_value) >= 0.7:
                    explanation += f"🔴 **{col1}** と **{col2}** は 強い相関 があります！（相関係数: {corr_value:.2f}）\n"
                elif abs(corr_value) >= 0.4:
                    explanation += f"🟠 **{col1}** と **{col2}** は 中程度の相関 があります。（相関係数: {corr_value:.2f}）\n"
                elif abs(corr_value) >= 0.2:
                    explanation += f"🟡 **{col1}** と **{col2}** は 弱い相関 があります。（相関係数: {corr_value:.2f}）\n"
                else:
                    explanation += f"⚪ **{col1}** と **{col2}** は ほぼ関係がありません。（相関係数: {corr_value:.2f}）\n"
    
    st.markdown(explanation)