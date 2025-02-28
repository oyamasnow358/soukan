import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
import matplotlib as mpl
import shutil

# フォント設定（サーバー環境用）
server_font_path = "/tmp/ipaexg.ttf"

# ローカルのフォントファイルをサーバーにコピー
local_font_path = "ipaexg.ttf"  # フォントファイルをアプリフォルダに同梱
if not os.path.exists(server_font_path):
    shutil.copy(local_font_path, server_font_path)

# フォントの設定
if os.path.exists(server_font_path):
    font_prop = fm.FontProperties(fname=server_font_path)
    mpl.rcParams["font.family"] = font_prop.get_name()
    plt.rc("font", family=font_prop.get_name())
    st.write(f"✅ フォント設定: {mpl.rcParams['font.family']}")
else:
    st.error("❌ フォントファイルが見つかりません。")

st.title("相関分析 Web アプリ")
# 初心者向け説明の表示切り替え
if "show_explanation" not in st.session_state:
           st.session_state.show_explanation = False
        # ボタンを押すたびにセッションステートを切り替える
if st.button("説明を表示/非表示"):
           st.session_state.show_explanation = not st.session_state.show_explanation

         # セッションステートに基づいて説明を表示
if st.session_state.show_explanation:
           st.markdown("""
           ### **相関分析とは？**
    
            - **p値の意味**    
                       
             相関分析（そうかんぶんせき）とは、2つのデータの関係がどれくらい強いかを調べる方法です。簡単に言うと、「Aが変わるとBも変わる？」という関係性を数値で表します。

             相関分析は、子どもの発達や支援方法の改善にも役立ちます。例えば、次のような場面で使えます。  
                       
          ① 言語理解と社会性の関係

            子どもの言語理解のレベルと友達との関わりの回数に関係があるか調べる。
           「言語理解のスコアが高い子ほど、友達とよく遊ぶ」という結果なら、言語支援が社会性の向上にも影響すると考えられる。

          ② 書字能力と視覚記憶の関係

            ひらがなを書く力（書字）と、目で見た情報を記憶する力（視覚記憶）の関係を分析する。
            相関が強ければ、視覚記憶のトレーニングをすることで書字能力が向上する可能性がある。

          ③ 落ち着きと学習の関係

            子どもが授業中に座っていられる時間と、学習の理解度（テストの点数）に関係があるか調べる。
           「落ち着いて座れる子の方が学習の理解が進んでいる」という結果なら、座る時間を伸ばすための支援が学習にも良い影響を与えると分かる。

           - ### **相関の強さを表す数値**

             相関分析では、「相関係数（そうかんけいすう）」 という数値（-1～1の間）を使って関係の強さを表します。
             相関係数	関係の強さ	例
             ＋1.0 に近い	強い正の相関（Aが増えるとBも増える）	言語理解が高い子ほど友達と多く遊ぶ
             0 に近い	相関なし（関係がない）	好きな色と学習成績
             −1.0 に近い	強い負の相関（Aが増えるとBは減る）	スマホの使用時間が増えると睡眠時間が減る
        
           - ### **特別支援教育での活用ポイント**

            子どもの得意・不得意を知る
             → 例えば、「視覚記憶が苦手な子は、聞いて学ぶ方が得意かも？」と支援の方向を決める。
            どんな支援が効果的か考える
             → 「発語の少ない子には、ジェスチャーを使った支援が有効か？」などをデータで確認。
            支援の効果を測る
             → 例えば、「音読の練習を増やしたら、読字スコアが上がるか？」を分析。""")

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
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig', encoding_errors='replace')
        st.write("### アップロードされたデータ")
        st.dataframe(df.head())

        # 相関係数の計算
        correlation_matrix = df.corr()
        
        st.write("### 変数間の相関係数")
        st.dataframe(correlation_matrix)

        # ヒートマップの描画
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)

        # 軸ラベルの日本語設定
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontproperties=font_prop)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop)

        ax.set_title("相関行列", fontproperties=font_prop)
        st.pyplot(fig)
        plt.close(fig)

        # 相関の説明
        st.write("### 相関の解釈")
        explanation = ""  # 説明文用の変数
        processed_pairs = set()  # 処理済みの組み合わせを記録

        for col1 in correlation_matrix.columns:
           for col2 in correlation_matrix.columns:
             if col1 != col2 and (col2, col1) not in processed_pairs:
                corr_value = correlation_matrix.loc[col1, col2]
                processed_pairs.add((col1, col2))  # 処理済みとして記録

                if abs(corr_value) >= 0.7:
                   explanation += f"🔴 **{col1}** と **{col2}** は 強い相関 があります！（相関係数: {corr_value:.2f}）\n\n"
                elif abs(corr_value) >= 0.4:
                   explanation += f"🟠 **{col1}** と **{col2}** は 中程度の相関 があります。（相関係数: {corr_value:.2f}）\n\n"
                elif abs(corr_value) >= 0.2:
                   explanation += f"🟡 **{col1}** と **{col2}** は 弱い相関 があります。（相関係数: {corr_value:.2f}）\n\n"
                else:
                   explanation += f"⚪ **{col1}** と **{col2}** は ほぼ関係がありません。（相関係数: {corr_value:.2f}）\n\n"

        st.markdown(explanation)


    except Exception as e:
        st.error(f"❌ CSVの読み込み時にエラーが発生しました: {e}")
