import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from scipy import stats # p値の計算に必要

# --- 1. 初期設定 ---

# Streamlitページの基本設定
st.set_page_config(
    page_title="相関分析 Webアプリ",
    page_icon="🔗",
    layout="wide"
)

# --- 2. フォント設定 ---

def setup_japanese_font():
    """
    Matplotlib/Seabornで日本語を表示するためのフォントを設定します。
    IPAexゴシックフォントファイル（ipaexg.ttf）が同じディレクトリにあることを想定しています。
    """
    font_path = "ipaexg.ttf"
    
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_theme(style='whitegrid', font=font_prop.get_name())
    else:
        st.sidebar.warning("⚠️ 日本語フォントファイル（ipaexg.ttf）が見つかりません。グラフが文字化けします。")

# --- 3. UIコンポーネント関数 ---

def show_app_explanation():
    """初心者向けの相関分析の説明を表示する"""
    with st.expander("🔍 相関分析とは？（クリックで表示）", expanded=False):
        st.markdown("""
        ### **相関分析って、なに？**
        2つのデータの「関係の強さ」を調べる分析手法です。「片方が増えるともう片方も増える（または減る）」といった関係性を**相関係数**という-1から1の間の数値で表します。
        
        #### 相関係数の見方
        - **+1.0 に近い (正の相関)**: 片方が増えると、もう片方も増える傾向が強い。（例: `勉強時間`と`テストの点数`）
        - **-1.0 に近い (負の相関)**: 片方が増えると、もう片方は減る傾向が強い。（例: `スマホの使用時間`と`睡眠時間`）
        - **0 に近い (無相関)**: 2つのデータにほとんど関係がない。

        #### p値 (有意確率) の見方
        このアプリのヒートマップでは、相関係数にアスタリスク(`*`)が付くことがあります。
        - `*` **p < 0.05**: この相関は「統計的に意味がある（有意である）」可能性が高いです。偶然そうなったとは考えにくい、ということです。
        - `**` **p < 0.01**: この相関は「統計的に強く意味がある」可能性が非常に高いです。

        #### 特別支援教育での活用例
        - **得意・不得意の発見**: `視覚記憶`と`書字能力`の相関を調べ、支援の方向性を探る。
        - **支援効果の検証**: `音読練習の時間`と`読字スコア`の分析をし、練習の効果を測る。
        """)

def create_csv_template():
    """分析用のCSVテンプレートを作成し、ダウンロードボタンを設置する"""
    st.markdown("#### 1. データを用意する")
    template_df = pd.DataFrame({
        '国語の点数': [80, 65, 92, 75, 58],
        '算数の点数': [75, 70, 88, 78, 62],
        '勉強時間(分)': [120, 90, 150, 100, 70],
        '睡眠時間(時間)': [7.5, 8.0, 7.0, 7.2, 8.5]
    })
    
    csv_string = "# これは相関分析用のデータテンプレートです。\n# 自身のデータに書き換えてお使いください。\n" + template_df.to_csv(index=False, encoding='utf-8-sig')
    
    st.download_button(
        label="📥 CSVテンプレートをダウンロード",
        data=csv_string.encode('utf-8-sig'),
        file_name="correlation_template.csv",
        mime="text/csv",
        help="分析に使用するデータ形式のサンプルです。"
    )

# --- 4. 分析ロジック関数 ---

### 修正箇所 ###
# pearsonrの戻り値を確実にスカラに変換する処理を追加
def run_correlation_analysis(df):
    """相関分析を実行し、相関行列とp値行列を計算する"""
    df_corr = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    df_p_values = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    
    for col1 in df.columns:
        for col2 in df.columns:
            valid_data = df[[col1, col2]].dropna()
            
            if len(valid_data) < 3:
                corr, p_value = np.nan, np.nan
            else:
                try:
                    # pearsonrの計算
                    corr, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
                    
                    # 戻り値が配列の場合に備え、スカラ値に変換する
                    corr = float(corr)
                    p_value = float(p_value)

                except (ValueError, TypeError): # 計算不能なデータの場合
                    corr, p_value = np.nan, np.nan
            
            df_corr.loc[col1, col2] = corr
            df_p_values.loc[col1, col2] = p_value
            
    return df_corr, df_p_values

# --- 5. 結果表示関数 ---

def display_analysis_results(df_selected, corr_matrix, p_value_matrix):
    """分析結果をタブ形式で表示する"""
    st.header("🔗 相関分析の結果", divider="rainbow")

    tab1, tab2, tab3, tab4 = st.tabs(["相関ヒートマップ", "散布図マトリックス", "相関の要約", "使用データ"])

    # Tab1: 相関ヒートマップ
    with tab1:
        st.subheader("相関ヒートマップ (p値付き)")
        
        # applymapを推奨されるmapに変更
        annot = corr_matrix.map('{:.2f}'.format).astype(str)
        annot[(p_value_matrix < 0.05) & (p_value_matrix >= 0.01)] += '*'
        annot[p_value_matrix < 0.01] += '**'
        
        fig, ax = plt.subplots(figsize=(max(8, len(df_selected.columns)), max(6, len(df_selected.columns))))
        sns.heatmap(corr_matrix, annot=annot, fmt='s', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title("相関ヒートマップ")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        st.markdown("""
        - **色の意味**: 赤色が濃いほど「強い正の相関」、青色が濃いほど「強い負の相関」を示します。
        - **記号の意味**: `*` はp値が0.05未満、`**` はp値が0.01未満であることを示し、統計的に意味のある相関である可能性が高いことを表します。
        """)

    # Tab2: 散布図マトリックス
    with tab2:
        st.subheader("散布図マトリックス")
        if len(df_selected.columns) > 10:
            st.warning("⚠️ 変数の数が10を超えているため、表示が遅くなる可能性があります。")
        
        with st.spinner("グラフを描画中..."):
            # dropna()を追加して欠損値がある場合のKDEエラーを回避
            fig = sns.pairplot(df_selected.dropna(), diag_kind='kde')
            st.pyplot(fig)
        st.info("各変数ペアの関係性を散布図で可視化したものです。右肩上がりの傾向なら正の相関、右肩下がりなら負の相関があると考えられます。")

    # Tab3: 相関の要約
    with tab3:
        st.subheader("相関の強い組み合わせ")
        
        summary = corr_matrix.stack().reset_index()
        summary.columns = ['変数1', '変数2', '相関係数']
        summary = summary[summary['変数1'] != summary['変数2']].copy()
        summary['pair_key'] = summary.apply(lambda row: tuple(sorted((row['変数1'], row['変数2']))), axis=1)
        summary = summary.drop_duplicates(subset='pair_key')
        summary['abs_corr'] = summary['相関係数'].abs()
        summary = summary.sort_values(by='abs_corr', ascending=False).drop(columns=['pair_key', 'abs_corr'])
        
        st.dataframe(summary.style.format({'相関係数': '{:.3f}'})
                                .background_gradient(cmap='coolwarm', subset=['相関係数'], vmin=-1, vmax=1),
                     use_container_width=True,
                     hide_index=True)

    # Tab4: 使用データ
    with tab4:
        st.subheader("分析に使用したデータ")
        st.dataframe(df_selected)

# --- 6. メイン実行部 ---

def main():
    # 1. 初期設定
    setup_japanese_font()

    # 2. アプリのタイトルと説明
    st.title("🔗 相関分析 Webアプリ")
    st.write("アップロードしたデータの変数間の相関を、ヒートマップや散布図で分かりやすく可視化します。")
    show_app_explanation()
    st.markdown("---")

    # 3. サイドバーの設定
    with st.sidebar:
        st.header("⚙️ 設定パネル")
        create_csv_template()

        st.markdown("#### 2. ファイルをアップロード")
        uploaded_file = st.file_uploader(
            "CSVファイルをここにドラッグ＆ドロップ",
            type=["csv"],
            help="ヘッダー行を含む数値データで構成されたCSVファイルを選択してください。"
        )
    
    # ファイルがアップロードされた後の処理
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, comment='#', encoding='utf-8-sig')
            df_numeric = df.select_dtypes(include=np.number)
            
            if df_numeric.empty or len(df_numeric.columns) < 2:
                st.error("❌ 2列以上の数値データが見つかりませんでした。分析には数値データが必要です。")
                if 'analysis_results' in st.session_state:
                    del st.session_state['analysis_results']
                return

            st.success(f"✅ ファイル「{uploaded_file.name}」を読み込みました。数値型の {len(df_numeric.columns)} 変数が検出されました。")

            with st.sidebar:
                st.markdown("#### 3. 分析対象の変数を選択")
                default_vars = df_numeric.columns.tolist()
                selected_vars = st.multiselect(
                    "変数を選択（2つ以上）",
                    options=df_numeric.columns.tolist(),
                    default=default_vars
                )
                
                run_button = st.button("分析を実行", type="primary", use_container_width=True)

            if run_button:
                if len(selected_vars) < 2:
                    st.warning("⚠️ 分析するには、変数を2つ以上選択してください。")
                else:
                    df_selected = df_numeric[selected_vars]
                    with st.spinner("相関を計算中..."):
                        corr_matrix, p_value_matrix = run_correlation_analysis(df_selected)
                        st.session_state['analysis_results'] = {
                            "df_selected": df_selected,
                            "corr_matrix": corr_matrix,
                            "p_value_matrix": p_value_matrix
                        }
        
        except Exception as e:
            st.error(f"❌ ファイルの処理中にエラーが発生しました: {e}")
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']

    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        display_analysis_results(results['df_selected'], results['corr_matrix'], results['p_value_matrix'])

if __name__ == "__main__":
    main()