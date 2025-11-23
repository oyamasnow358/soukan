import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
import statsmodels.api as sm

# --- 1. 初期設定 ---
st.set_page_config(
    page_title="因果・相関分析マスター",
    page_icon="🔍",
    layout="wide"
)

# --- 2. 計算ロジック関数 ---

def calculate_partial_correlation(df, x, y, covar):
    """
    偏相関係数を計算する関数
    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    """
    try:
        r_xy = df[x].corr(df[y])
        r_xz = df[x].corr(df[covar])
        r_yz = df[y].corr(df[covar])
        
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if denominator == 0:
            return np.nan
        
        p_corr = numerator / denominator
        return p_corr, r_xy
    except:
        return np.nan, np.nan

def create_csv_template():
    """テンプレートCSVの生成"""
    template_df = pd.DataFrame({
        '国語テスト': [80, 65, 92, 75, 58, 85, 70, 95, 60, 78],
        '読書量(冊)': [5, 2, 8, 4, 1, 6, 3, 10, 1, 5],
        '語彙力スコア': [60, 45, 70, 55, 40, 62, 50, 75, 38, 58],
        'スマホ時間(分)': [60, 120, 30, 90, 150, 50, 100, 20, 160, 80]
    })
    return template_df.to_csv(index=False, encoding='utf-8-sig')

# --- 3. UIコンポーネント ---

def show_explanation():
    with st.expander("📚 このアプリでできること（因果と相関の違い）"):
        st.markdown("""
        ### 1. 相関関係 (Correlation)
        「片方が増えると、もう片方も増える/減る」という関係。
        *   例：アイスクリームの売上と水難事故の数（両方とも夏に増えるだけで、直接の関係はないかも？）

        ### 2. 疑似因果の検証 (Partial Correlation) 🔥 **New**
        「第三の要因」の影響を取り除いても、関係が残るかを確認します。
        *   例：「アイス」と「水難事故」の関係から「気温」の影響を取り除くと、関係は消えるはずです。これがわかると、より**因果関係**に近い推論ができます。
        
        ### 3. 影響度の予測 (Regression)
        「Xを変化させたら、Yは具体的にどれくらい変わるか？」を数式にします。
        """)

# --- 4. メイン処理 ---

def main():
    st.title("🔍 因果・相関分析マスター Webアプリ")
    st.markdown("データの**相関関係**だけでなく、第三の要因を考慮した**因果の可能性**を探求するためのツールです。")
    
    show_explanation()
    
    # --- サイドバー: データアップロード ---
    with st.sidebar:
        st.header("📂 データ入力")
        
        uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        
        st.markdown("---")
        st.markdown("##### テスト用データ")
        csv_data = create_csv_template()
        st.download_button(
            label="📥 サンプルCSVをダウンロード",
            data=csv_data,
            file_name="sample_data.csv",
            mime="text/csv"
        )

    # データの読み込み
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except:
            try:
                df = pd.read_csv(uploaded_file, encoding='shift-jis')
            except:
                st.error("CSVの読み込みに失敗しました。文字コードを確認してください。")
                return
    else:
        # デモデータを使用
        df = pd.read_csv(pd.compat.StringIO(create_csv_template()), encoding='utf-8-sig')
        st.info("💡 現在はサンプルデータで動作しています。自身のデータを分析するには左側からCSVをアップロードしてください。")

    # 数値データの抽出
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.shape[1] < 2:
        st.warning("⚠️ 数値データが2列以上あるCSVを使用してください。")
        return

    # --- タブによる機能切り替え ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 相関マトリックス", 
        "🕵️ 因果・交絡分析 (重要)", 
        "📈 回帰・散布図詳細", 
        "📋 データ一覧"
    ])

    # ==========================================
    # Tab 1: 相関マトリックス (Plotly版)
    # ==========================================
    with tab1:
        st.subheader("変数の全体的な関係性を把握する")
        
        corr_matrix = df_numeric.corr()
        
        # Plotly Heatmap
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            labels=dict(color="相関係数")
        )
        fig_corr.update_layout(title="相関ヒートマップ（インタラクティブ）", height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        *   **赤色**: 正の相関（片方が増えるともう片方も増える）
        *   **青色**: 負の相関（片方が増えるともう片方は減る）
        """)

    # ==========================================
    # Tab 2: 因果・交絡分析 (偏相関) - 🌟 今回の目玉機能
    # ==========================================
    with tab2:
        st.subheader("🕵️ その関係は「見せかけ」ではありませんか？")
        st.markdown("ある2つの変数に関係があっても、それは**「第三の変数（交絡因子）」**の影響かもしれません。その影響を取り除いてみましょう。")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        
        with col_cfg1:
            target_x = st.selectbox("要因 (X)", df_numeric.columns, index=0)
        with col_cfg2:
            target_y = st.selectbox("結果 (Y)", df_numeric.columns, index=1)
        with col_cfg3:
            # XとY以外のカラムを候補にする
            confounder_candidates = [c for c in df_numeric.columns if c not in [target_x, target_y]]
            control_z = st.selectbox("第三の変数 (Z: 制御変数)", confounder_candidates)

        if target_x and target_y and control_z:
            p_corr, raw_corr = calculate_partial_correlation(df_numeric, target_x, target_y, control_z)
            
            st.markdown("### 分析結果")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("元の相関係数", f"{raw_corr:.3f}")
            with col_res2:
                st.metric(f"{control_z}の影響を除いた相関（偏相関）", f"{p_corr:.3f}", 
                          delta=f"{p_corr - raw_corr:.3f}", delta_color="inverse")
            with col_res3:
                change_ratio = abs((raw_corr - p_corr) / raw_corr * 100) if raw_corr != 0 else 0
                st.metric("関係性の変化率", f"{change_ratio:.1f}%")

            # 解釈の自動生成
            st.info(f"💡 **AI解釈アシスト**: \n\n"
                    f"「{target_x}」と「{target_y}」の関係から、「{control_z}」の影響を取り除くと、"
                    f"相関係数は **{raw_corr:.2f}** から **{p_corr:.2f}** に変化しました。")

            if abs(p_corr) < 0.2 and abs(raw_corr) > 0.4:
                st.error(f"⚠️ **注意**: 元の相関は「{control_z}」による**見せかけの相関（疑似相関）**である可能性が高いです。{target_x}が直接{target_y}に影響しているわけではないかもしれません。")
            elif abs(p_corr - raw_corr) < 0.1:
                st.success(f"✅ 「{control_z}」を考慮しても関係性はほとんど変わりません。{target_x}と{target_y}の直接的な結びつきは強い可能性があります。")
            else:
                st.warning(f"🤔 「{control_z}」が関係性の一部を説明しています。因果関係を考える際は{control_z}も考慮に入れる必要があります。")

    # ==========================================
    # Tab 3: 回帰・散布図詳細
    # ==========================================
    with tab3:
        st.subheader("📈 データの分布と予測")
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            x_axis = st.selectbox("横軸 (原因?)", df_numeric.columns, index=0, key='scatter_x')
        with col_sel2:
            y_axis = st.selectbox("縦軸 (結果?)", df_numeric.columns, index=1, key='scatter_y')

        # 散布図 with 回帰直線 (Plotly)
        fig_scatter = px.scatter(
            df, x=x_axis, y=y_axis, 
            trendline="ols", 
            trendline_color_override="red",
            hover_data=df.columns
        )
        fig_scatter.update_layout(title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 回帰分析の詳細統計 (Statsmodels)
        st.markdown("#### 📊 統計的な詳細（単回帰分析）")
        
        X = df_numeric[x_axis]
        Y = df_numeric[y_axis]
        X = sm.add_constant(X) # 定数項を追加
        
        model = sm.OLS(Y, X).fit()
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("決定係数 (R2)", f"{model.rsquared:.3f}", help="1に近いほど、横軸のデータで縦軸のデータをうまく説明できています。")
        with col_stat2:
            st.metric("P値 (有意確率)", f"{model.pvalues[1]:.4f}", help="0.05未満なら、統計的に偶然とは言えない関係があります。")
        with col_stat3:
            coef = model.params[1]
            st.metric("回帰係数 (傾き)", f"{coef:.3f}", help=f"{x_axis}が1増えると、{y_axis}が約{coef:.2f}変化すると予測されます。")

        with st.expander("詳細な統計レポートを見る"):
            st.text(model.summary())

    # ==========================================
    # Tab 4: データ一覧
    # ==========================================
    with tab4:
        st.subheader("📋 生データ確認")
        st.dataframe(df, use_container_width=True)
        st.caption(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")

if __name__ == "__main__":
    main()