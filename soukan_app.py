import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import io

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
    """
    try:
        temp_df = df[[x, y, covar]].dropna()
        if len(temp_df) < 3: return np.nan, np.nan

        r_xy = temp_df[x].corr(temp_df[y])
        r_xz = temp_df[x].corr(temp_df[covar])
        r_yz = temp_df[y].corr(temp_df[covar])
        
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if denominator == 0: return np.nan, np.nan
        return numerator / denominator, r_xy
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
    return template_df.to_csv(index=False)

def interpret_correlation(coef):
    """相関係数の日本語解釈"""
    abs_coef = abs(coef)
    if abs_coef >= 0.7: return "かなり強い関係"
    elif abs_coef >= 0.4: return "まあまあの関係"
    elif abs_coef >= 0.2: return "弱い関係"
    else: return "ほとんど関係なし"

# --- 3. メイン処理 ---

def main():
    st.title("🔍 因果・相関分析マスター")
    st.markdown("数値を入れるだけで、「関係の強さ」や「予測」を自動で分析します。")
    
    # --- サイドバー ---
    with st.sidebar:
        st.header("📂 データ入力")
        uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        st.markdown("---")
        st.markdown("##### テスト用データ")
        csv_text = create_csv_template()
        st.download_button("📥 サンプルCSV", csv_text.encode('utf-8-sig'), "sample_data.csv", "text/csv")

    # データ読み込み
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except:
            try: df = pd.read_csv(uploaded_file, encoding='shift-jis')
            except: st.error("読込エラー: 文字コードを確認してください"); return
    else:
        df = pd.read_csv(io.StringIO(create_csv_template()))
        st.info("💡 現在はサンプルデータモードです。")

    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] < 2:
        st.warning("⚠️ 数値列が2つ以上必要です。")
        return

    # --- タブ ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 関係を見る (相関)", 
        "🕵️ 本当の原因を探る (偏相関)", 
        "🔮 未来を予測する (回帰)", 
        "📋 データ表"
    ])

    # === Tab 1: 相関 ===
    with tab1:
        st.subheader("全体の関係性をチェック")
        corr_matrix = df_numeric.corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", 
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("赤＝片方が増えると相手も増える、青＝片方が増えると相手は減る")

    # === Tab 2: 偏相関 ===
    with tab2:
        st.subheader("見せかけの関係を見抜く")
        c1, c2, c3 = st.columns(3)
        if len(df_numeric.columns) >= 3:
            with c1: tx = st.selectbox("要因 (X)", df_numeric.columns, 0)
            with c2: ty = st.selectbox("結果 (Y)", df_numeric.columns, 1)
            with c3: 
                cands = [c for c in df_numeric.columns if c not in [tx, ty]]
                tz = st.selectbox("第三の要因 (Z)", cands) if cands else None

            if tx and ty and tz:
                if tx == ty:
                    st.warning("XとYは別の変数にしてください")
                else:
                    p_corr, raw_corr = calculate_partial_correlation(df_numeric, tx, ty, tz)
                    if np.isnan(p_corr):
                        st.error("計算できませんでした")
                    else:
                        st.markdown("### 診断結果")
                        diff = abs(raw_corr - p_corr)
                        
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("見た目の相関", f"{raw_corr:.3f}")
                        with m2:
                            st.metric(f"「{tz}」の影響を除いた本当の相関", f"{p_corr:.3f}", 
                                      delta=f"{p_corr - raw_corr:.3f}", delta_color="inverse")
                        
                        # 親しみやすい診断メッセージ
                        st.success("📝 **AI診断**: ")
                        if diff > 0.3:
                            st.markdown(f"⚠️ **注意！** 元の関係は「{tz}」による**見せかけ**の可能性が高いです。直接的な関係はもっと弱いです。")
                        elif diff < 0.1:
                            st.markdown(f"✅ **本物かも？** 「{tz}」を考慮しても関係は変わりません。{tx}と{ty}は直接つながっている可能性があります。")
                        else:
                            st.markdown(f"🤔 **一部影響あり** 「{tz}」が関係の一部を説明しています。")
        else:
            st.warning("変数が3つ以上必要です")

    # === Tab 3: 回帰 (大幅改修) ===
    with tab3:
        st.subheader("🔮 データの傾向から予測する")
        
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1: x_col = st.selectbox("何を変えると (X)", df_numeric.columns, 0, key='reg_x')
        with c_sel2: y_col = st.selectbox("何が変わる？ (Y)", df_numeric.columns, 1, key='reg_y')

        if x_col == y_col:
            st.warning("XとYは別の変数を選んでください。")
        else:
            plot_df = df.dropna(subset=[x_col, y_col])
            if len(plot_df) > 0:
                # 統計計算
                X = sm.add_constant(plot_df[x_col])
                model = sm.OLS(plot_df[y_col], X).fit()
                
                slope = model.params.iloc[1] # 傾き
                intercept = model.params.iloc[0] # 切片
                r2 = model.rsquared # 決定係数
                p_val = model.pvalues.iloc[1] # P値

                # --- レイアウト分割: 左にグラフ、右に見方ガイド ---
                col_graph, col_guide = st.columns([2, 1])
                
                with col_graph:
                    # 散布図作成
                    fig = px.scatter(
                        plot_df, x=x_col, y=y_col, trendline="ols",
                        trendline_color_override="red", hover_data=df.columns
                    )
                    fig.update_layout(title=f"{x_col} と {y_col} の散布図")
                    st.plotly_chart(fig, use_container_width=True)

                with col_guide:
                    st.info("💡 **グラフの見方ガイド**")
                    st.markdown("""
                    - **青い点**: 一人ひとりのデータです。
                    - **赤い線**: 全体の「傾向」を表す線です。
                    - **線の傾き**: 急なほど、影響が大きいことを意味します。
                    - **点の散らばり**: 線に近いほど、精度の高い予測ができます。
                    """)

                st.markdown("---")

                # --- わかりやすい言葉でのレポート ---
                st.subheader("📝 AI分析レポート")
                
                # 1. 信頼性判定
                rep_col1, rep_col2, rep_col3 = st.columns(3)
                with rep_col1:
                    st.markdown("**① この関係は信頼できる？**")
                    if p_val < 0.05:
                        st.success(f"✅ **信頼できます**\n\n(偶然そうなった確率は{(p_val*100):.1f}%と非常に低いです)")
                    elif p_val < 0.1:
                        st.warning(f"🤔 **微妙です**\n\n(統計的な確証まであと少しです)")
                    else:
                        st.error(f"❌ **偶然かもしれません**\n\n(データ上のたまたまの偏りの可能性があります)")

                with rep_col2:
                    st.markdown("**② 関係の強さは？**")
                    strength = interpret_correlation(np.sqrt(r2) if slope > 0 else -np.sqrt(r2))
                    st.info(f"**{strength}** です\n\n(予測の精度: {r2*100:.1f}%)")

                with rep_col3:
                    st.markdown("**③ 具体的にどう変わる？**")
                    direction = "増え" if slope > 0 else "減り"
                    st.write(f"「{x_col}」が **1** 増えると...")
                    st.write(f"👉 「{y_col}」は約 **{slope:.2f}** {direction}ます。")

                # --- インタラクティブ・シミュレーター ---
                st.markdown("---")
                st.subheader("🎛️ 予測シミュレーター")
                st.write("「もし、Xが〇〇だったら、Yはどうなる？」を計算します。")
                
                sim_col1, sim_col2, sim_col3 = st.columns([1, 0.5, 1])
                with sim_col1:
                    user_x = st.number_input(
                        f"もし {x_col} が...", 
                        value=float(plot_df[x_col].mean()),
                        step=1.0
                    )
                with sim_col2:
                    st.markdown("<h2 style='text-align: center; margin-top: 20px;'>👉</h2>", unsafe_allow_html=True)
                with sim_col3:
                    predicted_y = slope * user_x + intercept
                    st.metric(f"予測される {y_col}", f"{predicted_y:.2f}")

    # === Tab 4: データ ===
    with tab4:
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()