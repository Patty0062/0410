import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

sns.set_theme(style="whitegrid", palette="muted")

st.set_page_config(page_title="互動式數據儀表板", layout="wide")

# --- 頂部設定面板區 ---
st.title("數據分析與機器學習儀表板")

# 使用 st.expander 預設開啟 (expanded=True)，或者直接放在 container
with st.container(border=True):
    st.subheader("操作控制台")
    
    # 第一排：檔案上傳
    uploaded_file = st.file_uploader("請先上傳你的 CSV 檔", type=["csv"])
    
    st.divider()
    
    # 第二排：功能選擇 (並排顯示)
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
    
    with col1:
        st.write("**1. 分析開關**")
        show_summary = st.checkbox("數據摘要", value=True)
        show_corr = st.checkbox("相關性熱圖")
        show_ml = st.checkbox("執行 ML 訓練")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # 二元化邏輯
        if 'count' in df.columns:
            df['target'] = (df['count'] > df['count'].median()).astype(int)
        
        # 讓使用者選擇分析哪個欄位 (針對分布圖)
        with col2:
            st.write("**2. 分布分析欄位**")
            target_col = st.selectbox("選擇你要觀察的欄位", df.select_dtypes(include=['number']).columns)

        # 機器學習參數
        with col3:
            st.write("**3. ML 參數設定**")
            test_size = st.select_slider("測試集比例", options=[0.2, 0.3, 0.4], value=0.3)
            
        with col4:
            st.write("**4. 演算法微調**")
            kernel_type = st.radio("SVM 核函數", ["rbf", "linear"], horizontal=True)

# --- 主數據顯示區 ---
if uploaded_file:
    st.divider()
    
    # 用 Tabs 讓介面更好看
    tab1, tab2, tab3 = st.tabs(["數據總覽", "關聯分析", "模型預測"])

    with tab1:
        if show_summary:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("資料維度：", df.shape)
                st.write("各欄位統計：")
                st.dataframe(df.describe().T)
            with c2:
                st.write(f"{target_col} 的數值分布：")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df[target_col], kde=True, ax=ax, color='#87AFC7', edgecolor='white')
                st.pyplot(fig)

    with tab2:
        if show_corr:
            st.subheader("特徵相關性矩陣")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
            numeric_df = df.select_dtypes(include=['number'])
            custom_cmap = sns.light_palette("#FFD8D8", as_cmap=True)
            sns.heatmap(numeric_df.corr(), annot=True, cmap=custom_cmap, ax=ax_corr, fmt=".2f")
            st.pyplot(fig_corr)
        else:
            st.info("要勾選上方的「相關性熱圖」才能看到喔")

    with tab3:
        if show_ml:
            st.subheader(f"SVM 模型評估 ({kernel_type} Kernel)")
            # 訓練邏輯
            X = df.select_dtypes(include=['number']).drop(columns=['target', 'count'], errors='ignore')
            y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(kernel=kernel_type, probability=True)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # 顯示結果
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("**分類報表：**")
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)
            with res_col2:
                st.markdown("**混淆矩陣：**")
                fig_cm, ax_cm = plt.subplots()
                # 使用 sns.light_palette 根據顏色產生漸層色
                custom_cmap = sns.light_palette("#CCCCFF", as_cmap=True)

                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cmap=custom_cmap, ax=ax_cm)
                st.pyplot(fig_cm)
else:
    st.warning("要先上傳 CSV 檔案喔")
