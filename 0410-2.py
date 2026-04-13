# 先下載
# pip install streamlit pandas seaborn matplotlib scikit-learn
# 執行
# streamlit run 0410-2.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 基本設定
sns.set_theme(style="whitegrid", palette="muted")
st.set_page_config(page_title="數學系人工智慧概論", layout="wide")

st.title("數據分析與演算法小工具")

with st.container(border=True):
    st.subheader("嗨嗨~")
    uploaded_file = st.file_uploader("請先上傳你的 CSV 檔", type=["csv"])
    st.divider()
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3]) # # 按比例分割欄位
    
    with col1:
        st.write("**1.功能清單**")
        show_summary = st.checkbox("基本資訊")
        show_corr = st.checkbox("相關性矩陣")
        show_ml = st.checkbox("模型預測")

    if uploaded_file:
        df = pd.read_csv(uploaded_file) # 讀取上傳的 CSV
        
        # 二元化轉化（以中位數為界）
        if 'count' in df.columns:
            #算出 count 欄位的中位數
            median_val = df['count'].median()
            # 高於中位數設為1(熱門), 低於或等於設為0(冷門)
            df['target'] = (df['count'] > median_val).astype(int)
            # 然後就會多出一個 'target' ，圓餅圖就是畫這個
        
        with col2:
            st.write("**2. 設定目標與分佈**")
            # 從選擇的數值欄位中來畫分佈圖
            target_col = st.selectbox("選擇要觀察的欄位", df.select_dtypes(include=['number']).columns)
        
        with col3:
            st.write("**3. 訓練設定**")
            # 用滑動的方式設定測試集比例
            test_size_percent = st.slider("資料集比例 (%)", 20, 40, 30, 5)
            test_size = test_size_percent / 100
            
            st.caption(f"訓練集: {100-test_size_percent}% | 測試集: {test_size_percent}%")
            
            # 演算法名稱，可以選擇要使用的演算法
            algo_choice = st.selectbox("選擇演算法", [
                "k-Nearest Neighbors (kNN)",
                "決策樹 (Decision Tree)", 
                "隨機森林 (Random Forest)", 
                "邏輯斯迴歸 (Logistic Regression)",
                "支持向量機 (SVM)"
            ])
            
        with col4:
            st.write("**4. 演算法參數微調**")
            # 根據選擇的演算法，來顯示對應的參數設定
            
            if algo_choice == "支持向量機 (SVM)":
                kernel_type = st.radio("Kernel (核函數)", ["rbf", "linear"], horizontal=True)
                c_val = st.number_input("C (懲罰係數)", 0.01, 100.0, 1.0, 0.1)
            
            elif algo_choice == "k-Nearest Neighbors (kNN)":
                k_val = st.number_input("k (鄰居數)", 1, 100, 5, 1)
                
            elif algo_choice == "決策樹 (Decision Tree)":
                crit = st.radio("Criterion (判定標準)", ["gini", "entropy"], horizontal=True)
                depth = st.number_input("Max Depth (最大深度)", 1, 100, 5, 1)
                
            elif algo_choice == "隨機森林 (Random Forest)":
                n_trees = st.number_input("n_estimators (森林樹數)", 1, 1000, 100, 10)
                
            elif algo_choice == "邏輯斯迴歸 (Logistic Regression)":
                c_lr = st.number_input("C (正則化強度)", 0.01, 100.0, 1.0, 0.1)

# 主要的數據展示區
if uploaded_file:
    st.divider()
    tab1, tab2, tab3 = st.tabs(["數據總覽", "相關性矩陣", "模型預測"])

    with tab1:
        if show_summary:
            # --- 第一層：基本維度與缺失值 ---
            st.markdown("### 基本資訊")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("資料列數 (Rows)", df.shape[0])
            with m2:
                st.metric("資料欄數 (Columns)", df.shape[1])
            with m3:
                total_null = df.isnull().sum().sum()
                st.metric("總缺失值數量", total_null)

            st.divider()

            # --- 第二層：特徵資訊與缺失值細節 ---
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 特徵欄位與類型")
                # 建立一個 DataFrame 來顯示欄位類型和缺失值
                info_df = pd.DataFrame({
                    "資料類型": df.dtypes.astype(str),
                    "缺失值數量": df.isnull().sum(),
                    "非空值數量": df.notnull().sum()
                })
                st.dataframe(info_df, use_container_width=True)
            
            with c2:
                st.markdown("#### 各欄位統計摘要")
                st.dataframe(df.describe().T)

            st.divider()

            # 第三層：類別分佈情形 (Label Distribution) 
            # 輸出長方圖 與 圓餅圖
            st.markdown("### 分佈分析")
            d1, d2 = st.columns(2)
            
            with d1:
                st.write(f"**數值型分佈: {target_col}**")
                fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
                sns.histplot(df[target_col], kde=True, ax=ax_hist, color='#87AFC7')
                st.pyplot(fig_hist)

            with d2:
                # 類別分佈 (Label Distribution)
                label_col = 'target' if 'target' in df.columns else df.select_dtypes(include=['object', 'int']).columns[-1]
                st.write(f"**類別分布 (Label Distribution): `{label_col}`**")
                
                fig_pie, ax_pie = plt.subplots(figsize=(5, 3.5))
                # 取得 target 的計數 (0 有幾個，1 有幾個)
                dist = df[label_col].value_counts()
                
                # 畫圓餅圖
                ax_pie.pie(dist, labels=['冷門 (0)', '熱門 (1)'], # 設定標籤
                           autopct='%1.1f%%', # 顯示百分比到小數點後一位
                           startangle=140, # 旋轉角度
                           colors=sns.color_palette("pastel")) # 顏色
                ax_pie.axis('equal')  # 確保是圓形
                st.pyplot(fig_pie)
                
                # 同時顯示數值表格
                st.write("詳細計數：")
                st.table(dist)
        else:
            st.info("要勾選上方的「基本資訊」才能看到喔")
            aol1, aol2, aol3 = st.columns(3)

            with aol2:
                st.image("0410-2.png")
    with tab2:
        if show_corr:
            st.subheader("特徵相關性矩陣")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
            numeric_df = df.select_dtypes(include=['number'])
            custom_cmap = sns.light_palette("#FFD8D8", as_cmap=True)
            sns.heatmap(numeric_df.corr(), annot=True, cmap=custom_cmap, ax=ax_corr, fmt=".2f")
            st.pyplot(fig_corr) 
        else:
            st.info("要勾選上方的「相關性矩陣」才能看到喔")
            aol1, aol2, aol3 = st.columns(3)

            with aol2:
                st.image("0410-4.png")

    with tab3:
        if show_ml:
            st.subheader(f"模型評估: {algo_choice}")
            
            # 1. 準備數據
            # X 為特徵 y 為目標（預測對象）
            #X = df.select_dtypes(include=['number']).drop(columns=['target', 'count'], errors='ignore')
            #y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
            X = df.select_dtypes(include=['number']).drop(columns=['target', 'count'], errors='ignore')
            y = df['target']

            # 切分訓練集與測試集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # 標準化特徵 (Standardization)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 2. 根據介面上選擇的演算法與選取參數
            if algo_choice == "支持向量機 (SVM)":
                model = SVC(kernel=kernel_type, C=c_val, probability=True)
            elif algo_choice == "k-Nearest Neighbors (kNN)":
                model = KNeighborsClassifier(n_neighbors=k_val)
            elif algo_choice == "決策樹 (Decision Tree)":
                model = DecisionTreeClassifier(criterion=crit, max_depth=depth)
            elif algo_choice == "隨機森林 (Random Forest)":
                model = RandomForestClassifier(n_estimators=n_trees)
            elif algo_choice == "邏輯斯迴歸 (Logistic Regression)":
                model = LogisticRegression(C=c_lr)

            # 3. 模型訓練
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # 4. 獲取預測機率 (畫 ROC 曲線)
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_probs = model.decision_function(X_test_scaled)

            # 5. 計算 ROC 與 AUC
            #from sklearn.metrics import roc_curve, auc 
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            
            # 結果展示環節 
            st.markdown("### 1. 量化指標 (Classification Report)")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

            # 混淆矩陣與 ROC 曲線
            st.markdown("### 2. 視覺化圖表")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.write("**混淆矩陣 (Confusion Matrix)**")
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                # 使用 sns.light_palette 根據顏色產生漸層色
                custom_cmap = sns.light_palette("#CCCCFF", as_cmap=True)

                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cmap=custom_cmap, ax=ax_cm)
                st.pyplot(fig_cm)
            
            with res_col2:
                st.write(f"**ROC 曲線 (AUC = {roc_auc:.2f})**")
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
                ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--') # 畫 0.5 對角的基準線
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
        else:
            st.info("要勾選上方的「模型評估報表」才能看到喔")  
            aol1, aol2, aol3 = st.columns([3,2, 1])

            with aol2:
                st.image("0410-1.png")
else:
    st.info("記得上傳 CSV 檔！")
    col1, col2, col3 = st.columns(3)

    with col2:
        pot_list = ["giphy.gif", "0410-8.gif", "pika.gif"]
        
        # 隨機選一個
        random_pot = random.choice(pot_list)
        st.image(random_pot, caption="隨機出現的驚喜！")
