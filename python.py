import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cấu hình trang
st.set_page_config(
    page_title="Fraud Data Integrity Auditor",
    page_icon="🛡️",
    layout="wide"
)

# Giao diện Header
st.title("🛡️ Fraud Data Integrity Auditor")
st.markdown("""
    **Hệ thống kiểm định chất lượng dữ liệu** dành cho đại dự án *Knowledge Matrix*.
    Công cụ này giúp Nữ vương phát hiện các logic "ngáo" trước khi đưa dữ liệu vào huấn luyện Model AI.
""")

# Upload file
uploaded_file = st.file_uploader("Quăng file master_fraud_data.csv vào đây bồ ơi", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- TAB NAVIGATION ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard Tổng quan", 
            "🔍 Kiểm tra Nghịch lý", 
            "🌡️ Ma trận Tương quan",
            "📋 Dữ liệu thô"
        ])

        with tab1:
            st.subheader("Chỉ số sức khỏe dữ liệu")
            c1, c2, c3, c4 = st.columns(4)
            
            total_rows = len(df)
            c1.metric("Tổng giao dịch", f"{total_rows:,}")
            
            if 'is_fraud' in df.columns:
                fraud_rate = (df['is_fraud'].sum() / total_rows) * 100
                c2.metric("Tỷ lệ Gian lận", f"{fraud_rate:.2f}%")
            
            c3.metric("Số lượng Features", len(df.columns))
            
            # Check missing values
            missing = df.isnull().sum().sum()
            c4.metric("Dữ liệu trống (NULL)", missing, delta_color="inverse" if missing > 0 else "normal")

            st.write("---")
            st.subheader("Phân bổ loại hình Gian lận (Fraud Type)")
            if 'fraud_type' in df.columns:
                fraud_counts = df['fraud_type'].value_counts()
                st.bar_chart(fraud_counts)
            else:
                st.info("Không tìm thấy cột 'fraud_type' để vẽ biểu đồ.")

        with tab2:
            st.subheader("Phát hiện Nghịch lý Hành vi")
            
            # 1. Kiểm tra Vận tốc phi lý (Impossible Travel)
            if all(col in df.columns for col in ['location_diff_km', 'time_gap_prev_min']):
                st.markdown("##### 🚀 1. Kiểm tra di chuyển 'siêu thanh'")
                temp_df = df[df['time_gap_prev_min'] > 0].copy()
                temp_df['speed'] = temp_df['location_diff_km'] / (temp_df['time_gap_prev_min'] / 60)
                
                crazy_travel = temp_df[temp_df['speed'] > 1200]
                if not crazy_travel.empty:
                    st.error(f"Phát hiện {len(crazy_travel)} giao dịch có vận tốc di chuyển > 1200km/h (Nhanh hơn máy bay).")
                    st.dataframe(crazy_travel[['transaction_id', 'location_diff_km', 'time_gap_prev_min', 'speed']])
                else:
                    st.success("Không phát hiện nghịch lý di chuyển. Logic Địa lý: OK!")

            # 2. Kiểm tra Số dư và Số tiền tiêu
            if all(col in df.columns for col in ['amount', 'balance_before']):
                st.markdown("##### 💸 2. Kiểm tra chi tiêu vượt số dư")
                over_spend = df[df['amount'] > df['balance_before']]
                if not over_spend.empty:
                    st.error(f"Phát hiện {len(over_spend)} giao dịch có số tiền lớn hơn số dư tài khoản.")
                    st.dataframe(over_spend[['transaction_id', 'amount', 'balance_before']])
                else:
                    st.success("Tất cả giao dịch đều nằm trong phạm vi số dư. Logic Tài chính: OK!")

            # 3. Kiểm tra logic TX đầu tiên
            if 'is_first_large_tx' in df.columns and 'amount' in df.columns:
                st.markdown("##### ⚠️ 3. Kiểm tra nhãn 'First Large TX'")
                # Giả sử threshold tối thiểu cho Large TX là 1,000,000
                wrong_label = df[(df['is_first_large_tx'] == 1) & (df['amount'] < 1000000)]
                if not wrong_label.empty:
                    st.warning(f"Có {len(wrong_label)} dòng ghi là 'Lớn đầu tiên' nhưng số tiền < 1 triệu.")
                    st.dataframe(wrong_label[['transaction_id', 'amount', 'is_first_large_tx']])
                else:
                    st.success("Logic phân loại giao dịch lớn đầu tiên: OK!")

        with tab3:
            st.subheader("Correlation Heatmap - Soi độ 'Khôn' của Features")
            st.write("Dữ liệu khôn là khi các biến FORMULA (như amount và amount_log) có tương quan gần bằng 1 (đỏ đậm).")
            
            # Chỉ lấy top 20 features quan trọng nhất để tránh heatmap quá rối
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 20:
                selected_cols = numeric_cols[:20]
                st.write("*(Đang hiển thị 20 features đầu tiên để tối ưu giao diện)*")
            else:
                selected_cols = numeric_cols

            if selected_cols:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = df[selected_cols].corr()
                sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, ax=ax)
                st.pyplot(fig)

        with tab4:
            st.subheader("Dữ liệu thô (Raw Data Exploration)")
            st.dataframe(df)

    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc file: {e}")

else:
    st.info("Bồ vui lòng upload file CSV để tôi bắt đầu kiểm tra nhé! 🌸🌙👑")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("⚡ **Tình trạng máy i7:** Đang gánh dự án rất tốt!")
st.sidebar.write("👑 **Nữ vương:** Đang online")
