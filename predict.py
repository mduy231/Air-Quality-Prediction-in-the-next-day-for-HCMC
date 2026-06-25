"""
predict.py - Pipeline dự đoán AQI ngày mai (Regression + Classification)
Nhóm 6 - Nhập môn Khoa học Dữ liệu

Chạy mỗi tối 22:00 (hoặc một giờ nào đó cố định hàng ngày) để:
    1. Gọi Open-Meteo API lấy dữ liệu hôm nay
    2. Tạo features (lag, rolling, sin/cos...)
    3. Dự đoán GIÁ TRỊ AQI ngày mai (Regression - best_model.pkl)
    4. Dự đoán MỨC ĐỘ AQI ngày mai (Classification - best_classifier.pkl)
    5. Đưa ra khuyến nghị hành động cụ thể
    6. Ghi tất cả vào predictions_log.csv

Cách chạy:
    python predict.py
"""

import os
import json
import joblib
import requests
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# CẤU HÌNH CHUNG
warnings.filterwarnings("ignore", category=UserWarning)

BASE = os.path.dirname(os.path.abspath(__file__))

LAT, LON = 10.82302, 106.62965  # Tọa độ TP. Hồ Chí Minh
PROC_PATH = os.path.join(BASE, "outputs", "data_processed.csv")
LOG_PATH = os.path.join(BASE, "outputs", "predictions_log.csv")
JSON_PATH = os.path.join(BASE, "outputs", "latest_prediction.json")

REG_PATH = os.path.join(BASE, "models", "best_regressor.pkl")
CLF_PATH = os.path.join(BASE, "models", "best_classifier.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler.pkl")
FEATURE_PATH = os.path.join(BASE, "models", "feature_cols.pkl")

AQI_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

# HÀNH ĐỘNG CỤ THỂ THEO TỪNG MỨC AQI
ACTION_MAP = {
    "Good": {
        "emoji": "🟢",
        "summary": "Không khí trong lành",
        "actions": [
            "An toàn cho mọi hoạt động ngoài trời, kể cả thể thao cường độ cao",
            "Có thể mở cửa sổ thông gió tự nhiên",
            "Không cần đeo khẩu trang chống bụi mịn",
        ],
    },
    "Moderate": {
        "emoji": "🟡",
        "summary": "Chất lượng không khí ở mức chấp nhận được",
        "actions": [
            "Người nhạy cảm (trẻ em, người già, người có bệnh hô hấp) nên hạn chế "
            "hoạt động ngoài trời kéo dài",
            "Người bình thường vẫn sinh hoạt như thường lệ",
            "Có thể cân nhắc đeo khẩu trang nếu phải ra ngoài lâu",
        ],
    },
    "Unhealthy for Sensitive Groups": {
        "emoji": "🟠",
        "summary": "Ảnh hưởng đến nhóm nhạy cảm",
        "actions": [
            "Trẻ em, người già, phụ nữ mang thai, người có bệnh tim/phổi nên "
            "hạn chế ra ngoài",
            "Đeo khẩu trang đạt chuẩn (N95/KN95) khi ra đường",
            "Hạn chế tập thể dục ngoài trời, đặc biệt vào giờ cao điểm giao thông",
            "Đóng cửa sổ nếu nhà gần đường lớn",
        ],
    },
    "Unhealthy": {
        "emoji": "🔴",
        "summary": "Có hại cho sức khỏe",
        "actions": [
            "Toàn bộ người dân nên hạn chế hoạt động ngoài trời",
            "Bắt buộc đeo khẩu trang N95/KN95 khi ra ngoài",
            "Đóng kín cửa sổ, sử dụng máy lọc không khí trong nhà nếu có",
            "Người có bệnh hô hấp/tim mạch nên ở trong nhà hoàn toàn",
            "Trẻ em nên hạn chế ra ngoài chơi, vui chơi trong nhà",
        ],
    },
    "Very Unhealthy": {
        "emoji": "🟣",
        "summary": "Cảnh báo sức khỏe nghiêm trọng",
        "actions": [
            "Khuyến cáo ở trong nhà - chỉ ra ngoài khi thực sự cần thiết",
            "Đeo khẩu trang chuyên dụng (N95 trở lên) nếu bắt buộc phải ra ngoài",
            "Sử dụng máy lọc không khí, tránh mọi hoạt động thể chất ngoài trời",
            "Theo dõi sát sức khỏe nếu có triệu chứng ho, khó thở, tức ngực",
            "Cân nhắc thông báo cho người thân lớn tuổi, trẻ nhỏ trong khu vực",
        ],
    },
    "Hazardous": {
        "emoji": "⚫",
        "summary": "Mức nguy hiểm - Báo động đỏ",
        "actions": [
            "Không ra ngoài trừ trường hợp khẩn cấp",
            "Đóng kín toàn bộ cửa, dùng máy lọc không khí công suất cao",
            "Liên hệ cơ sở y tế ngay nếu có dấu hiệu khó thở",
            "Theo dõi thông báo từ cơ quan y tế và môi trường địa phương",
        ],
    },
}


def AQI_CATEGORY(value):
    """
    Phân loại mức AQI theo thang US EPA
    Input : Giá trị AQI (số thực)
    Output: Tên mức (string)
    """
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Moderate"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif value <= 200:
        return "Unhealthy"
    elif value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# 1. GỌI API
def fetch_today():
    """Lấy dữ liệu chất lượng không khí hôm nay từ Open-Meteo"""

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=pm2_5,pm10,us_aqi,nitrogen_dioxide,ozone,"
        "carbon_monoxide,sulphur_dioxide,"
        "aerosol_optical_depth,dust,uv_index"
        "&timezone=Asia%2FHo_Chi_Minh"
        "&forecast_days=1"
    )

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()["hourly"]

        def daily_mean(key):
            vals = [v for v in data.get(key, []) if v is not None]
            return round(float(np.mean(vals)), 2) if vals else np.nan

        result = {
            "date": data["time"][0][:10],
            "pm2_5": daily_mean("pm2_5"),
            "pm10": daily_mean("pm10"),
            "us_aqi": daily_mean("us_aqi"),
            "nitrogen_dioxide": daily_mean("nitrogen_dioxide"),
            "ozone": daily_mean("ozone"),
            "carbon_monoxide": daily_mean("carbon_monoxide"),
            "sulphur_dioxide": daily_mean("sulphur_dioxide"),
            "aerosol_optical_depth": daily_mean("aerosol_optical_depth"),
            "dust": daily_mean("dust"),
            "uv_index": daily_mean("uv_index"),
        }
        print(
            f"Gọi API hoàn tất: Đã lấy được các chỉ số hiện tại - pm2_5 = {result['pm2_5']}, us_aqi = {result['us_aqi']}"  # 2 chỉ số quan trọng nhất
        )
        return result

    except Exception as e:
        print(f"Gọi API lỗi: {e} - dùng dữ liệu ngày cuối cùng trong lịch sử")
        return None


# 2. TẠO FEATURES
def make_features(today_data, hist_df):
    """Tạo đầy đủ features (lag, rolling, diff, sin/cos, binary) cho ngày hôm nay"""

    today = pd.DataFrame([today_data])
    today["date"] = pd.to_datetime(today["date"])

    # Kết hợp 60 ngày gần nhất với hôm nay để đủ dữ liệu tính rolling_30
    combined = pd.concat([hist_df.tail(60), today], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    target_col = "us_aqi"

    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        combined[f"t-{lag}"] = combined[target_col].shift(lag)

    # Rolling features (shift(1) trước rolling - tránh data leakage)
    for w in [3, 7, 14, 30]:
        shifted = combined[target_col].shift(1)
        combined[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        combined[f"rolling_std_{w}"] = shifted.rolling(w).std()
        combined[f"rolling_max_{w}"] = shifted.rolling(w).max()
        combined[f"rolling_min_{w}"] = shifted.rolling(w).min()

    # Diff features
    combined["diff_1"] = combined[target_col].diff(1)
    combined["diff_7"] = combined[target_col].diff(7)

    # Time features
    combined["month"] = combined["date"].dt.month
    combined["weekday"] = combined["date"].dt.dayofweek
    combined["day_of_year"] = combined["date"].dt.dayofyear

    combined["month_sin"] = np.sin(2 * np.pi * combined["month"] / 12)
    combined["month_cos"] = np.cos(2 * np.pi * combined["month"] / 12)
    combined["weekday_sin"] = np.sin(2 * np.pi * combined["weekday"] / 7)
    combined["weekday_cos"] = np.cos(2 * np.pi * combined["weekday"] / 7)
    combined["day_of_year_sin"] = np.sin(2 * np.pi * combined["day_of_year"] / 365)
    combined["day_of_year_cos"] = np.cos(2 * np.pi * combined["day_of_year"] / 365)

    combined["is_weekend"] = (combined["weekday"] >= 5).astype(int)
    combined["is_dry_season"] = combined["month"].isin([12, 1, 2, 3, 4]).astype(int)

    # Lấy dòng cuối cùng = hôm nay
    return combined.iloc[[-1]].copy()


# 3. DỰ ĐOÁN
def predict_regression(features_row, reg, scaler, feature_cols):
    """Dự đoán GIÁ TRỊ AQI ngày mai"""

    # Đồng bộ hóa các cột và sắp xếp đúng thứ tự lúc huấn luyện mô hình
    available = [c for c in feature_cols if c in features_row.columns]
    X = features_row[available].copy()
    X = X.fillna(X.median())

    # Áp dụng Scaler
    X_scaled = X.copy()
    if hasattr(scaler, "feature_names_in_"):
        # Lọc ra những cột Scaler
        scale_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
        if scale_cols:
            # Chỉ áp dụng biến đổi cho các cột đó
            X_scaled[scale_cols] = scaler.transform(X[scale_cols])
    else:
        X_scaled = scaler.transform(X)

    pred = float(
        np.clip(reg.predict(X_scaled)[0], 0, 500)
    )  # Gọi mô hình hồi quy để dự đoán giá trị

    return round(pred, 1)


def predict_classification(features_row, clf, scaler, feature_cols):
    """Dự đoán MỨC ĐỘ AQI ngày mai + độ tin cậy"""

    # Đồng bộ hóa các cột và sắp xếp đúng thứ tự lúc huấn luyện mô hình
    available = [c for c in feature_cols if c in features_row.columns]
    X = features_row[available].copy()
    X = X.fillna(X.median())

    # Áp dụng Scaler
    X_scaled = X.copy()
    if hasattr(scaler, "feature_names_in_"):
        scale_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
        if scale_cols:
            X_scaled[scale_cols] = scaler.transform(X[scale_cols])
    else:
        X_scaled = scaler.transform(X)

    pred_idx = int(clf.predict(X_scaled)[0])  # Gọi mô hình phân loại để dự đoán mức độ
    pred_label = AQI_LABELS[
        pred_idx
    ]  # Đưa các mức độ (Số nguyên) về chuỗi kí tự để dễ dàng nhận biết

    """
    Đo lường Độ tin cậy (Confidence Score) của mô hình.
    Thay vì chỉ nói "Ngày mai không khí Xấu", nó sẽ cho biết "Chắc chắn 85.5% là ngày mai không khí Xấu".
    """
    confidence = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_scaled)[0]
        confidence = round(float(proba[pred_idx]) * 100, 1)

    return pred_label, confidence


# 4. LƯU KẾT QUẢ
def save_prediction(
    today_date,
    today_aqi,
    pred_reg_aqi,
    pred_class_label,
    class_confidence,
    reg_class_agree,
):
    """Ghi kết quả vào predictions_log.csv (thêm 1 hàng mỗi ngày, không trùng)"""

    tomorrow = (datetime.strptime(today_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    utc_plus_7 = datetime.now(timezone.utc) + timedelta(hours=7)

    row = {
        "prediction_date": today_date,  # Ngày thực hiện dự báo (Ngày hôm nay)
        "predicted_for_date": tomorrow,  # Ngày mục tiêu (Ngày mai)
        "actual_aqi_today": round(today_aqi, 1)
        if pd.notna(today_aqi)
        else None,  # AQI thực tế hôm nay
        "predicted_aqi_tomorrow": pred_reg_aqi,  # Dự đoán AQI ngày mai (Hồi quy)
        "aqi_category": AQI_CATEGORY(
            pred_reg_aqi
        ),  # Dự đoán mức độ AQI ngày mai (Hồi quy)
        "predicted_class": pred_class_label,  # Dự đoán mức độ AQI ngày mai (Phân loại)
        "class_confidence": class_confidence,  # Mức độ tin cậy
        "models_agree": reg_class_agree,  # Sự đồng thuận của 2 mô hình
        "generated_at": utc_plus_7.strftime("%Y-%m-%d %H:%M:%S"),
    }

    df_new = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
        df_log = df_log[df_log["prediction_date"] != today_date]  # tránh duplicate
        df_log = pd.concat([df_log, df_new], ignore_index=True)
    else:
        df_log = df_new

    df_log.to_csv(LOG_PATH, index=False)
    return row


# MAIN
def main():
    now_vn = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime(
        "%Y-%m-%d %H:%M"
    )
    print(f"\n[{now_vn}] Bắt đầu dự đoán AQI cho ngày mai!")
    print("─" * 60)

    # Load models
    print("- Load models.")
    if not os.path.exists(REG_PATH):
        print(
            f"  Không tìm thấy {REG_PATH} - chạy notebook 03 trước hoặc chạy notebook tổng hợp"
        )
        return
    if not os.path.exists(CLF_PATH):
        print(
            f"  Không tìm thấy {CLF_PATH} - chạy notebook 04 trước hoặc chạy notebook tổng hợp"
        )
        return

    model = joblib.load(REG_PATH)  # Regression - dự đoán CHỈ SỐ
    clf = joblib.load(CLF_PATH)  # Classification - dự đoán MỨC ĐỘ
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    hist_df = pd.read_csv(PROC_PATH, parse_dates=["date"])
    print(f"    Đã load {len(feature_cols)} features cho cả 2 model")

    # Lấy dữ liệu hôm nay
    print("- Gọi Open-Meteo API.")
    today_data = fetch_today()

    if today_data is None:
        print(
            "Gọi Open-Meteo API thất bại sẽ lấy giá trị các chỉ số của ngày hôm qua để dự đoán (Nếu không tồn tại sẽ lấy giá trị trung bình của dữ liệu trong quá khứ)."
        )
        last = hist_df.iloc[-1]
        today_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "us_aqi": float(last["us_aqi"]),
            "pm2_5": float(last.get("pm2_5", 26.33)),
            "pm10": float(last.get("pm10", 34.48)),
            "nitrogen_dioxide": float(last.get("nitrogen_dioxide", 31.89)),
            "ozone": float(last.get("ozone", 53.66)),
            "carbon_monoxide": float(last.get("carbon_monoxide", 606.45)),
            "sulphur_dioxide": float(last.get("sulphur_dioxide", 20.34)),
            "aerosol_optical_depth": float(last.get("aerosol_optical_depth", 0.35)),
            "dust": float(last.get("dust", 0.50)),
            "uv_index": float(last.get("uv_index", 1.83)),
        }

    # Tạo features
    print("- Tạo features.")
    features = make_features(today_data, hist_df)
    print(f"    {features.shape[1]} cột features")

    # Dự đoán bằng cả 2 models
    print("- Dự đoán (Regression + Classification).")

    # Model 1: Regression - dự đoán CHỈ SỐ AQI
    pred_reg_aqi = predict_regression(features, model, scaler, feature_cols)
    reg_category = AQI_CATEGORY(pred_reg_aqi)

    # Model 2: Classification - dự đoán MỨC ĐỘ AQI
    pred_class_label, class_confidence = predict_classification(
        features, clf, scaler, feature_cols
    )

    # Kiểm tra 2 model có đồng thuận không
    models_agree = reg_category == pred_class_label

    print(f"    Regression     : AQI = {pred_reg_aqi} → {reg_category}")
    print(
        f"    Classification : {pred_class_label} (Mức độ tin cậy {class_confidence}%)"
    )
    if not models_agree:
        print("    ⚠️  2 model KHÔNG ĐỒNG THUẬN")
        print("         → Ưu tiên Classification làm cơ sở khuyến nghị")
        print(
            "         → Regression có thể đang underestimate do AQI hôm nay cao bất thường"
        )
    else:
        print("    ✅  2 model ĐỒNG THUẬN - độ tin cậy cao")

    # Lưu kết quả
    print("- Lưu kết quả.")
    today_aqi = today_data.get("us_aqi", np.nan)
    result = save_prediction(
        today_data["date"],
        today_aqi,
        pred_reg_aqi,
        pred_class_label,
        class_confidence,
        models_agree,
    )

    # Khuyến nghị hành động cụ thể
    # Dùng kết quả Classification làm cơ sở khuyến nghị (đáng tin cậy hơn cho
    # quyết định hành vi vì được huấn luyện với mục đích phân loại mức độ)
    action_info = ACTION_MAP.get(pred_class_label, ACTION_MAP["Moderate"])

    print("─" * 60)
    print(f"{action_info['emoji']}  KẾT QUẢ DỰ ĐOÁN CHO NGÀY MAI")
    print("─" * 60)
    print(f"   Chỉ số AQI dự đoán  : {pred_reg_aqi} ({reg_category})")
    print(f"   Mức độ phân loại    : {pred_class_label}")
    print(f"   Độ tin cậy          : {class_confidence}%")
    print(f"   Tóm tắt             : {action_info['summary']}")
    print("\n   📋 KHUYẾN NGHỊ HÀNH ĐỘNG:")
    for i, action in enumerate(action_info["actions"], 1):
        print(f"      {i}. {action}")
    print("─" * 60)
    print(f"   Lưu vào: {LOG_PATH}")
    print("   Dashboard sẽ cập nhật trong vòng 24h (dcc.Interval)\n")

    # Xuất JSON chi tiết (để debug / tích hợp hệ thống khác)
    full_result = {
        **result,
        "reg_category": reg_category,
        "action_summary": action_info["summary"],
        "action_list": action_info["actions"],
        "action_emoji": action_info["emoji"],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)

    return full_result


if __name__ == "__main__":
    main()
