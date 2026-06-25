"""
dashboard.py - Dashboard Chất Lượng Không Khí TP. Hồ Chí Minh
Nhóm 6 - Nhập môn Khoa học Dữ liệu

Cách chạy:
    pip install dash plotly dash-bootstrap-components pandas numpy joblib shap statsmodels
    python dashboard.py
    → Mở trình duyệt: http://localhost:8050

Cấu trúc 4 tab:
    Tab EDA     : 11 biểu đồ phân tích dữ liệu lịch sử
    Tab Model   : Kết quả mô hình ML      (Cần best_regressor.pkl)
    Tab SHAP    : Giải thích mô hình      (Cần best_regressor.pkl + shap)
    Tab Dự đoán : Kết quả predict.py      (Cần predictions_log.csv)
"""

# ══════════════════════════════════════════════════════════════════════════════
# KHAI BÁO THƯ VIỆN
# ══════════════════════════════════════════════════════════════════════════════

# Thư viện chuẩn của Python
import json
import os
import warnings
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Xử lý dữ liệu & Thống kê
import numpy as np
import pandas as pd
import shap
from statsmodels.tsa.seasonal import seasonal_decompose

# Trực quan hóa dữ liệu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap

# Giao diện Web & Dashboard
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dash_table, dcc, html

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════════════

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "air_quality_historical.csv")
PROC_PATH = os.path.join(BASE, "outputs", "data_processed.csv")
MDL_PATH = os.path.join(BASE, "models", "best_regressor.pkl")
XTEST = os.path.join(BASE, "outputs", "X_test.csv")
YTEST = os.path.join(BASE, "outputs", "y_test.csv")
META = os.path.join(BASE, "models", "metadata.json")
PRED = os.path.join(BASE, "outputs", "predictions_log.csv")

# Thiết lập khoảng thời gian (Interval) tính bằng mili-giây
IV_MS = 24 * 60 * 60 * 1000  # Tương đương 1 ngày

MONTHS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]
DAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

AQI_COLORS = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
}
AQI_BINS = [0, 50, 100, 150, 200, 300, 500]
AQI_LABELS = list(AQI_COLORS.keys())
AQI_MAPPING = {label: idx for idx, label in enumerate(AQI_LABELS)}


# Hàm AQI_CATEGORY()
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


# Tạo Custom Gradient Colormap từ AQI_COLORS và AQI_BINS
# Chuẩn hóa các mốc AQI vì LinearSegmentedColormap là thang đo 0.0 đến 1.0
MAX_AQI = 500
positions = [
    val / MAX_AQI for val in AQI_BINS
]  # positions sẽ trở thành [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]

# Vì AQI_BINS có 7 mốc nhưng AQI_COLORS chỉ có 6 màu
colors = list(AQI_COLORS.values())
gradient_colors = colors + [colors[-1]]  # Nhân đôi màu cuối cùng

# Ghép vị trí và màu sắc lại để tạo dải gradient
color_mapping = list(
    zip(positions, gradient_colors)
)  # Gán các màu tương ứng với từng giá trị
AQI_gradient_cmap = LinearSegmentedColormap.from_list(
    "AQI_gradient", color_mapping
)  # Tạo ra dãy màu liên tục thay vì riêng lẻ

# Màu sắc dùng chung cho toàn bộ giao diện và biểu đồ
C = dict(
    bg="#0F172A",  # Màu nền chính của toàn trang web
    card="#1E293B",  # Màu nền của các thẻ
    border="#334155",  # Màu của các đường viền
    text="#F1F5F9",
    sub="#94A3B8",
    accent="#38BDF8",
    acc2="#818CF8",
    green="#00E400",
    yellow="#FFFF00",
    orange="#FF7E00",
    red="#FF0000",
)

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="Inter, Segoe UI, sans-serif", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"]),
    margin=dict(l=44, r=20, t=48, b=44),
)

POLS = {
    "us_aqi": "US AQI",
    "pm2_5": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "ozone": "Ozone (µg/m³)",
    "nitrogen_dioxide": "NO₂ (µg/m³)",
    "carbon_monoxide": "CO (µg/m³)",
    "sulphur_dioxide": "SO₂ (µg/m³)",
}

# ══════════════════════════════════════════════════════════════════════════════
# TẢI DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════════════


def load_hist():
    path = PROC_PATH if os.path.exists(PROC_PATH) else DATA_PATH
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().sum():
            df[col] = df[col].interpolate(method="linear")

    df["aqi_category"] = df["us_aqi"].apply(AQI_CATEGORY)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear

    return df


def load_model():
    if not os.path.exists(MDL_PATH):
        return None, None, None, None, {}

    try:
        model = joblib.load(MDL_PATH)
        X_test = pd.read_csv(XTEST, index_col=0, parse_dates=True)
        y_test = pd.read_csv(YTEST, index_col=0, parse_dates=True)
        y_pred = np.clip(model.predict(X_test), 0, 500)
        meta = json.load(open(META)) if os.path.exists(META) else {}

        return model, X_test, y_test, y_pred, meta

    except Exception as e:
        print(f"[WARN] load_model: {e}")
        return None, None, None, None, {}


def load_pred():
    if not os.path.exists(PRED):
        return None

    try:
        log = pd.read_csv(PRED, parse_dates=["prediction_date"])
        if "predicted_for_date" in log.columns:
            log["predicted_for_date"] = pd.to_datetime(
                log["predicted_for_date"], errors="coerce"
            )
        return log.sort_values("prediction_date").reset_index(drop=True)

    except Exception as e:
        print(f"[WARN] load_pred: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def aqi_clr(value):
    return AQI_COLORS.get(AQI_CATEGORY(value), C["yellow"])


def card(ch, ex=None):
    ex = ex or {}

    return html.Div(
        ch,
        style={
            "background": C["card"],
            "borderRadius": "12px",
            "padding": "18px",
            "border": f"1px solid {C['border']}",
            **ex,
        },
    )


def sec(icon, text):
    return html.Div(
        [
            html.I(
                className=f"bi bi-{icon}",
                style={"color": C["accent"], "marginRight": "7px", "fontSize": "1rem"},
            ),
            html.Span(
                text,
                style={"color": C["text"], "fontWeight": "600", "fontSize": "0.9rem"},
            ),
        ],
        style={"marginBottom": "12px", "display": "flex", "alignItems": "center"},
    )


def kpi(icon, label, value, color, sub=""):
    return html.Div(
        [
            html.I(
                className=f"bi bi-{icon}",
                style={
                    "fontSize": "1.4rem",
                    "color": color,
                    "display": "block",
                    "marginBottom": "5px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "1.8rem",
                    "fontWeight": "700",
                    "color": color,
                    "lineHeight": "1",
                },
            ),
            html.Div(
                label,
                style={
                    "color": C["text"],
                    "fontWeight": "600",
                    "marginTop": "4px",
                    "fontSize": "0.78rem",
                },
            ),
            html.Div(
                sub,
                style={"color": C["sub"], "fontSize": "0.68rem", "marginTop": "2px"},
            ),
        ],
        style={
            "background": C["card"],
            "border": f"1px solid {C['border']}",
            "borderRadius": "12px",
            "borderTop": f"3px solid {color}",
            "padding": "14px",
            "textAlign": "center",
            "flex": "1",
            "minWidth": "0",
        },
    )


def row(*ch, gap="12px", mb="12px"):
    return html.Div(
        list(ch),
        style={"display": "flex", "gap": gap, "marginBottom": mb, "flexWrap": "wrap"},
    )


def col(ch, flex="1"):
    return html.Div(ch, style={"flex": flex, "minWidth": "0"})


def gph(gid, h=300):
    return dcc.Graph(
        id=gid,
        config={
            "displayModeBar": False,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
        style={"height": f"{h}px"},
    )


def empty_st(emoji, title, sub, cmd=""):
    return html.Div(
        [
            html.Div(emoji, style={"fontSize": "3rem", "marginBottom": "14px"}),
            html.Div(
                title,
                style={
                    "color": C["text"],
                    "fontWeight": "700",
                    "fontSize": "1rem",
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                sub,
                style={
                    "color": C["sub"],
                    "fontSize": "0.85rem",
                    "marginBottom": "14px",
                    "maxWidth": "400px",
                    "margin": "0 auto 14px",
                },
            ),
            html.Code(
                cmd,
                style={
                    "background": "#0B1120",
                    "padding": "10px 20px",
                    "borderRadius": "8px",
                    "color": C["accent"],
                    "fontSize": "0.82rem",
                    "display": "inline-block",
                },
            )
            if cmd
            else html.Span(),
        ],
        style={"textAlign": "center", "padding": "80px 40px"},
    )


def note(*items):
    return html.Div(
        [
            html.Div(
                [
                    html.I(
                        className="bi bi-lightbulb-fill",
                        style={
                            "color": C["yellow"],
                            "fontSize": "0.9rem",
                            "marginRight": "6px",
                        },
                    ),
                    html.Span(
                        "Nhận xét",
                        style={
                            "color": C["text"],
                            "fontWeight": "700",
                            "fontSize": "0.84rem",
                        },
                    ),
                ],
                style={
                    "marginBottom": "8px",
                    "display": "flex",
                    "alignItems": "center",
                },
            ),
            html.Ul(
                [html.Li(i, style={"marginBottom": "3px"}) for i in items],
                style={
                    "color": C["sub"],
                    "fontSize": "0.79rem",
                    "lineHeight": "1.75",
                    "paddingLeft": "18px",
                    "margin": 0,
                },
            ),
        ],
        style={
            "background": C["card"],
            "border": f"1px solid {C['border']}",
            "borderLeft": f"3px solid {C['yellow']}",
            "borderRadius": "0 10px 10px 0",
            "padding": "12px 14px",
            "marginTop": "10px",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# APP DASH
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP],
    title="AQI Dashboard - TP.HCM | Nhóm 6",
    meta_tags=[{"name": "viewport", "content": "width=device-width,initial-scale=1"}],
    suppress_callback_exceptions=True,
)

_T = dict(
    background=C["card"],
    color=C["sub"],
    border=f"1px solid {C['border']}",
    borderBottom="none",
    borderRadius="8px 8px 0 0",
    padding="10px 20px",
    fontWeight="500",
    fontSize="0.83rem",
)

_TS = {
    **_T,
    "background": C["bg"],
    "color": C["accent"],
    "borderTop": f"2px solid {C['accent']}",
}

app.layout = html.Div(
    [
        dcc.Interval(id="iv", interval=IV_MS, n_intervals=0),
        # HEADER
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "🌿",
                                    style={"fontSize": "1.4rem", "marginRight": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.H5(
                                            "Chất Lượng Không Khí TP. Hồ Chí Minh",
                                            style={
                                                "margin": 0,
                                                "color": C["text"],
                                                "fontWeight": "700",
                                                "fontSize": "0.98rem",
                                            },
                                        ),
                                        html.P(
                                            "Dashboard - 2022-2026  |  Nhóm 6",
                                            style={
                                                "margin": 0,
                                                "color": C["sub"],
                                                "fontSize": "0.7rem",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        html.Div(
                            [
                                html.Span(
                                    id="hdr-t",
                                    style={"color": C["sub"], "fontSize": "0.7rem"},
                                ),
                                html.Span("  |  ", style={"color": C["border"]}),
                                html.Span(
                                    "Open-Meteo x Kaggle",
                                    style={
                                        "color": C["accent"],
                                        "fontSize": "0.7rem",
                                        "fontWeight": "600",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={
                        "maxWidth": "1400px",
                        "margin": "0 auto",
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                    },
                )
            ],
            style={
                "background": "#0B1120",
                "borderBottom": f"1px solid {C['border']}",
                "padding": "12px 24px",
                "position": "sticky",
                "top": "0",
                "zIndex": "100",
            },
        ),
        # TABS
        html.Div(
            [
                dcc.Tabs(
                    id="tabs",
                    value="eda",
                    style={"borderBottom": f"1px solid {C['border']}"},
                    children=[
                        dcc.Tab(
                            label="📊  EDA", value="eda", style=_T, selected_style=_TS
                        ),
                        dcc.Tab(
                            label="🤖  Model",
                            value="model",
                            style=_T,
                            selected_style=_TS,
                        ),
                        dcc.Tab(
                            label="🔍  SHAP", value="shap", style=_T, selected_style=_TS
                        ),
                        dcc.Tab(
                            label="🔮  Prediction",
                            value="predict",
                            style=_T,
                            selected_style=_TS,
                        ),
                    ],
                ),
                html.Div(
                    id="tb",
                    style={
                        "maxWidth": "1400px",
                        "margin": "0 auto",
                        "padding": "20px 16px",
                    },
                ),
            ],
            style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px 16px 0"},
        ),
        # FOOTER
        html.Div(
            "Nhập môn Khoa học Dữ liệu - Khoa Toán - Tin học, HCMUS  |  "
            "24KDL - Nhóm 6  |  Nguồn: Open-Meteo x Kaggle",
            style={
                "textAlign": "center",
                "color": C["sub"],
                "fontSize": "0.68rem",
                "padding": "16px",
                "borderTop": f"1px solid {C['border']}",
                "marginTop": "20px",
            },
        ),
    ],
    style={
        "background": C["bg"],
        "minHeight": "100vh",
        "fontFamily": "'Inter','Segoe UI',sans-serif",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB ROUTER
# ══════════════════════════════════════════════════════════════════════════════


@app.callback(
    Output("tb", "children"),
    Output("hdr-t", "children"),
    Input("tabs", "value"),
    Input("iv", "n_intervals"),
)
def route(tab, _):
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime(
        "Cập nhật: %H:%M (UTC+7)  %d/%m/%Y"
    )
    if tab == "eda":
        return eda_layout(), now
    elif tab == "model":
        return model_layout(), now
    elif tab == "shap":
        return shap_layout(), now
    elif tab == "predict":
        return predict_layout(), now
    return html.Div(), now


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - EDA
# ══════════════════════════════════════════════════════════════════════════════


def eda_layout():
    df = load_hist()
    aqi = df["us_aqi"]

    # 6 KPI Cards
    kpi_row = html.Div(
        [
            kpi(
                "activity",
                "AQI Trung bình",
                f"{aqi.mean():.1f}",
                aqi_clr(aqi.mean()),
                AQI_CATEGORY(aqi.mean()),
            ),
            kpi(
                "arrow-up-circle",
                "AQI Cao nhất",
                f"{aqi.max():.0f}",
                C["red"],
                df.loc[aqi.idxmax(), "date"].strftime("%d/%m/%Y"),
            ),
            kpi(
                "arrow-down-circle",
                "AQI Thấp nhất",
                f"{aqi.min():.0f}",
                C["green"],
                df.loc[aqi.idxmin(), "date"].strftime("%d/%m/%Y"),
            ),
            kpi(
                "exclamation-triangle",
                "Ngày AQI > 100",
                f"{(aqi > 100).sum()}",
                C["yellow"],
                f"{(aqi > 100).mean() * 100:.1f}% số ngày",
            ),
            kpi(
                "sun",
                "Ngày AQI ≤ 50",
                f"{(aqi <= 50).sum()}",
                C["green"],
                f"{(aqi <= 50).mean() * 100:.1f}% số ngày",
            ),
            kpi(
                "calendar-range",
                "Tổng số ngày",
                f"{len(df):,}",
                C["acc2"],
                f"{df['date'].min().strftime('%d/%m/%Y')} → "
                f"{df['date'].max().strftime('%d/%m/%Y')}",
            ),
        ],
        style={
            "display": "flex",
            "gap": "10px",
            "marginBottom": "12px",
            "flexWrap": "wrap",
        },
    )

    scale = card(
        [
            html.Div(
                [
                    html.Span(
                        "THANG AQI (US EPA): ",
                        style={
                            "fontSize": "0.7rem",
                            "color": C["sub"],
                            "fontWeight": "600",
                            "marginRight": "8px",
                            "whiteSpace": "nowrap",
                        },
                    ),
                    *[
                        html.Span(
                            lbl,
                            style={
                                "fontSize": "0.68rem",
                                "fontWeight": "600",
                                "padding": "2px 9px",
                                "borderRadius": "12px",
                                "background": bg,
                                "color": "#000" if i < 2 else "#fff",
                                "marginRight": "4px",
                            },
                        )
                        for i, (lbl, bg) in enumerate(
                            [
                                ("0-50 Tốt", "#00E400"),
                                ("51-100 Trung bình", "#FFFF00"),
                                ("101-150 Nhạy cảm", "#FF7E00"),
                                ("151-200 Có hại", "#FF0000"),
                                ("201-300 Rất có hại", "#8F3F97"),
                                ("301+ Nguy hiểm", "#7E0023"),
                            ]
                        )
                    ],
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "gap": "4px",
                },
            )
        ],
        ex={"padding": "10px 16px", "marginBottom": "12px"},
    )

    controls = card(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Khoảng thời gian",
                                style={
                                    "color": C["sub"],
                                    "fontSize": "0.72rem",
                                    "marginBottom": "4px",
                                    "display": "block",
                                },
                            ),
                            dcc.DatePickerRange(
                                id="eda-dr",
                                min_date_allowed=df["date"].min(),
                                max_date_allowed=df["date"].max(),
                                start_date=df["date"].min(),
                                end_date=df["date"].max(),
                                display_format="DD/MM/YYYY",
                            ),
                        ],
                        style={"flex": "2", "minWidth": "260px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Chỉ số chính",
                                style={
                                    "color": C["sub"],
                                    "fontSize": "0.72rem",
                                    "marginBottom": "4px",
                                    "display": "block",
                                },
                            ),
                            dcc.Dropdown(
                                id="eda-pol",
                                clearable=False,
                                value="us_aqi",
                                options=[
                                    {"label": v, "value": k}
                                    for k, v in POLS.items()
                                    if k in df.columns
                                ],
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "155px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "So sánh năm",
                                style={
                                    "color": C["sub"],
                                    "fontSize": "0.72rem",
                                    "marginBottom": "4px",
                                    "display": "block",
                                },
                            ),
                            dcc.Dropdown(
                                id="eda-yrs",
                                multi=True,
                                value=sorted(df["year"].unique())[:4],
                                options=[
                                    {"label": str(y), "value": y}
                                    for y in sorted(df["year"].unique())
                                ],
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "175px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                id="eda-ma-lbl",
                                style={
                                    "color": C["sub"],
                                    "fontSize": "0.72rem",
                                    "marginBottom": "6px",
                                    "display": "block",
                                },
                            ),
                            dcc.Slider(
                                id="eda-ma",
                                min=1,
                                max=90,
                                step=1,
                                value=30,
                                marks={
                                    1: "1d",
                                    7: "7d",
                                    30: "30d",
                                    60: "60d",
                                    90: "90d",
                                },
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": False,
                                },
                            ),
                        ],
                        style={"flex": "1", "minWidth": "200px", "paddingTop": "4px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "14px",
                    "flexWrap": "wrap",
                    "alignItems": "flex-end",
                },
            )
        ],
        ex={"marginBottom": "12px"},
    )

    return html.Div(
        [
            kpi_row,
            scale,
            controls,
            # 2.1 Chuỗi thời gian
            card(
                [
                    sec("graph-up", "Chuỗi Thời Gian"),
                    gph("eda-ts", 320),
                ],
                ex={"marginBottom": "12px"},
            ),
            # 2.2 Histogram + Pie
            row(
                col(
                    card(
                        [
                            sec("bar-chart-fill", "Phân Phối Các Chỉ Số"),
                            gph("eda-hist", 290),
                        ]
                    )
                ),
                col(
                    card(
                        [
                            sec("pie-chart-fill", "Phân Bố Mức AQI"),
                            gph("eda-pie", 290),
                        ]
                    )
                ),
            ),
            # 2.3 Monthly + Day of week
            row(
                col(
                    card(
                        [
                            sec("calendar3", "Chỉ Số Theo Tháng"),
                            gph("eda-monthly", 290),
                        ]
                    ),
                    flex="2",
                ),
                col(
                    card(
                        [
                            sec("calendar-week", "AQI Theo Ngày Trong Tuần"),
                            gph("eda-dow", 290),
                        ]
                    )
                ),
            ),
            # 2.4 Boxplot tháng
            card(
                [
                    sec("box-seam", "Chỉ Số Theo Tháng - Biến Động Và Outliers"),
                    gph("eda-box", 330),
                ],
                ex={"marginBottom": "12px"},
            ),
            # 2.5 So sánh năm
            card(
                [
                    sec("bar-chart-line", "So Sánh Chỉ Số Từng Năm Theo Tháng"),
                    gph("eda-year", 300),
                ],
                ex={"marginBottom": "12px"},
            ),
            # 2.6 Heatmap
            card(
                [
                    sec("grid-3x3-gap", "AQI - Ngày Trong Tuần x Tháng"),
                    gph("eda-hm", 320),
                ],
                ex={"marginBottom": "12px"},
            ),
            # 2.7 Decomposition
            card(
                [
                    sec(
                        "graph-down-arrow",
                        "Time Series Decomposition",
                    ),
                    gph("eda-dc", 500),
                ],
                ex={"marginBottom": "12px"},
            ),
            # 2.8 Scatter PM2.5 vs AQI
            card(
                [
                    sec("scatter-chart", "PM2.5 - US AQI"),
                    gph("eda-sc", 300),
                ],
                ex={"marginBottom": "12px"},
            ),
            # 2.9 Correlation heatmap
            card(
                [
                    sec("diagram-3", "Ma Trận Tương Quan Pearson"),
                    gph("eda-corr", 380),
                ],
                ex={"marginBottom": "12px"},
            ),
            # 2.10 PM2.5 boxplot
            card(
                [
                    sec("box-arrow-in-down", "PM2.5 Theo Tháng"),
                    gph("eda-pm25", 300),
                ],
                ex={"marginBottom": "12px"},
            ),
            # Describe table
            card(
                [sec("table", "Bảng Thống Kê"), html.Div(id="eda-desc")],
                ex={"marginBottom": "12px"},
            ),
        ]
    )


@app.callback(
    Output("eda-ma-lbl", "children"),
    Input("eda-ma", "value"),
)
def update_ma_label(val):
    return f"Rolling Mean - {val} ngày"


@app.callback(
    [
        Output("eda-ts", "figure"),
        Output("eda-hist", "figure"),
        Output("eda-pie", "figure"),
        Output("eda-monthly", "figure"),
        Output("eda-dow", "figure"),
        Output("eda-box", "figure"),
        Output("eda-year", "figure"),
        Output("eda-hm", "figure"),
        Output("eda-dc", "figure"),
        Output("eda-sc", "figure"),
        Output("eda-corr", "figure"),
        Output("eda-pm25", "figure"),
        Output("eda-desc", "children"),
    ],
    [
        Input("eda-dr", "start_date"),
        Input("eda-dr", "end_date"),
        Input("eda-pol", "value"),
        Input("eda-yrs", "value"),
        Input("eda-ma", "value"),
        Input("iv", "n_intervals"),
    ],
)
def upd_eda(start, end, pol, years, ma, _):
    df = load_hist()
    lbl = POLS.get(pol, pol)
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    dff = df[mask].copy()

    ef = go.Figure()
    ef.update_layout(**PL)
    if dff.empty:
        return [ef] * 12 + [html.Div()]

    # ────────────────────────────────────────────────────────────────
    # TIMESERIES
    # ────────────────────────────────────────────────────────────────

    f1 = go.Figure()
    if pol == "us_aqi":
        for y0, y1, bg, nm in [
            (0, 50, "rgba(0,200,83,0.05)", "Good"),
            (50, 100, "rgba(255,214,0,0.05)", "Moderate"),
            (100, 150, "rgba(255,109,0,0.05)", "USG"),
            (150, 220, "rgba(213,0,0,0.05)", "Unhealthy"),
        ]:
            f1.add_hrect(
                y0=y0,
                y1=y1,
                fillcolor=bg,
                layer="below",
                line_width=0,
                annotation_text=nm,
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=bg.replace("0.05", "0.7"),
            )
    f1.add_trace(
        go.Scatter(
            x=dff["date"],
            y=dff[pol],
            mode="lines",
            name=lbl,
            line=dict(color=C["accent"], width=0.9),
            opacity=0.45,
            hovertemplate="%{x|%d/%m/%Y}<br>"
            + lbl
            + ": <b>%{y:.1f}</b><extra></extra>",
        )
    )
    f1.add_trace(
        go.Scatter(
            x=dff["date"],
            y=dff[pol].rolling(ma, center=True).mean(),
            mode="lines",
            name=f"MA {ma}d",
            line=dict(color=C["acc2"], width=2.5),
            hovertemplate="%{x|%d/%m/%Y}<br>MA"
            + str(ma)
            + "d: <b>%{y:.1f}</b><extra></extra>",
        )
    )
    if pol == "us_aqi":
        ix = dff["us_aqi"].idxmax()
        f1.add_trace(
            go.Scatter(
                x=[dff.loc[ix, "date"]],
                y=[dff.loc[ix, "us_aqi"]],
                mode="markers+text",
                name="Max",
                marker=dict(
                    color=C["red"], size=10, line=dict(color="white", width=1.5)
                ),
                text=[f" {dff.loc[ix, 'us_aqi']:.0f}"],
                textposition="top right",
                textfont=dict(color=C["red"], size=10),
                hovertemplate="Max AQI: <b>%{y:.0f}</b><extra></extra>",
            )
        )
    f1.update_layout(**PL)
    f1.update_layout(
        yaxis_title=lbl,
        showlegend=True,
        title=dict(
            text=f"{lbl} | {dff['date'].min().strftime('%d/%m/%Y')} → "
            f"{dff['date'].max().strftime('%d/%m/%Y')}",
            font=dict(size=11, color=C["sub"]),
            x=0,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # ────────────────────────────────────────────────────────────────
    # HISTOGRAM
    # ────────────────────────────────────────────────────────────────

    f2 = go.Figure()
    if pol == "us_aqi":
        for lo, hi, clr in [
            (0, 50, AQI_COLORS["Good"]),
            (50, 100, AQI_COLORS["Moderate"]),
            (100, 150, AQI_COLORS["Unhealthy for Sensitive Groups"]),
            (150, 250, AQI_COLORS["Unhealthy"]),
        ]:
            v = dff[pol][(dff[pol] >= lo) & (dff[pol] < hi)]
            if len(v):
                f2.add_trace(
                    go.Histogram(
                        x=v,
                        xbins=dict(start=lo, end=hi, size=8),
                        marker_color=clr,
                        opacity=0.85,
                        showlegend=False,
                        hovertemplate="AQI: %{x:.0f}<br>Số ngày: %{y}<extra></extra>",
                    )
                )
    else:
        f2.add_trace(
            go.Histogram(
                x=dff[pol],
                nbinsx=40,
                marker_color=C["accent"],
                opacity=0.85,
                hovertemplate=lbl + ": %{x:.1f}<br>Số ngày: %{y}<extra></extra>",
            )
        )
    for v, clr2, lbl2, pos in [
        (dff[pol].mean(), C["red"], f"Mean={dff[pol].mean():.1f}", "top right"),
        (dff[pol].median(), C["yellow"], f"Median={dff[pol].median():.1f}", "top left"),
    ]:
        f2.add_vline(
            x=v,
            line_dash="dash",
            line_color=clr2,
            line_width=1.5,
            annotation_text=lbl2,
            annotation_position=pos,
            annotation_yshift=15,
            annotation_font_color=clr2,
            annotation_font_size=9,
        )
    f2.update_layout(
        **PL,
        yaxis_title="Số ngày",
        xaxis_title=lbl,
        showlegend=False,
        title=dict(
            text=f"Skewness = {dff[pol].skew():.3f} | Std = {dff[pol].std():.2f}",
            font=dict(size=11, color=C["sub"]),
            x=0,
        ),
    )

    # ────────────────────────────────────────────────────────────────
    # PIE/DONUT
    # ────────────────────────────────────────────────────────────────

    cc = dff["aqi_category"].value_counts()
    pc = [c for c in AQI_LABELS if c in cc.index]
    pull = [
        0.05
        if c in ["Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy"]
        else 0
        for c in pc
    ]
    f3 = go.Figure(
        go.Pie(
            labels=pc,
            values=[cc[c] for c in pc],
            marker_colors=[AQI_COLORS[c] for c in pc],
            hole=0.54,
            textinfo="percent",
            textfont_size=11,
            pull=pull,
            hovertemplate="<b>%{label}</b><br>%{value} ngày (%{percent})<extra></extra>",
        )
    )
    f3.update_layout(
        **{
            **PL,
            "showlegend": True,
            "legend": dict(
                orientation="v",
                font=dict(size=9),
                bgcolor="rgba(0,0,0,0)",
                x=1.02,
                y=0.5,
            ),
            "margin": dict(l=10, r=120, t=10, b=10),
            "annotations": [
                dict(
                    text=f"<b>{len(dff)}</b><br><sub>ngày</sub>",
                    x=0.47,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=18, color=C["text"]),
                )
            ],
        }
    )

    # ────────────────────────────────────────────────────────────────
    # MONTHLY BAR
    # ────────────────────────────────────────────────────────────────

    mo = dff.groupby("month")[pol].agg(["mean", "std", "max", "min"]).reset_index()
    mo["mn"] = mo["month"].apply(lambda x: MONTHS[x - 1])
    bc = (
        [
            C["green"]
            if v <= 50
            else C["yellow"]
            if v <= 100
            else C["orange"]
            if v <= 150
            else C["red"]
            for v in mo["mean"]
        ]
        if pol == "us_aqi"
        else [C["accent"]] * len(mo)
    )
    f4 = go.Figure()
    f4.add_trace(
        go.Bar(
            x=mo["mn"],
            y=mo["mean"].round(1),
            marker_color=bc,
            name="Trung bình",
            error_y=dict(
                type="data",
                array=mo["std"].round(1),
                color=C["border"],
                thickness=1.5,
                width=4,
            ),
            text=mo["mean"].round(1),
            textposition="outside",
            textfont=dict(size=9, color=C["text"]),
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.1f}<br>"
            "Max: %{customdata[0]:.0f}<br>Min: %{customdata[1]:.0f}<extra></extra>",
            customdata=mo[["max", "min"]].values,
        )
    )
    f4.add_trace(
        go.Scatter(
            x=mo["mn"],
            y=mo["mean"],
            mode="lines",
            line=dict(color="white", width=1.5, dash="dot"),
            showlegend=False,
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        )
    )
    f4.update_layout(
        **PL,
        yaxis_title=lbl,
        bargap=0.22,
        title=dict(
            text="Thanh sai số = ±1 Std | Mùa khô T11-T4 và Mùa mưa T5-T10",
            font=dict(size=11, color=C["sub"]),
            x=0,
        ),
    )

    # ────────────────────────────────────────────────────────────────
    # DAY OF WEEK
    # ────────────────────────────────────────────────────────────────

    dow = dff.groupby("dow")[pol].mean().reset_index()
    dow["dn"] = dow["dow"].apply(lambda x: DAYS[x] if x < 7 else str(x))
    bc2 = [C["acc2"] if d >= 5 else C["accent"] for d in dow["dow"]]
    f5 = go.Figure(
        go.Bar(
            x=dow["dn"],
            y=dow[pol].round(2),
            marker_color=bc2,
            text=dow[pol].round(1),
            textposition="outside",
            textfont=dict(size=9, color=C["text"]),
            hovertemplate="<b>%{x}</b><br>AQI: %{y:.2f}<extra></extra>",
        )
    )
    f5.update_layout(
        **PL,
        yaxis_title=lbl,
        bargap=0.3,
        title=dict(
            text="Màu tím = Cuối tuần (T7, CN)", font=dict(size=11, color=C["sub"]), x=0
        ),
    )

    # ────────────────────────────────────────────────────────────────
    # BOXPLOT THEO THÁNG
    # ────────────────────────────────────────────────────────────────

    f6 = go.Figure()
    for m in sorted(dff["month"].unique()):
        v2 = dff[dff["month"] == m][pol].dropna()
        mn2 = MONTHS[m - 1]
        med = v2.median()
        clr3 = (
            C["green"]
            if med <= 50
            else C["yellow"]
            if med <= 100
            else C["orange"]
            if med <= 150
            else C["red"]
        )
        f6.add_trace(
            go.Box(
                y=v2,
                name=mn2,
                marker_color=clr3,
                line_color=clr3,
                opacity=0.85,
                boxmean=True,
                showlegend=False,
                hovertemplate=f"<b>{mn2}</b><br>"
                f"Giá trị: <b>%{{y:.1f}}</b><extra></extra>",
            )
        )
    f6.update_layout(
        **PL,
        yaxis_title=lbl,
        showlegend=False,
        title=dict(
            text="Đường nét liền = Median | Đường nét đứt = Mean | Chấm tròn = Outlier",
            font=dict(size=11, color=C["sub"]),
            x=0,
        ),
    )

    # ────────────────────────────────────────────────────────────────
    # YEAR COMPARE
    # ────────────────────────────────────────────────────────────────

    CLR = [C["accent"], C["acc2"], "#F59E0B", "#10B981", "#F43F5E"]
    yrl = sorted(df["year"].unique().tolist())
    f7 = go.Figure()
    for i, yr in enumerate(yrl):
        dfy = df[df["year"] == yr].groupby("month")[pol].mean().reset_index()
        dfy["mn"] = dfy["month"].apply(lambda x: MONTHS[x - 1])
        f7.add_trace(
            go.Scatter(
                x=dfy["mn"],
                y=dfy[pol].round(1),
                mode="lines+markers",
                name=str(yr),
                line=dict(color=CLR[i % len(CLR)], width=2.5),
                marker=dict(size=7),
                hovertemplate=f"{yr} - %{{x}}: <b>%{{y:.1f}}</b><extra></extra>",
            )
        )
    f7.update_layout(**PL)
    f7.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=MONTHS),
        yaxis_title=lbl,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text="Mỗi đường = 1 năm | Quan sát sự thay đổi pattern qua các năm",
            font=dict(size=11, color=C["sub"]),
            x=0,
        ),
    )

    # ────────────────────────────────────────────────────────────────
    # HEATMAP NGÀY TRONG TUẦN × THÁNG
    # ────────────────────────────────────────────────────────────────

    piv = dff.pivot_table(values="us_aqi", index="dow", columns="month", aggfunc="mean")
    piv.index = [DAYS[i] if i < 7 else str(i) for i in piv.index]
    piv.columns = [MONTHS[m - 1] for m in piv.columns]
    z = np.round(piv.values, 1)
    f8 = go.Figure(
        go.Heatmap(
            z=z,
            x=piv.columns.tolist(),
            y=piv.index.tolist(),
            colorscale=[
                [0, "#00C853"],
                [0.32, "#FFD600"],
                [0.64, "#FF6D00"],
                [0.82, "#D50000"],
                [1, "#6A1B9A"],
            ],
            zmin=40,
            zmax=140,
            text=np.where(np.isnan(z), "", np.round(z, 0).astype(int).astype(str)),
            texttemplate="%{text}",
            textfont_size=9,
            hovertemplate="<b>%{y} - %{x}</b><br>AQI: %{z:.1f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(size=9),
                tickvals=[40, 50, 75, 100, 125, 140],
                ticktext=["40", "50", "75", "100", "125", "140"],
            ),
        )
    )
    f8.update_layout(
        **{
            **PL,
            "margin": dict(l=44, r=20, t=10, b=44),
            "xaxis": dict(title="Tháng", gridcolor="rgba(0,0,0,0)"),
            "yaxis": dict(title="Ngày Trong Tuần", gridcolor="rgba(0,0,0,0)"),
        }
    )

    # ────────────────────────────────────────────────────────────────
    # DECOMPOSITION
    # ────────────────────────────────────────────────────────────────

    try:
        ts_full = df.set_index("date")["us_aqi"].asfreq("D").interpolate(method="time")
        res = seasonal_decompose(ts_full, model="additive", period=365)
        f9 = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Chuỗi gốc (Observed)",
                "Xu hướng (Trend) - Trung bình dài hạn",
                "Mùa vụ (Seasonal) - Chu kỳ 365 ngày",
                "Phần dư (Residual) - Nhiễu ngẫu nhiên",
            ],
            vertical_spacing=0.07,
        )
        for (s, clr4, lw4, op4), ri in zip(
            [
                (res.observed, C["accent"], 0.9, 0.5),
                (res.trend, C["acc2"], 2.0, 1.0),
                (res.seasonal, C["green"], 0.9, 0.75),
                (res.resid, C["yellow"], 0.9, 0.65),
            ],
            range(1, 5),
        ):
            f9.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    line=dict(color=clr4, width=lw4),
                    opacity=op4,
                    showlegend=False,
                    hovertemplate="%{x|%d/%m/%Y}: %{y:.2f}<extra></extra>",
                ),
                row=ri,
                col=1,
            )
        f9.add_hline(
            y=0, row=4, col=1, line_dash="dash", line_color=C["border"], line_width=1
        )
        f9.update_layout(
            **{**PL, "height": 500, "margin": dict(l=44, r=20, t=44, b=44)}
        )
        for ri in range(1, 5):
            f9.update_yaxes(gridcolor=C["border"], row=ri, col=1)
            f9.update_xaxes(gridcolor=C["border"], row=ri, col=1)
    except Exception as e:
        f9 = go.Figure()
        f9.add_annotation(
            text=f"Lỗi Decomposition: {e}",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=C["sub"], size=13),
        )
        f9.update_layout(**PL)

    # ────────────────────────────────────────────────────────────────
    # SCATTER PM2.5 - AQI
    # ────────────────────────────────────────────────────────────────

    f10 = go.Figure()
    if "pm2_5" in dff.columns:
        for cat_n, cat_c in AQI_COLORS.items():
            mk = dff["aqi_category"] == cat_n
            if mk.sum() == 0:
                continue
            f10.add_trace(
                go.Scatter(
                    x=dff.loc[mk, "pm2_5"],
                    y=dff.loc[mk, "us_aqi"],
                    mode="markers",
                    name=cat_n,
                    marker=dict(color=cat_c, size=5, opacity=0.6),
                    hovertemplate="PM2.5: %{x:.1f}<br>AQI: %{y:.0f}<extra></extra>",
                )
            )
        vld = dff[["pm2_5", "us_aqi"]].dropna()
        if len(vld) > 2:
            zf = np.polyfit(vld["pm2_5"], vld["us_aqi"], 1)
            xl = np.linspace(vld["pm2_5"].min(), vld["pm2_5"].max(), 100)
            r = vld["pm2_5"].corr(vld["us_aqi"])
            f10.add_trace(
                go.Scatter(
                    x=xl,
                    y=np.polyval(zf, xl),
                    mode="lines",
                    name=f"Trend  r = {r:.3f}",
                    line=dict(color="white", width=1.5, dash="dash"),
                )
            )
    f10.update_layout(**PL)
    f10.update_layout(
        xaxis_title="PM2.5 (µg/m³)",
        yaxis_title="US AQI",
        showlegend=True,
        legend=dict(
            orientation="v", font=dict(size=9), bgcolor="rgba(0,0,0,0)", x=1.01, y=1
        ),
        title=dict(
            text="PM2.5 và US AQI | r ≈ 0.92 - Tương quan mạnh nhất",
            font=dict(size=11, color=C["sub"]),
            x=0,
        ),
    )

    # ────────────────────────────────────────────────────────────────
    # CORRELATION HEATMAP
    # ────────────────────────────────────────────────────────────────

    CC = [
        c
        for c in [
            "us_aqi",
            "pm2_5",
            "pm10",
            "ozone",
            "nitrogen_dioxide",
            "carbon_monoxide",
            "sulphur_dioxide",
        ]
        if c in dff.columns
    ]
    CL2 = {
        "us_aqi": "US AQI",
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "ozone": "O₃",
        "nitrogen_dioxide": "NO₂",
        "carbon_monoxide": "CO",
        "sulphur_dioxide": "SO₂",
    }
    cm = dff[CC].corr().round(3)
    tri = np.triu(np.ones_like(cm, dtype=bool), k=1)
    cp = cm.mask(tri)
    f11 = go.Figure(
        go.Heatmap(
            z=cp.values,
            x=[CL2.get(c, c) for c in cm.columns],
            y=[CL2.get(c, c) for c in cm.index],
            colorscale=[[0, "#F43F5E"], [0.5, "#1E293B"], [1, "#38BDF8"]],
            zmin=-1,
            zmax=1,
            text=cp.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(size=9),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0", "0.5", "1.0"],
            ),
        )
    )
    f11.update_layout(
        **{
            **PL,
            "margin": dict(l=60, r=20, t=20, b=60),
            "xaxis": dict(
                side="bottom", tickfont=dict(size=12), gridcolor="rgba(0,0,0,0)"
            ),
            "yaxis": dict(
                tickfont=dict(size=12), gridcolor="rgba(0,0,0,0)", autorange="reversed"
            ),
        }
    )

    # ────────────────────────────────────────────────────────────────
    # BOXPLOT PM2.5 THEO THÁNG
    # ────────────────────────────────────────────────────────────────

    f12 = go.Figure()
    if "pm2_5" in dff.columns:
        for m in sorted(dff["month"].unique()):
            v3 = dff[dff["month"] == m]["pm2_5"].dropna()
            mn3 = MONTHS[m - 1]
            f12.add_trace(
                go.Box(
                    y=v3,
                    name=mn3,
                    marker_color=C["acc2"],
                    line_color=C["acc2"],
                    opacity=0.8,
                    boxmean=True,
                    showlegend=False,
                    hovertemplate=f"<b>{mn3}</b><br>"
                    f"Giá trị: <b>%{{y:.1f}}</b><extra></extra>",
                )
            )
        f12.add_hline(
            y=12,
            line_dash="dash",
            line_color="#00C853",
            line_width=1.5,
            annotation_text="12 µg/m³",
            annotation_font_color="#00C853",
            annotation_font_size=9,
        )
        f12.add_hline(
            y=35.4,
            line_dash="dash",
            line_color="#FFD600",
            line_width=1.5,
            annotation_text="35.4 µg/m³",
            annotation_font_color="#FFD600",
            annotation_font_size=9,
        )
    f12.update_layout(
        **PL,
        yaxis_title="PM2.5 (µg/m³)",
        showlegend=False,
        title=dict(
            text="Good = 12 µg/m³ | Moderate = 35.4 µg/m³ (US EPA)",
            font=dict(size=11, color=C["sub"]),
            x=0,
        ),
    )

    # ────────────────────────────────────────────────────────────────
    # DESCRIBE TABLE
    # ────────────────────────────────────────────────────────────────

    dc = [
        c
        for c in [
            "us_aqi",
            "pm2_5",
            "pm10",
            "ozone",
            "nitrogen_dioxide",
            "carbon_monoxide",
        ]
        if c in dff.columns
    ]
    dsc = dff[dc].describe().round(2).reset_index()
    rename_dict = {"index": "Thống kê"}
    rename_dict.update(POLS)
    dsc.rename(columns=rename_dict, inplace=True)
    dsc["Thống kê"] = dsc["Thống kê"].replace(
        {
            "count": "Số mẫu",
            "mean": "Trung bình",
            "std": "Độ lệch chuẩn",
            "min": "Giá trị nhỏ nhất",
            "25%": "Q1 (25%)",
            "50%": "Trung vị",
            "75%": "Q3 (75%)",
            "max": "Giá trị lớn nhất",
        }
    )

    tbl = dash_table.DataTable(
        data=dsc.to_dict("records"),
        columns=[{"name": c, "id": c} for c in dsc.columns],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": C["border"],
            "color": C["text"],
            "fontWeight": "600",
            "fontSize": "0.78rem",
            "border": "none",
        },
        style_cell={
            "backgroundColor": C["card"],
            "color": C["text"],
            "fontSize": "0.78rem",
            "padding": "8px 12px",
            "border": f"1px solid {C['border']}",
            "textAlign": "center",
        },
        style_data_conditional=[
            {
                "if": {"column_id": "Thống kê"},
                "fontWeight": "600",
                "color": C["accent"],
                "textAlign": "left",
            }
        ],
    )

    return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, tbl


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - MODEL
# ══════════════════════════════════════════════════════════════════════════════


def model_layout():
    if not os.path.exists(MDL_PATH):
        return empty_st(
            "🤖",
            "Model chưa được train",
            "Chạy notebook 03 trước.",
            "jupyter notebook notebooks/03_model_regression.ipynb",
        )

    model, X_test, y_test, y_pred, meta = load_model()
    if model is None:
        return empty_st(
            "⚠️",
            "Không thể load model",
            "Kiểm tra best_regressor.pkl, X_test.csv, y_test.csv.",
        )

    from sklearn.metrics import (
        mean_absolute_error as mae_fn,
        mean_squared_error as mse_fn,
        r2_score,
    )

    y_true = y_test["target_reg_tomorrow"].values
    res = y_true - y_pred
    mae = mae_fn(y_true, y_pred)
    rmse = np.sqrt(mse_fn(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs(res / (y_true + 1e-9))) * 100
    mname = meta.get("best_model", "Best Model")
    mc = C["green"] if r2 > 0.8 else C["yellow"] if r2 > 0.65 else C["orange"]

    kpi_row = html.Div(
        [
            kpi(
                "bullseye",
                "MAE",
                f"{mae:.2f}",
                C["accent"],
                "Sai số tuyệt đối trung bình",
            ),
            kpi(
                "graph-down-arrow",
                "RMSE",
                f"{rmse:.2f}",
                C["acc2"],
                "Căn bậc hai của sai số bình phương trung bình",
            ),
            kpi("bar-chart-fill", "R²", f"{r2:.4f}", mc, "Hệ số xác định"),
            kpi(
                "percent",
                "MAPE",
                f"{mape:.2f}%",
                C["yellow"],
                "Sai số phần trăm tuyệt đối trung bình",
            ),
        ],
        style={
            "display": "flex",
            "gap": "10px",
            "marginBottom": "12px",
            "flexWrap": "wrap",
        },
    )

    info = card(
        [
            html.Div(
                [
                    html.Span("🏆  ", style={"fontSize": "1rem"}),
                    html.Span(
                        f"Best Model: {mname}",
                        style={
                            "color": C["accent"],
                            "fontWeight": "700",
                            "fontSize": "0.92rem",
                        },
                    ),
                ],
                style={"marginBottom": "6px"},
            ),
            html.Div(
                f"Train: {meta.get('train_period', {}).get('start', '?')} → "
                f"{meta.get('train_period', {}).get('end', '?')}",
                style={"color": C["sub"], "fontSize": "0.78rem"},
            ),
            html.Div(
                f"Tạo lúc: {meta.get('created_at', '?')}",
                style={"color": C["sub"], "fontSize": "0.78rem"},
            ),
        ],
        ex={"marginBottom": "12px"},
    )

    # Actual vs Predicted
    fa = go.Figure()
    fa.add_trace(
        go.Scatter(
            x=X_test.index,
            y=y_true,
            mode="lines",
            name="Thực tế",
            line=dict(color=C["accent"], width=1.2),
            opacity=0.7,
            hovertemplate="%{x|%d/%m/%Y}<br>Thực tế: <b>%{y:.0f}</b><extra></extra>",
        )
    )
    fa.add_trace(
        go.Scatter(
            x=X_test.index,
            y=y_pred,
            mode="lines",
            name="Dự đoán",
            line=dict(color=C["orange"], width=1.2, dash="dash"),
            hovertemplate="%{x|%d/%m/%Y}<br>Dự đoán: <b>%{y:.0f}</b><extra></extra>",
        )
    )
    fa.update_layout(**PL)
    fa.update_layout(
        yaxis_title="US AQI",
        showlegend=True,
        title=dict(
            text=f"Actual vs Predicted - {mname}",
            font=dict(size=13, color=C["accent"]),
            x=0,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Scatter
    fs = go.Figure()
    fs.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color=C["acc2"], size=5, opacity=0.6),
            hovertemplate="Thực tế: %{x:.0f}<br>Dự đoán: %{y:.0f}<extra></extra>",
        )
    )
    lim = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
    fs.add_trace(
        go.Scatter(
            x=lim,
            y=lim,
            mode="lines",
            name="y = x",
            line=dict(color=C["red"], dash="dash", width=1.5),
        )
    )
    fs.update_layout(
        **PL,
        xaxis_title="Thực tế",
        yaxis_title="Dự đoán",
        title=dict(
            text=f"Scatter Plot (R² = {r2:.4f})",
            font=dict(size=13, color=C["accent"]),
            x=0,
        ),
        showlegend=True,
    )

    # Residuals
    fr = go.Figure(
        go.Histogram(
            x=res,
            nbinsx=40,
            marker_color=C["acc2"],
            opacity=0.85,
            hovertemplate="Residual: %{x:.1f}<br>Số ngày: %{y}<extra></extra>",
        )
    )
    fr.add_vline(x=0, line_dash="dash", line_color=C["red"], line_width=1.5)
    fr.add_vline(
        x=res.mean(),
        line_dash="dot",
        line_color=C["yellow"],
        annotation_text=f"Mean={res.mean():.2f}",
        annotation_font_color=C["yellow"],
    )
    fr.update_layout(
        **PL,
        xaxis_title="Residual",
        yaxis_title="Số ngày",
        title=dict(
            text="Phân phối Residuals", font=dict(size=13, color=C["accent"]), x=0
        ),
    )

    # MAE theo tháng
    dfev = pd.DataFrame({"a": y_true, "p": y_pred, "m": X_test.index.month})
    mm = dfev.groupby("m").apply(lambda x: mae_fn(x["a"], x["p"])).reset_index()
    mm.columns = ["month", "MAE"]
    mm["mn"] = mm["month"].apply(lambda x: MONTHS[x - 1])
    bc3 = [C["red"] if v == mm["MAE"].max() else C["accent"] for v in mm["MAE"]]
    fm = go.Figure(
        go.Bar(
            x=mm["mn"],
            y=mm["MAE"].round(2),
            marker_color=bc3,
            text=mm["MAE"].round(1),
            textposition="outside",
            textfont=dict(size=9, color=C["text"]),
            hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f}<extra></extra>",
        )
    )
    fm.update_layout(
        **PL,
        yaxis_title="MAE",
        bargap=0.25,
        title=dict(
            text="MAE theo Tháng - Tháng nào dự đoán sai nhiều nhất?",
            font=dict(size=13, color=C["accent"]),
            x=0,
        ),
    )

    # Feature importance
    fi_block = html.Div()
    if hasattr(model, "coef_") and X_test is not None:
        fi = (
            pd.Series(np.abs(model.coef_.flatten()), index=X_test.columns)
            .sort_values(ascending=False)
            .head(20)
        )
        fif = go.Figure(
            go.Bar(
                x=fi.values[::-1],
                y=fi.index[::-1],
                orientation="h",
                marker_color=C["green"],
                opacity=0.85,
                hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}<extra></extra>",
            )
        )
        fif.update_layout(
            **{
                **PL,
                "height": 440,
                "margin": dict(l=210, r=20, t=48, b=44),
                "title": dict(
                    text="Feature Importance (Split) - Top 20",
                    font=dict(size=13, color=C["accent"]),
                    x=0,
                ),
                "xaxis_title": "Score",
            }
        )
        fi_block = card(
            [
                sec("list-ol", "Feature Importance"),
                dcc.Graph(
                    figure=fif,
                    config={"displayModeBar": False},
                    style={"height": "440px"},
                ),
            ],
            ex={"marginBottom": "12px"},
        )

    return html.Div(
        [
            kpi_row,
            info,
            card(
                [
                    sec("graph-up-arrow", "Actual - Predicted Theo Thời Gian"),
                    dcc.Graph(
                        figure=fa,
                        config={"displayModeBar": False},
                        style={"height": "300px"},
                    ),
                ],
                ex={"marginBottom": "12px"},
            ),
            row(
                col(
                    card(
                        [
                            sec("scatter-chart", "Scatter Plot"),
                            dcc.Graph(
                                figure=fs,
                                config={"displayModeBar": False},
                                style={"height": "280px"},
                            ),
                        ]
                    )
                ),
                col(
                    card(
                        [
                            sec("bar-chart", "Phân Phối Residuals"),
                            dcc.Graph(
                                figure=fr,
                                config={"displayModeBar": False},
                                style={"height": "280px"},
                            ),
                        ]
                    )
                ),
            ),
            card(
                [
                    sec("calendar3", "MAE Theo Tháng"),
                    dcc.Graph(
                        figure=fm,
                        config={"displayModeBar": False},
                        style={"height": "280px"},
                    ),
                ],
                ex={"marginBottom": "12px"},
            ),
            fi_block,
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - SHAP
# ══════════════════════════════════════════════════════════════════════════════


def shap_layout():
    if not os.path.exists(MDL_PATH):
        return empty_st(
            "🔍",
            "Model chưa được train",
            "Chạy notebook 03 trước.",
            "jupyter notebook notebooks/03_model_regression.ipynb",
        )
    try:
        model = joblib.load(MDL_PATH)
        X_test = pd.read_csv(XTEST, index_col=0, parse_dates=True)
        exp = shap.Explainer(model, X_test)
        sv = exp(X_test)
        vals = sv.values
        base = float(
            exp.expected_value
            if not hasattr(exp.expected_value, "__len__")
            else exp.expected_value[0]
        )
        mabs = np.abs(vals).mean(axis=0)
        fi_s = pd.Series(mabs, index=X_test.columns).sort_values(ascending=False)

        # Summary bar
        t20 = fi_s.head(20)
        fbar = go.Figure(
            go.Bar(
                x=t20.values[::-1],
                y=t20.index[::-1],
                orientation="h",
                marker=dict(
                    color=t20.values[::-1],
                    colorscale=[[0, C["acc2"]], [1, C["accent"]]],
                    showscale=False,
                ),
                opacity=0.9,
                hovertemplate="<b>%{y}</b><br>Mean|SHAP|: %{x:.4f}<extra></extra>",
            )
        )
        fbar.update_layout(
            **{
                **PL,
                "height": 480,
                "margin": dict(l=210, r=20, t=48, b=44),
                "title": dict(
                    text="SHAP Summary Bar - Mean |SHAP Value| Top 20",
                    font=dict(size=13, color=C["accent"]),
                    x=0,
                ),
                "xaxis_title": "Mean |SHAP Value|",
            }
        )

        # Beeswarm
        t15 = fi_s.head(15).index[::-1].tolist()
        fbee = go.Figure()
        np.random.seed(42)
        for i, feat in enumerate(t15):
            fi2 = list(X_test.columns).index(feat)
            sc = vals[:, fi2]
            fv = X_test[feat].values
            fn = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-9)
            cols2 = [
                f"rgba({int(255 * v)},{int(80 * (1 - v))},{int(220 * (1 - v))},0.65)"
                for v in fn
            ]
            jit = np.random.uniform(-0.2, 0.2, len(sc))
            fbee.add_trace(
                go.Scatter(
                    x=sc,
                    y=[i + jit[j] for j in range(len(sc))],
                    mode="markers",
                    marker=dict(size=4, color=cols2),
                    showlegend=False,
                    hovertemplate=f"<b>{feat}</b><br>SHAP:%{{x:.4f}}<br>"
                    f"Value:%{{text}}<extra></extra>",
                    text=[f"{v:.2f}" for v in fv],
                )
            )
        fbee.add_vline(x=0, line_dash="dash", line_color=C["border"], line_width=1)
        fbee.update_layout(
            **{
                **PL,
                "height": 480,
                "title": dict(
                    text="SHAP Beeswarm - Đỏ = giá trị cao, Xanh = giá trị thấp",
                    font=dict(size=13, color=C["accent"]),
                    x=0,
                ),
                "xaxis_title": "SHAP Value",
                "yaxis": dict(
                    tickmode="array",
                    tickvals=list(range(len(t15))),
                    ticktext=t15,
                    gridcolor=C["border"],
                ),
            }
        )

        # Waterfall
        ix = int(np.argmax(vals.sum(axis=1)))
        svd = vals[ix]
        ti = np.argsort(np.abs(svd))[::-1][:12]
        fnam = [X_test.columns[i] for i in ti]
        fval = svd[ti]
        pv = base + svd.sum()
        dl = (
            X_test.index[ix].strftime("%d/%m/%Y")
            if hasattr(X_test.index[ix], "strftime")
            else str(X_test.index[ix])
        )
        wc = [C["red"] if v > 0 else C["accent"] for v in fval]
        fwf = go.Figure(
            go.Bar(
                x=fval[::-1],
                y=fnam[::-1],
                orientation="h",
                marker_color=wc[::-1],
                opacity=0.85,
                hovertemplate="<b>%{y}</b><br>SHAP:%{x:+.4f}<extra></extra>",
            )
        )
        fwf.add_vline(x=0, line_dash="dash", line_color=C["border"], line_width=1)
        fwf.update_layout(
            **{
                **PL,
                "height": 420,
                "margin": dict(l=210, r=20, t=52, b=44),
                "title": dict(
                    text=f"Waterfall - Ngày {dl}  (Base={base:.1f} → Dự đoán={pv:.0f})",
                    font=dict(size=13, color=C["accent"]),
                    x=0,
                ),
                "xaxis_title": "SHAP Value",
            }
        )

        # Dependence plot t-1 × is_dry_season
        dep_block = html.Div()
        if "t-1" in X_test.columns:
            si2 = vals[:, list(X_test.columns).index("t-1")]
            fv2 = X_test["t-1"].values
            dry = (
                X_test["is_dry_season"].values
                if "is_dry_season" in X_test.columns
                else np.zeros(len(fv2))
            )
            cld = [C["orange"] if d else C["accent"] for d in dry]
            fdep = go.Figure(
                go.Scatter(
                    x=fv2,
                    y=si2,
                    mode="markers",
                    marker=dict(size=5, color=cld, opacity=0.6),
                    hovertemplate="t-1: %{x:.1f}<br>SHAP: %{y:.4f}<extra></extra>",
                )
            )
            fdep.add_hline(y=0, line_dash="dash", line_color=C["border"], line_width=1)
            fdep.update_layout(
                **{
                    **PL,
                    "height": 300,
                    "title": dict(
                        text="Dependence Plot (t-1)  (Cam = mùa khô, Xanh = mùa mưa)",
                        font=dict(size=13, color=C["accent"]),
                        x=0,
                    ),
                    "xaxis_title": "AQI ngày hôm qua (t-1)",
                    "yaxis_title": "SHAP Value của (t-1)",
                }
            )
            dep_block = card(
                [
                    sec("scatter-chart", "Dependence Plot (t-1) x is_dry_season"),
                    dcc.Graph(
                        figure=fdep,
                        config={"displayModeBar": False},
                        style={"height": "300px"},
                    ),
                ],
                ex={"marginBottom": "12px"},
            )

        insight = card(
            [
                html.Div(
                    [
                        html.I(
                            className="bi bi-lightbulb-fill",
                            style={
                                "color": C["yellow"],
                                "fontSize": "0.95rem",
                                "marginRight": "6px",
                            },
                        ),
                        html.Span(
                            "Insight từ SHAP",
                            style={
                                "color": C["text"],
                                "fontWeight": "700",
                                "fontSize": "0.9rem",
                            },
                        ),
                    ],
                    style={
                        "marginBottom": "10px",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
                html.Ul(
                    [
                        html.Li(
                            f'"{fi_s.index[0]}" là feature quan trọng nhất - '
                            f"Mean |SHAP| = {fi_s.iloc[0]:.4f}."
                        ),
                        html.Li(
                            f'"{fi_s.index[1]}" đứng thứ 2 - '
                            f"Mean |SHAP| = {fi_s.iloc[1]:.4f}."
                        ),
                        html.Li(
                            "Màu đỏ trong Beeswarm: Giá trị feature cao → đẩy AQI lên. "
                            "Màu xanh: Giá trị thấp → kéo AQI xuống."
                        ),
                        html.Li(
                            f"Base value (AQI trung bình tập Train) = {base:.1f}. "
                            "Model điều chỉnh từ giá trị này dựa trên từng feature."
                        ),
                    ],
                    style={
                        "color": C["sub"],
                        "fontSize": "0.8rem",
                        "lineHeight": "1.8",
                        "paddingLeft": "18px",
                        "margin": 0,
                    },
                ),
            ],
            ex={
                "marginBottom": "12px",
                "borderLeft": f"3px solid {C['yellow']}",
                "borderTop": f"3px solid {C['yellow']}",
            },
        )

        return html.Div(
            [
                insight,
                card(
                    [
                        sec("bar-chart-fill", "SHAP Summary Bar"),
                        dcc.Graph(
                            figure=fbar,
                            config={"displayModeBar": False},
                            style={"height": "480px"},
                        ),
                    ],
                    ex={"marginBottom": "12px"},
                ),
                card(
                    [
                        sec("scatter-chart", "SHAP Beeswarm"),
                        dcc.Graph(
                            figure=fbee,
                            config={"displayModeBar": False},
                            style={"height": "480px"},
                        ),
                    ],
                    ex={"marginBottom": "12px"},
                ),
                card(
                    [
                        sec("waterfall", "Waterfall Plot - Giải thích 1 dự đoán"),
                        dcc.Graph(
                            figure=fwf,
                            config={"displayModeBar": False},
                            style={"height": "420px"},
                        ),
                    ],
                    ex={"marginBottom": "12px"},
                ),
                dep_block,
            ]
        )
    except ImportError:
        return empty_st(
            "📦", "Thiếu thư viện SHAP", "Cài đặt: pip install shap", "pip install shap"
        )
    except Exception as e:
        return empty_st("⚠️", "Lỗi khi tính SHAP", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 - DỰ ĐOÁN
# ══════════════════════════════════════════════════════════════════════════════


def predict_layout():
    log = load_pred()
    if log is None or len(log) == 0:
        return empty_st(
            "🔮",
            "Chưa có dự đoán nào",
            "Chạy predict.py để tạo dự đoán đầu tiên.",
            "python predict.py",
        )

    lt = log.iloc[-1]
    pred = float(lt.get("predicted_aqi_tomorrow", 0))
    cat = str(lt.get("aqi_category", AQI_CATEGORY(pred)))
    color = AQI_COLORS.get(cat, C["yellow"])
    tmr = (pd.to_datetime(lt["prediction_date"]) + timedelta(days=1)).strftime(
        "%d/%m/%Y"
    )

    kpi_p = html.Div(
        [
            kpi(
                "lightning-charge-fill",
                f"AQI Ngày mai - {tmr}",
                f"{pred:.0f}",
                color,
                cat,
            ),
            kpi(
                "activity",
                "AQI Hôm nay",
                f"{float(lt.get('actual_aqi_today', 0)):.0f}"
                if pd.notna(lt.get("actual_aqi_today"))
                else "?",
                C["accent"],
                "Thực tế ngày hôm nay",
            ),
            kpi(
                "calendar-check",
                "Tổng dự đoán đã lưu",
                f"{len(log)}",
                C["acc2"],
                "Số ngày đã chạy",
            ),
            kpi(
                "clock-history",
                "Cập nhật lúc",
                str(lt.get("generated_at", "?"))[:16],
                C["sub"],
                "",
            ),
        ],
        style={
            "display": "flex",
            "gap": "10px",
            "marginBottom": "12px",
            "flexWrap": "wrap",
        },
    )

    # Gauge
    fg = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred,
            number={"font": {"size": 48, "color": color}},
            gauge={
                "axis": {"range": [0, 200], "tickfont": {"size": 10}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": C["card"],
                "bordercolor": C["border"],
                "steps": [
                    {"range": [0, 50], "color": "rgba(0,200,83,0.15)"},
                    {"range": [50, 100], "color": "rgba(255,214,0,0.15)"},
                    {"range": [100, 150], "color": "rgba(255,109,0,0.15)"},
                    {"range": [150, 200], "color": "rgba(213,0,0,0.15)"},
                ],
            },
            title={"text": f"<b>{cat}</b>", "font": {"size": 14, "color": color}},
        )
    )
    fg.update_layout(**{**PL, "margin": dict(l=20, r=20, t=60, b=20), "height": 280})

    # Lịch sử
    fh = go.Figure()
    if "actual_aqi_today" in log.columns:
        fh.add_trace(
            go.Scatter(
                x=log["prediction_date"],
                y=log["actual_aqi_today"],
                mode="lines+markers",
                name="AQI thực tế hôm nay",
                line=dict(color=C["accent"], width=2),
                marker=dict(size=6),
                hovertemplate="%{x|%d/%m/%Y}<br>Thực tế:<b>%{y:.0f}</b><extra></extra>",
            )
        )
    fh.add_trace(
        go.Scatter(
            x=log["prediction_date"],
            y=log["predicted_aqi_tomorrow"],
            mode="lines+markers",
            name="AQI dự đoán ngày mai",
            line=dict(color=C["yellow"], width=2, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="%{x|%d/%m/%Y}<br>Dự đoán:<b>%{y:.0f}</b><extra></extra>",
        )
    )
    for thr, clr5, ll in [
        (50, "#00C853", "Good"),
        (100, "#FFD600", "Moderate"),
        (150, "#FF6D00", "USG"),
    ]:
        fh.add_hline(
            y=thr,
            line_dash="dot",
            line_color=clr5,
            opacity=0.3,
            annotation_text=ll,
            annotation_position="top right",
            annotation_font_size=9,
            annotation_font_color=clr5,
        )
    fh.update_layout(**PL)
    fh.update_layout(
        yaxis_title="US AQI",
        showlegend=True,
        title=dict(
            text="Lịch Sử Dự Đoán vs Thực Tế",
            font=dict(size=13, color=C["accent"]),
            x=0,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Bảng
    sl = log.copy().tail(10).iloc[::-1]
    sl["prediction_date"] = sl["prediction_date"].dt.strftime("%d/%m/%Y")
    if "predicted_for_date" in sl.columns:
        sl["predicted_for_date"] = pd.to_datetime(
            sl["predicted_for_date"], errors="coerce"
        ).dt.strftime("%d/%m/%Y")
    COLS = [
        c
        for c in [
            "prediction_date",
            "actual_aqi_today",
            "predicted_aqi_tomorrow",
            "aqi_category",
            "generated_at",
        ]
        if c in sl.columns
    ]
    CLBL = {
        "prediction_date": "Ngày chạy",
        "actual_aqi_today": "AQI hôm nay",
        "predicted_aqi_tomorrow": "AQI dự đoán",
        "aqi_category": "Mức độ",
        "generated_at": "Dự đoán lúc",
    }
    tbl2 = dash_table.DataTable(
        data=sl[COLS].to_dict("records"),
        columns=[{"name": CLBL.get(c, c), "id": c} for c in COLS],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": C["border"],
            "color": C["text"],
            "fontWeight": "600",
            "fontSize": "0.78rem",
            "border": "none",
        },
        style_cell={
            "backgroundColor": C["card"],
            "color": C["text"],
            "fontSize": "0.8rem",
            "padding": "10px 14px",
            "border": f"1px solid {C['border']}",
            "textAlign": "center",
        },
        style_data_conditional=[
            {
                "if": {"row_index": 0},
                "backgroundColor": "#1a3a5c",
                "color": C["accent"],
                "fontWeight": "600",
            }
        ],
    )

    return html.Div(
        [
            kpi_p,
            row(
                col(
                    card(
                        [
                            sec("speedometer", f"AQI Dự Đoán Cho {tmr}"),
                            dcc.Graph(
                                figure=fg,
                                config={"displayModeBar": False},
                                style={"height": "280px"},
                            ),
                        ]
                    ),
                    flex="1",
                ),
                col(
                    card(
                        [
                            sec("graph-up", "Lịch Sử Dự Đoán"),
                            dcc.Graph(
                                figure=fh,
                                config={"displayModeBar": False},
                                style={"height": "280px"},
                            ),
                        ]
                    ),
                    flex="3",
                ),
            ),
            card([sec("table", "10 Dự Đoán Gần Nhất"), tbl2]),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  🌿  AQI Dashboard - TP. Hồ Chí Minh  |  Nhóm 6")
    print("=" * 56)
    print(f"  Data    : {PROC_PATH}")
    print(
        f"  Model   : {'✅' if os.path.exists(MDL_PATH) else '⏳ chưa có'} {MDL_PATH}"
    )
    print(f"  Predict : {'✅' if os.path.exists(PRED) else '⏳ chưa có'} {PRED}")
    print("\n  📌  http://localhost:8050")
    print("  🔄  Auto-reload mỗi 24h (dcc.Interval)")
    print("=" * 56 + "\n")
    app.run(debug=False, port=8050)
