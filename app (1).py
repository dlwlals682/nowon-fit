"""
NOWON-FIT
노원구 청년 이탈 조기경보 · 정책 처방 · 양면 추천 통합 시스템

📊 데이터 안내 (2026-07 기준)
[실데이터]
- 청년 인구(월간, 2023.01~2025.12, 동별 청년/청소년/아동 인구): 행안부 주민등록 인구 기타현황
- 총인구(연간, 2016~2025): 행안부 주민등록 인구 및 세대현황
- 평균연령(연간, 2016~2025): 행안부 주민등록 인구 기타현황(평균연령)
- 주거비: 서울시 부동산 실거래가 정보(아파트, 법정동 단위 → 행정동 균등 배분 추정)
- 안전(방재 인프라): 노원구 비상대피시설 목록(행정동 단위 실측 — 시설수/수용면적)
- 복지시설: 서울시 사회복지시설 목록 4종(노인주거·노인의료·장애인재활·다문화가족,
  법정동 단위 → 행정동 균등 배분 추정)

[시뮬레이션 — 실데이터 미확보]
- 생활 인프라 밀도(상가업소 수): 소상공인시장진흥공단 상가정보 API 인증키 필요, 미연동
- 통근시간: 실데이터 미확보

※ 법정동 → 행정동 배분 관련: 실거래가·복지시설 원본 주소는 법정동(예: '상계동')까지만
표기되어 있어, 그 법정동에 속한 여러 행정동(예: 상계1~10동)에 평균값을 동일하게
적용했습니다. 행정동별 정밀한 차이는 반영되지 않은 근사치입니다.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NOWON-FIT | 노원구 청년 정착 지수 시스템",
    page_icon="🏙️",
    layout="wide",
)

DONGS = [
    "월계1동", "월계2동", "월계3동", "공릉1동", "공릉2동",
    "하계1동", "하계2동", "중계본동", "중계1동", "중계4동",
    "중계2.3동", "상계1동", "상계2동", "상계3.4동", "상계5동",
    "상계6.7동", "상계8동", "상계9동", "상계10동",
]

_DIR = os.path.dirname(os.path.abspath(__file__))
POP_DATA_FILE = os.path.join(_DIR, "nowon_population.csv")
AGE_DATA_FILE = os.path.join(_DIR, "nowon_avg_age.csv")
YOUTH_DATA_FILE = os.path.join(_DIR, "nowon_youth_population.csv")
SHELTER_DATA_FILE = os.path.join(_DIR, "nowon_shelters.csv")
HOUSING_DATA_FILE = os.path.join(_DIR, "nowon_housing_price.csv")
WELFARE_DATA_FILE = os.path.join(_DIR, "nowon_welfare.csv")

POLICY_OPTIONS = {
    "월세 지원 (청년 1인당 月 20만원)": {"unit_cost": 2_400_000, "effect": {"housing": 0.9, "infra": 0.0, "safety": 0.0}},
    "공유오피스 조성 (1개소)":         {"unit_cost": 80_000_000, "effect": {"housing": 0.0, "infra": 0.7, "safety": 0.1}},
    "커뮤니티·복지 공간 조성 (1개소)":  {"unit_cost": 50_000_000, "effect": {"housing": 0.0, "infra": 0.6, "safety": 0.2}},
    "재난 대비시설·대피소 접근성 정비": {"unit_cost": 30_000_000, "effect": {"housing": 0.0, "infra": 0.1, "safety": 0.8}},
    "청년 창업공간 지원 (1개소)":       {"unit_cost": 60_000_000, "effect": {"housing": 0.1, "infra": 0.5, "safety": 0.0}},
}


# ----------------------------------------------------------------------------
# 데이터 로드 — 실데이터
# ----------------------------------------------------------------------------
@st.cache_data
def load_population_data():
    """행안부 주민등록 인구 및 세대현황 (2016-2025, 동별 총인구, 전 연령)"""
    df = pd.read_csv(POP_DATA_FILE)
    df = df[df["동"].isin(DONGS)].copy()
    df = df.rename(columns={"총인구": "인구"})
    return df[["동", "연도", "인구", "세대수", "세대당인구"]]


@st.cache_data
def load_avg_age_data():
    """행안부 주민등록 인구기타현황(평균연령) (2016-2025)"""
    df = pd.read_csv(AGE_DATA_FILE)
    df = df[df["동"].isin(DONGS)].copy()
    return df[["동", "연도", "평균연령", "남자평균연령", "여자평균연령"]]


@st.cache_data
def load_youth_population_data():
    """행안부 주민등록 인구기타현황(아동청소년청년) (2023.01-2025.12, 월간)"""
    df = pd.read_csv(YOUTH_DATA_FILE)
    df = df[df["동"].isin(DONGS)].copy()
    df["청년비중"] = (df["청년전체"] / df["전체"] * 100).round(1)
    return df


@st.cache_data
def load_real_indicators():
    """실거래가·비상대피시설·복지시설 실데이터 + 상가/통근시간 시뮬레이션을 결합."""
    housing = pd.read_csv(HOUSING_DATA_FILE)[["동", "평균평당가_만원"]]
    shelter = pd.read_csv(SHELTER_DATA_FILE)[["동", "대피시설수", "대피시설총면적"]]
    welfare = pd.read_csv(WELFARE_DATA_FILE)[["동", "복지시설수_추정"]]

    static_df = housing.merge(shelter, on="동").merge(welfare, on="동")
    static_df = static_df[static_df["동"].isin(DONGS)].reset_index(drop=True)

    # 상가밀도·통근시간: 실데이터 미확보 → 시뮬레이션 (재현 가능하도록 시드 고정)
    rng = np.random.default_rng(42)
    order = static_df["동"].tolist()
    idx_map = {d: i for i, d in enumerate(order)}
    base_bias = np.array([
        -0.15 if "상계" in d else (0.1 if d in ("공릉1동", "중계4동") else 0.0)
        for d in order
    ])
    infra_density_sim = np.clip(rng.normal(55, 18, len(order)) + base_bias * 40, 5, 100)
    commute_time_sim = np.clip(rng.normal(48, 9, len(order)) - base_bias * 15, 25, 80)

    static_df["상가밀도_시뮬"] = infra_density_sim
    static_df["통근시간_시뮬"] = commute_time_sim
    return static_df


def normalize(series, invert=False):
    s = (series - series.min()) / (series.max() - series.min() + 1e-9) * 100
    return 100 - s if invert else s


@st.cache_data
def compute_ysi(static_df):
    df = static_df.copy()
    df["주거비점수"] = normalize(df["평균평당가_만원"], invert=True)
    df["안전점수"] = (
        normalize(df["대피시설수"], invert=False) * 0.4
        + normalize(df["대피시설총면적"], invert=False) * 0.6
    )
    infra_score = normalize(df["상가밀도_시뮬"], invert=False)
    welfare_score = normalize(df["복지시설수_추정"], invert=False)
    df["인프라점수"] = infra_score * 0.55 + welfare_score * 0.45
    df["통근점수"] = normalize(df["통근시간_시뮬"], invert=True)
    df["YSI"] = (
        df["주거비점수"] * 0.30
        + df["인프라점수"] * 0.25
        + df["안전점수"] * 0.25
        + df["통근점수"] * 0.20
    ).round(1)
    return df


# ----------------------------------------------------------------------------
# 예측 로직
# ----------------------------------------------------------------------------
@st.cache_data
def forecast_population(pop_df, age_df, horizon=2027):
    """총인구(10년 연간 실데이터) 기반 예측 — 보조 지표"""
    results = []
    for dong in DONGS:
        sub = pop_df[pop_df["동"] == dong].sort_values("연도")
        x = sub["연도"].values
        y = sub["인구"].values
        coef = np.polyfit(x, y, 1)
        pred_years = list(x) + list(range(x.max() + 1, horizon + 1))
        pred_vals = np.polyval(coef, pred_years)
        pop_2025 = y[-1]
        pop_horizon = pred_vals[-1]
        change_rate = (pop_horizon - pop_2025) / pop_2025 * 100

        age_sub = age_df[age_df["동"] == dong].sort_values("연도")
        age_2016 = age_sub[age_sub["연도"] == age_sub["연도"].min()]["평균연령"].values[0]
        age_2025 = age_sub[age_sub["연도"] == age_sub["연도"].max()]["평균연령"].values[0]

        results.append({
            "동": dong,
            "2025인구": pop_2025,
            f"{horizon}예측인구": max(pop_horizon, 0),
            "변화율(%)": round(change_rate, 1),
            "연간감소율(%)": round(-coef[0] / pop_2025 * 100, 2) if pop_2025 else 0,
            "2016평균연령": age_2016,
            "2025평균연령": age_2025,
            "평균연령상승폭": round(age_2025 - age_2016, 1),
        })
    return pd.DataFrame(results)


@st.cache_data
def forecast_youth(youth_df, horizon_year=2027):
    """청년인구(3년 월간 실데이터) 기반 예측 — 조기경보 핵심 지표"""
    results = []
    curves = {}
    for dong in DONGS:
        sub = youth_df[youth_df["동"] == dong].copy()
        sub["x"] = (sub["연도"] - 2023) * 12 + sub["월"]
        sub = sub.sort_values("x")
        x = sub["x"].values
        y = sub["청년전체"].values
        coef = np.polyfit(x, y, 1)

        x_horizon = (horizon_year - 2023) * 12 + 12
        future_x = list(range(int(x.max()) + 1, x_horizon + 1))
        future_y = np.polyval(coef, future_x)

        latest = y[-1]
        pred_horizon = future_y[-1] if len(future_y) else latest
        change_rate = (pred_horizon - latest) / latest * 100 if latest else 0
        monthly_decline = -coef[0]

        results.append({
            "동": dong,
            "2025-12청년인구": latest,
            f"{horizon_year}-12예측청년인구": max(pred_horizon, 0),
            "변화율(%)": round(change_rate, 1),
            "월평균감소": round(monthly_decline, 1),
            "청년비중_최신": sub["청년비중"].iloc[-1],
        })
        curves[dong] = {
            "x": list(x), "y": list(y),
            "future_x": future_x, "future_y": list(future_y),
        }
    return pd.DataFrame(results), curves


def risk_level_youth(change_rate):
    if change_rate <= -8:
        return "🔴 고위험"
    elif change_rate <= -4:
        return "🟠 주의"
    else:
        return "🟢 안정"


def risk_level_pop(change_rate):
    if change_rate <= -12:
        return "🔴 고위험"
    elif change_rate <= -6:
        return "🟠 주의"
    else:
        return "🟢 안정"


# ----------------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------------
pop_df = load_population_data()
age_df = load_avg_age_data()
youth_df = load_youth_population_data()
static_df = load_real_indicators()
ysi_df = compute_ysi(static_df)

pop_forecast_df = forecast_population(pop_df, age_df)
pop_forecast_df["위험도"] = pop_forecast_df["변화율(%)"].apply(risk_level_pop)

youth_forecast_df, youth_curves = forecast_youth(youth_df)
youth_forecast_df["위험도"] = youth_forecast_df["변화율(%)"].apply(risk_level_youth)

# ----------------------------------------------------------------------------
# 헤더
# ----------------------------------------------------------------------------
st.title("🏙️ NOWON-FIT")
st.markdown(
    "**노원구 19개 동을 대상으로 청년 이탈을 예측하고, 최적의 정책을 처방하고, "
    "청년에게는 살 곳을 추천하는 통합 시스템**"
)
st.caption(
    "✅ 실데이터: 청년인구(월간 2023-2025) · 총인구·평균연령(연간 2016-2025) · "
    "실거래가(주거비) · 비상대피시설(안전) · 사회복지시설 4종(복지) &nbsp;|&nbsp; "
    "⚠️ 시뮬레이션: 상가 밀도, 통근시간 &nbsp;|&nbsp; "
    "※ 실거래가·복지시설은 법정동→행정동 균등 배분 추정치",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("노원구 대상 동 수", f"{len(DONGS)}개")
col2.metric("2025-12 청년인구(합계)", f"{int(youth_forecast_df['2025-12청년인구'].sum()):,}명")
col3.metric("2027-12 예측(합계)", f"{int(youth_forecast_df['2027-12예측청년인구'].sum()):,}명",
            delta=f"{youth_forecast_df['변화율(%)'].mean():.1f}% (평균)")
col4.metric("고위험 동 수", f"{(youth_forecast_df['위험도']=='🔴 고위험').sum()}개")

st.divider()

tab1, tab2, tab3 = st.tabs(["🚨 조기경보", "💊 정책 처방", "🔄 양면 추천"])

# ----------------------------------------------------------------------------
# 기능 1. 조기경보
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("이 동은 몇 년 안에 위험해질까요?")
    st.markdown(
        "행안부 주민등록 실데이터(**청년인구, 2023.01~2025.12 월간**)의 추세를 선형 회귀로 학습해, "
        "현재 추세가 이어질 경우 **2027년 말까지의 동별 청년인구 변화**를 예측합니다."
    )

    left, right = st.columns([1.3, 1])

    with left:
        sorted_forecast = youth_forecast_df.sort_values("변화율(%)")
        fig = px.bar(
            sorted_forecast, x="변화율(%)", y="동", orientation="h",
            color="위험도",
            color_discrete_map={"🔴 고위험": "#e74c3c", "🟠 주의": "#f39c12", "🟢 안정": "#2ecc71"},
            title="2025-12 → 2027-12 청년인구 변화율 예측",
            height=650,
        )
        fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': sorted_forecast["동"]})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("##### 동별 상세 조회")
        selected_dong = st.selectbox("동 선택", DONGS, key="warn_dong")

        row = youth_forecast_df[youth_forecast_df["동"] == selected_dong].iloc[0]
        st.metric("2025-12 청년인구", f"{int(row['2025-12청년인구']):,}명")
        st.metric("2027-12 예측", f"{int(row['2027-12예측청년인구']):,}명", delta=f"{row['변화율(%)']}%")
        st.metric("위험도", row["위험도"])
        st.metric("청년 비중(전체인구 대비)", f"{row['청년비중_최신']}%")

        curve = youth_curves[selected_dong]
        x_labels = [f"{2023 + (m-1)//12}.{(m-1)%12+1:02d}" for m in curve["x"]]
        fx_labels = [f"{2023 + (m-1)//12}.{(m-1)%12+1:02d}" for m in curve["future_x"]]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_labels, y=curve["y"], mode="lines", name="실측(2023.01-2025.12)", line=dict(color="#3498db")))
        fig2.add_trace(go.Scatter(
            x=[x_labels[-1]] + fx_labels, y=[curve["y"][-1]] + curve["future_y"],
            mode="lines", name="예측(2026.01-2027.12)", line=dict(color="#e74c3c", dash="dash")
        ))
        fig2.update_layout(title=f"{selected_dong} 청년인구 추이", height=300, margin=dict(t=40, b=20),
                            xaxis=dict(tickmode="array", tickvals=list(range(0, len(x_labels)+len(fx_labels), 6))))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### 전체 동 청년인구 위험도 테이블")
    st.dataframe(
        youth_forecast_df.sort_values("변화율(%)")[
            ["동", "2025-12청년인구", "2027-12예측청년인구", "변화율(%)", "청년비중_최신", "위험도"]
        ].rename(columns={"청년비중_최신": "청년비중(%)"})
        .style.format({"2025-12청년인구": "{:,.0f}", "2027-12예측청년인구": "{:,.0f}"}),
        use_container_width=True, hide_index=True,
    )

    with st.expander("📌 보조 지표 — 총인구·평균연령 10년 추이 (실데이터, 2016-2025)"):
        st.caption("전 연령 총인구 기준 지표로, 위 청년인구 조기경보를 뒷받침하는 배경 신호로 참고하세요.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**동별 총인구 변화율 (2025→2027 예측)**")
            fig3 = px.bar(
                pop_forecast_df.sort_values("변화율(%)"), x="변화율(%)", y="동", orientation="h",
                color="위험도",
                color_discrete_map={"🔴 고위험": "#e74c3c", "🟠 주의": "#f39c12", "🟢 안정": "#2ecc71"},
                height=550,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            st.markdown("**인구감소 × 고령화 속도**")
            st.caption("가로축: 총인구 변화율 예측 · 세로축: 2016~2025 평균연령 상승폭(실측)")
            fig4 = px.scatter(
                pop_forecast_df, x="변화율(%)", y="평균연령상승폭", text="동", color="위험도",
                color_discrete_map={"🔴 고위험": "#e74c3c", "🟠 주의": "#f39c12", "🟢 안정": "#2ecc71"},
                height=550,
            )
            fig4.update_traces(textposition="top center", marker=dict(size=11))
            fig4.update_layout(xaxis_title="총인구 변화율(%)", yaxis_title="평균연령 상승폭(세)")
            st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------------------------
# 기능 2. 정책 처방
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("이 예산으로는 무엇을 먼저 해야 할까요?")
    st.markdown(
        "동의 **YSI(청년 정착 지수)**를 구성하는 요소별 취약도와 정책 수단별 비용·효과를 바탕으로, "
        "주어진 예산 안에서 YSI 상승폭이 최대인 정책 조합을 자동으로 계산합니다. "
        "(주거비=실거래가, 안전=비상대피시설 실데이터 / 인프라 일부=복지시설 실데이터+상가 시뮬레이션 / 통근=시뮬레이션)"
    )

    c1, c2 = st.columns([1, 2])

    with c1:
        target_dong = st.selectbox("대상 동", DONGS, key="policy_dong")
        budget = st.slider("가용 예산 (원)", 50_000_000, 500_000_000, 300_000_000, step=10_000_000,
                            format="%d")
        st.caption(f"💰 선택 예산: {budget:,}원")

        st.markdown("##### 정책 수단 선택 (선택 안 하면 전체 자동 검토)")
        candidate_policies = st.multiselect(
            "고려할 정책", list(POLICY_OPTIONS.keys()), default=list(POLICY_OPTIONS.keys())
        )

    dong_row = ysi_df[ysi_df["동"] == target_dong].iloc[0]

    weakness = {
        "housing": 100 - dong_row["주거비점수"],
        "infra": 100 - dong_row["인프라점수"],
        "safety": 100 - dong_row["안전점수"],
    }

    def marginal_value(policy_name, remaining_budget):
        p = POLICY_OPTIONS[policy_name]
        if p["unit_cost"] > remaining_budget:
            return 0, 0
        gain = sum(weakness[k] * v for k, v in p["effect"].items()) * 0.15
        return gain, p["unit_cost"]

    remaining = budget
    plan = []
    pool = candidate_policies.copy()
    max_units_each = 6

    while remaining > 0 and pool:
        best_ratio, best_policy, best_gain, best_cost = -1, None, 0, 0
        for name in pool:
            used = sum(1 for x in plan if x["정책"] == name)
            if used >= max_units_each:
                continue
            gain, cost = marginal_value(name, remaining)
            if cost == 0:
                continue
            ratio = gain / cost
            if ratio > best_ratio:
                best_ratio, best_policy, best_gain, best_cost = ratio, name, gain, cost
        if best_policy is None:
            break
        plan.append({"정책": best_policy, "비용": best_cost, "YSI기여": round(best_gain, 2)})
        remaining -= best_cost
        for k, v in POLICY_OPTIONS[best_policy]["effect"].items():
            weakness[k] = max(weakness[k] - v * 8, 0)

    with c2:
        if plan:
            plan_df = pd.DataFrame(plan)
            summary = plan_df.groupby("정책").agg(수량=("정책", "count"), 총비용=("비용", "sum"), 총YSI기여=("YSI기여", "sum")).reset_index()
            st.markdown(f"##### ✅ {target_dong} 추천 정책 조합 (예산 {budget:,}원 중 {budget-remaining:,}원 사용)")
            st.dataframe(
                summary.style.format({"총비용": "{:,.0f}", "총YSI기여": "{:.1f}"}),
                use_container_width=True, hide_index=True,
            )

            total_gain = min(plan_df["YSI기여"].sum(), 100 - dong_row["YSI"])
            new_ysi = min(dong_row["YSI"] + total_gain, 100)

            fig5 = go.Figure(go.Bar(
                x=["현재 YSI", "정책 적용 후 YSI"], y=[dong_row["YSI"], new_ysi],
                marker_color=["#95a5a6", "#27ae60"], text=[f"{dong_row['YSI']}", f"{new_ysi:.1f}"],
                textposition="outside",
            ))
            fig5.update_layout(title=f"{target_dong} YSI 변화 예측", yaxis_range=[0, 100], height=350)
            st.plotly_chart(fig5, use_container_width=True)

            st.info(f"잔여 예산: {remaining:,}원 (추가 정책 실행에 부족하여 배정 종료)")
        else:
            st.warning("선택한 예산으로 실행 가능한 정책이 없습니다. 예산을 늘리거나 정책을 다시 선택해주세요.")

    with st.expander("📌 이 동의 YSI 구성요소 상세 보기"):
        radar_cats = ["주거비점수", "인프라점수", "안전점수", "통근점수"]
        fig6 = go.Figure()
        fig6.add_trace(go.Scatterpolar(r=dong_row[radar_cats].values, theta=radar_cats, fill='toself', name=target_dong))
        fig6.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400)
        st.plotly_chart(fig6, use_container_width=True)
        st.caption(
            f"주거비: 평당 {dong_row['평균평당가_만원']:,.0f}만원(실거래가) · "
            f"대피시설 {int(dong_row['대피시설수'])}개소/{dong_row['대피시설총면적']:,.0f}㎡(실데이터) · "
            f"복지시설 {dong_row['복지시설수_추정']:.1f}개소(추정) · 상가밀도·통근시간은 시뮬레이션"
        )

# ----------------------------------------------------------------------------
# 기능 3. 양면 추천
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("같은 데이터, 다른 두 개의 화면")
    mode = st.radio("모드 선택", ["🧑 청년 모드 (살 곳 찾기)", "🏛️ 행정 모드 (정책 설계)"], horizontal=True)

    if mode.startswith("🧑"):
        st.markdown("당신에게 중요한 것을 조정하면, 노원구 19개 동 중 가장 적합한 곳 3곳을 추천합니다.")

        s1, s2, s3, s4 = st.columns(4)
        w_housing = s1.slider("💰 주거비", 0, 10, 8)
        w_infra = s2.slider("🏪 생활 인프라·복지", 0, 10, 5)
        w_safety = s3.slider("🛡️ 재난 안전(대피시설)", 0, 10, 6)
        w_commute = s4.slider("🚇 통근시간", 0, 10, 7)

        total_w = max(w_housing + w_infra + w_safety + w_commute, 1)
        df = ysi_df.copy()
        df["맞춤점수"] = (
            df["주거비점수"] * w_housing
            + df["인프라점수"] * w_infra
            + df["안전점수"] * w_safety
            + df["통근점수"] * w_commute
        ) / total_w
        df = df.sort_values("맞춤점수", ascending=False)

        top3 = df.head(3)
        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for i, (col, (_, r)) in enumerate(zip(cols, top3.iterrows())):
            with col:
                st.markdown(f"### {medals[i]} {r['동']}")
                st.metric("맞춤 점수", f"{r['맞춤점수']:.1f}점")
                st.progress(min(r["맞춤점수"] / 100, 1.0))
                st.caption(f"주거비 {r['주거비점수']:.0f} · 인프라 {r['인프라점수']:.0f} · 안전 {r['안전점수']:.0f} · 통근 {r['통근점수']:.0f}")

        st.markdown("##### 전체 동 순위")
        fig7 = px.bar(df, x="맞춤점수", y="동", orientation="h", height=650,
                       color="맞춤점수", color_continuous_scale="Blues")
        fig7.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig7, use_container_width=True)

    else:
        st.markdown("동을 선택하면 가장 취약한 요소와, 이를 개선했을 때의 청년 유입 효과를 확인합니다.")

        admin_dong = st.selectbox("동 선택", DONGS, key="admin_dong")
        row = ysi_df[ysi_df["동"] == admin_dong].iloc[0]

        factors = {"주거비": row["주거비점수"], "인프라": row["인프라점수"], "안전": row["안전점수"], "통근": row["통근점수"]}
        weakest = min(factors, key=factors.get)

        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.metric("현재 YSI", f"{row['YSI']}점")
            st.metric("가장 취약한 요소", f"{weakest} ({factors[weakest]:.0f}점)")

            improve = st.slider(f"'{weakest}' 요소를 몇 % 개선하면?", 0, 50, 20, step=5)
            improved_score = min(factors[weakest] * (1 + improve / 100), 100)
            weight_map = {"주거비": 0.3, "인프라": 0.25, "안전": 0.25, "통근": 0.2}
            new_ysi = row["YSI"] + (improved_score - factors[weakest]) * weight_map[weakest]
            new_ysi = min(new_ysi, 100)

            est_inflow = int((new_ysi - row["YSI"]) * 15)
            st.success(f"예상 YSI: {row['YSI']}점 → **{new_ysi:.1f}점**")
            st.success(f"예상 청년 순유입 효과: 약 **{max(est_inflow, 0):,}명**")

        with c2:
            radar_cats = list(factors.keys())
            improved_factors = factors.copy()
            improved_factors[weakest] = improved_score
            fig8 = go.Figure()
            fig8.add_trace(go.Scatterpolar(r=list(factors.values()), theta=radar_cats, fill='toself', name="현재"))
            fig8.add_trace(go.Scatterpolar(r=list(improved_factors.values()), theta=radar_cats, fill='toself', name="개선 후"))
            fig8.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420,
                                title=f"{admin_dong} 요소별 점수 비교")
            st.plotly_chart(fig8, use_container_width=True)

st.divider()
st.caption("NOWON-FIT · 노원구 19개 동 청년 정착 지원 통합 시스템")
