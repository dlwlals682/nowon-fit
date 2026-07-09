"""
NOWON-FIT
노원구 청년 이탈 조기경보 · 정책 처방 · 양면 추천 통합 시스템

⚠️ 데이터 안내
이 앱은 행안부 주민등록통계, 국토부 실거래가, 소상공인진흥공단 상가DB,
서울 열린데이터광장 범죄통계, 노원구 열린데이터광장 복지시설 데이터를
연동하는 것을 목표로 설계되었습니다. 현재 버전은 실제 API 키 없이도
바로 실행/시연할 수 있도록 위 데이터의 통계적 특성을 반영한 시뮬레이션
데이터를 사용합니다. `load_data()` 함수의 각 블록을 실제 공공데이터
포털 API 호출로 교체하면 동일한 로직으로 실데이터 기반 시스템이 됩니다.
"""

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
    "하계1동", "하계2동", "중계본동", "중계1동", "중계2·3동",
    "중계4동", "상계1동", "상계2동", "상계3·4동", "상계5동",
    "상계6·7동", "상계8동", "상계9동", "상계10동",
]

YEARS = list(range(2015, 2026))

POLICY_OPTIONS = {
    "월세 지원 (청년 1인당 月 20만원)": {"unit_cost": 2_400_000, "effect": {"housing": 0.9, "infra": 0.0, "safety": 0.0}},
    "공유오피스 조성 (1개소)":         {"unit_cost": 80_000_000, "effect": {"housing": 0.0, "infra": 0.7, "safety": 0.1}},
    "커뮤니티 공간 조성 (1개소)":       {"unit_cost": 50_000_000, "effect": {"housing": 0.0, "infra": 0.6, "safety": 0.2}},
    "CCTV 증설 (10대)":               {"unit_cost": 30_000_000, "effect": {"housing": 0.0, "infra": 0.1, "safety": 0.8}},
    "청년 창업공간 지원 (1개소)":       {"unit_cost": 60_000_000, "effect": {"housing": 0.1, "infra": 0.5, "safety": 0.0}},
}


# ----------------------------------------------------------------------------
# 데이터 생성 (실제 서비스 시 공공데이터 API 호출로 대체)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    rng = np.random.default_rng(42)

    old_apt_ratio = np.clip(rng.normal(0.55, 0.15, len(DONGS)), 0.15, 0.9)
    old_apt_ratio[DONGS.index("상계5동")] = 0.82
    old_apt_ratio[DONGS.index("상계6·7동")] = 0.85
    old_apt_ratio[DONGS.index("중계4동")] = 0.35
    old_apt_ratio[DONGS.index("공릉1동")] = 0.30

    base_youth = rng.integers(9000, 22000, len(DONGS)).astype(float)
    decline_rate = 0.004 + old_apt_ratio * 0.035 + rng.normal(0, 0.006, len(DONGS))
    decline_rate = np.clip(decline_rate, -0.01, 0.06)

    pop_records = []
    for i, dong in enumerate(DONGS):
        pop = base_youth[i]
        for y in YEARS:
            noise = rng.normal(0, pop * 0.01)
            pop_records.append({"동": dong, "연도": y, "청년인구": max(pop + noise, 0)})
            pop *= (1 - decline_rate[i])
    pop_df = pd.DataFrame(pop_records)

    housing_cost = np.clip(rng.normal(650, 120, len(DONGS)) - old_apt_ratio * 80, 350, 1100)  # 전월세 환산가(만원, 임의지수)
    infra_density = np.clip(rng.normal(55, 20, len(DONGS)) - old_apt_ratio * 25, 5, 100)       # 상가/인프라 밀도 지수
    crime_index = np.clip(rng.normal(40, 15, len(DONGS)), 5, 95)                                # 낮을수록 안전
    welfare_count = rng.integers(3, 25, len(DONGS))
    commute_time = np.clip(rng.normal(48, 10, len(DONGS)) + old_apt_ratio * 8, 25, 80)          # 도심 통근시간(분, 임의지수)

    static_df = pd.DataFrame({
        "동": DONGS,
        "노후아파트비율": old_apt_ratio,
        "주거비지수": housing_cost,
        "인프라밀도": infra_density,
        "범죄지수": crime_index,
        "복지시설수": welfare_count,
        "통근시간": commute_time,
    })

    return pop_df, static_df


def normalize(series, invert=False):
    s = (series - series.min()) / (series.max() - series.min() + 1e-9) * 100
    return 100 - s if invert else s


@st.cache_data
def compute_ysi(static_df):
    df = static_df.copy()
    df["주거비점수"] = normalize(df["주거비지수"], invert=True)
    df["인프라점수"] = normalize(df["인프라밀도"], invert=False)
    df["안전점수"] = normalize(df["범죄지수"], invert=True)
    df["통근점수"] = normalize(df["통근시간"], invert=True)
    df["YSI"] = (
        df["주거비점수"] * 0.3
        + df["인프라점수"] * 0.25
        + df["안전점수"] * 0.25
        + df["통근점수"] * 0.2
    ).round(1)
    return df


@st.cache_data
def forecast_population(pop_df, horizon=2027):
    results = []
    for dong in DONGS:
        sub = pop_df[pop_df["동"] == dong].sort_values("연도")
        x = sub["연도"].values
        y = sub["청년인구"].values
        coef = np.polyfit(x, y, 1)
        pred_years = list(x) + list(range(x.max() + 1, horizon + 1))
        pred_vals = np.polyval(coef, pred_years)
        pop_2025 = y[-1]
        pop_horizon = pred_vals[-1]
        change_rate = (pop_horizon - pop_2025) / pop_2025 * 100
        results.append({
            "동": dong,
            "2025인구": pop_2025,
            f"{horizon}예측인구": max(pop_horizon, 0),
            "변화율(%)": round(change_rate, 1),
            "연간감소율(%)": round(-coef[0] / pop_2025 * 100, 2) if pop_2025 else 0,
        })
    return pd.DataFrame(results)


def risk_level(change_rate):
    if change_rate <= -12:
        return "🔴 고위험"
    elif change_rate <= -6:
        return "🟠 주의"
    else:
        return "🟢 안정"


# ----------------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------------
pop_df, static_df = load_data()
ysi_df = compute_ysi(static_df)
forecast_df = forecast_population(pop_df)
forecast_df["위험도"] = forecast_df["변화율(%)"].apply(risk_level)

# ----------------------------------------------------------------------------
# 헤더
# ----------------------------------------------------------------------------
st.title("🏙️ NOWON-FIT")
st.markdown(
    "**노원구 19개 동을 대상으로 청년 이탈을 예측하고, 최적의 정책을 처방하고, "
    "청년에게는 살 곳을 추천하는 통합 시스템**"
)
st.caption(
    "⚠️ 데모 모드: 현재 화면의 수치는 문서에 기술된 공공데이터 출처(행안부 주민등록통계, "
    "국토부 실거래가, 소상공인진흥공단 상가DB, 서울 열린데이터광장 범죄통계 등)의 통계적 "
    "패턴을 반영한 시뮬레이션 데이터입니다. 실 서비스 전환 시 `load_data()`를 실제 API 호출로 교체하세요."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("노원구 대상 동 수", f"{len(DONGS)}개")
col2.metric("2025년 청년인구(합계)", f"{int(pop_df[pop_df['연도']==2025]['청년인구'].sum()):,}명")
col3.metric("2027년 예측(합계)", f"{int(forecast_df['2027예측인구'].sum()):,}명",
            delta=f"{forecast_df['변화율(%)'].mean():.1f}% (평균)")
col4.metric("고위험 동 수", f"{(forecast_df['위험도']=='🔴 고위험').sum()}개")

st.divider()

tab1, tab2, tab3 = st.tabs(["🚨 조기경보", "💊 정책 처방", "🔄 양면 추천"])

# ----------------------------------------------------------------------------
# 기능 1. 조기경보
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("이 동은 몇 년 안에 위험해질까요?")
    st.markdown(
        "행안부 주민등록통계 기반 시계열 추세를 선형 회귀로 학습해, 현재 추세가 이어질 경우 "
        "**2027년까지의 동별 청년인구 변화**를 예측합니다."
    )

    left, right = st.columns([1.3, 1])

    with left:
        sorted_forecast = forecast_df.sort_values("변화율(%)")
        fig = px.bar(
            sorted_forecast, x="변화율(%)", y="동", orientation="h",
            color="위험도",
            color_discrete_map={"🔴 고위험": "#e74c3c", "🟠 주의": "#f39c12", "🟢 안정": "#2ecc71"},
            title="2025 → 2027 청년인구 변화율 예측",
            height=650,
        )
        fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': sorted_forecast["동"]})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("##### 동별 상세 조회")
        selected_dong = st.selectbox("동 선택", DONGS, key="warn_dong")

        sub = pop_df[pop_df["동"] == selected_dong].sort_values("연도")
        row = forecast_df[forecast_df["동"] == selected_dong].iloc[0]

        st.metric("2025년 청년인구", f"{int(row['2025인구']):,}명")
        st.metric("2027년 예측", f"{int(row['2027예측인구']):,}명", delta=f"{row['변화율(%)']}%")
        st.metric("위험도", row["위험도"])
        st.metric("연평균 감소율", f"{row['연간감소율(%)']}%")

        x = sub["연도"].tolist()
        y = sub["청년인구"].tolist()
        coef = np.polyfit(sub["연도"], sub["청년인구"], 1)
        future_years = [2026, 2027]
        future_vals = np.polyval(coef, future_years)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="실측(2015-2025)", line=dict(color="#3498db")))
        fig2.add_trace(go.Scatter(
            x=[x[-1]] + future_years, y=[y[-1]] + list(future_vals),
            mode="lines+markers", name="예측(2026-2027)", line=dict(color="#e74c3c", dash="dash")
        ))
        fig2.update_layout(title=f"{selected_dong} 청년인구 추이", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### 전체 동 위험도 테이블")
    st.dataframe(
        forecast_df.sort_values("변화율(%)")[["동", "2025인구", "2027예측인구", "변화율(%)", "연간감소율(%)", "위험도"]]
        .style.format({"2025인구": "{:,.0f}", "2027예측인구": "{:,.0f}"}),
        use_container_width=True, hide_index=True,
    )

# ----------------------------------------------------------------------------
# 기능 2. 정책 처방
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("이 예산으로는 무엇을 먼저 해야 할까요?")
    st.markdown(
        "동의 **YSI(청년 정착 지수)**를 구성하는 요소별 취약도와 정책 수단별 비용·효과를 바탕으로, "
        "주어진 예산 안에서 YSI 상승폭이 최대인 정책 조합을 자동으로 계산합니다."
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

            fig3 = go.Figure(go.Bar(
                x=["현재 YSI", "정책 적용 후 YSI"], y=[dong_row["YSI"], new_ysi],
                marker_color=["#95a5a6", "#27ae60"], text=[f"{dong_row['YSI']}", f"{new_ysi:.1f}"],
                textposition="outside",
            ))
            fig3.update_layout(title=f"{target_dong} YSI 변화 예측", yaxis_range=[0, 100], height=350)
            st.plotly_chart(fig3, use_container_width=True)

            st.info(f"잔여 예산: {remaining:,}원 (추가 정책 실행에 부족하여 배정 종료)")
        else:
            st.warning("선택한 예산으로 실행 가능한 정책이 없습니다. 예산을 늘리거나 정책을 다시 선택해주세요.")

    with st.expander("📌 이 동의 YSI 구성요소 상세 보기"):
        radar_cats = ["주거비점수", "인프라점수", "안전점수", "통근점수"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatterpolar(r=dong_row[radar_cats].values, theta=radar_cats, fill='toself', name=target_dong))
        fig4.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400)
        st.plotly_chart(fig4, use_container_width=True)

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
        w_infra = s2.slider("🏪 생활 인프라", 0, 10, 5)
        w_safety = s3.slider("🛡️ 안전", 0, 10, 6)
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
        fig5 = px.bar(df, x="맞춤점수", y="동", orientation="h", height=650,
                       color="맞춤점수", color_continuous_scale="Blues")
        fig5.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig5, use_container_width=True)

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
            fig6 = go.Figure()
            fig6.add_trace(go.Scatterpolar(r=list(factors.values()), theta=radar_cats, fill='toself', name="현재"))
            fig6.add_trace(go.Scatterpolar(r=list(improved_factors.values()), theta=radar_cats, fill='toself', name="개선 후"))
            fig6.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420,
                                title=f"{admin_dong} 요소별 점수 비교")
            st.plotly_chart(fig6, use_container_width=True)

st.divider()
st.caption("NOWON-FIT · 노원구 19개 동 청년 정착 지원 통합 시스템 (데모 버전)")
