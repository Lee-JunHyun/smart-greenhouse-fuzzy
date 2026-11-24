"""
스마트 온실 퍼지 추론 제어 시스템 (Fuzzy Greenhouse Controller)
대학교 2학년 인공지능 과제 수준의 직관적이고 깔끔한 대시보드
주요 기능:
  1. 센서 입력 조절 (온도, 습도, 토양수분, 일조량)
  2. 퍼지 추론 결과 vs 기존 On-Off 방식 직관적 비교
  3. 퍼지 추론 3단계 시각화 (소속함수 퍼지화 -> 규칙 추론 -> 무게중심법 COG 비퍼지화)
  4. 간단한 1스텝 환경 피드백 테스트
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import importlib
import fuzzy_engine
importlib.reload(fuzzy_engine)
from fuzzy_engine import FuzzyGreenhouseSystem, CROP_PRESETS, trimf, trapmf

# -------------------------------------------------------------
# 1. 페이지 설정 및 심플 스타일
# -------------------------------------------------------------
st.set_page_config(
    page_title="스마트 온실 퍼지 제어 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 깔끔하고 정돈된 카드 및 컨테이너 스타일
st.markdown("""
<style>
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .compare-card-fuzzy {
        background-color: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 8px;
        padding: 18px;
    }
    .compare-card-onoff {
        background-color: #fef2f2;
        border: 1.5px solid #fca5a5;
        border-radius: 8px;
        padding: 18px;
    }
    .input-container {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .badge-good {
        color: #15803d;
        background-color: #dcfce7;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-bad {
        color: #b91c1c;
        background-color: #fee2e2;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. 사이드바: 작물 선택 및 권장 환경 기준 (마크다운 파싱 오류 없도록 순수 텍스트 처리)
# -------------------------------------------------------------
with st.sidebar:
    st.header("재배 작물 설정")
    
    clean_names = ["토마토", "딸기", "파프리카", "상추"]
    crop_list = [k for k in clean_names if k in CROP_PRESETS]
    if not crop_list:
        crop_list = list(CROP_PRESETS.keys())
        
    selected_crop = st.selectbox("작물 선택", crop_list, index=0)
    crop_info = CROP_PRESETS.get(selected_crop, list(CROP_PRESETS.values())[0])
    
    st.subheader(f"{selected_crop} 환경 정보")
    st.write(crop_info["description"])

    t_min, t_max = crop_info["temp_opt"]
    h_min, h_max = crop_info["humid_opt"]
    m_min, m_max = crop_info["moist_opt"]
    s_min, s_max = crop_info["solar_opt"]

    st.markdown("**권장 생육 기준**")
    st.write(f"- 적정 온도: {t_min:.1f} ~ {t_max:.1f} °C")
    st.write(f"- 적정 습도: {h_min:.1f} ~ {h_max:.1f} %")
    st.write(f"- 적정 수분: {m_min:.1f} ~ {m_max:.1f} %")
    st.write(f"- 적정 일조량: {s_min:.0f} ~ {s_max:.0f} W/m²")


# -------------------------------------------------------------
# 3. 메인 타이틀
# -------------------------------------------------------------
st.title("스마트 온실 퍼지 추론 제어 시스템")
st.caption("인공지능 퍼지 논리(Fuzzy Logic)를 활용한 온실 환경 자동 제어 및 기존 On-Off 방식 비교")
st.markdown("---")


# -------------------------------------------------------------
# 4. 섹션 1: 온실 환경 센서 입력 패널
# -------------------------------------------------------------
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.subheader("1. 온실 환경 센서 입력")
st.caption("슬라이더를 조절하거나 상황별 버튼을 눌러 센서값을 변경할 수 있습니다.")

t_mid = float((t_min + t_max) / 2.0)
h_mid = float((h_min + h_max) / 2.0)
m_mid = float((m_min + m_max) / 2.0)
s_mid = float((s_min + s_max) / 2.0)

# 퀵 프리셋 버튼
btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])
with btn_col1:
    if st.button("적정 상태", use_container_width=True):
        st.session_state["in_temp"] = t_mid
        st.session_state["in_humid"] = h_mid
        st.session_state["in_moist"] = m_mid
        st.session_state["in_solar"] = s_mid
with btn_col2:
    if st.button("고온·건조 상태", use_container_width=True):
        st.session_state["in_temp"] = 32.0
        st.session_state["in_humid"] = 40.0
        st.session_state["in_moist"] = 45.0
        st.session_state["in_solar"] = 800.0
with btn_col3:
    if st.button("저온·과습 상태", use_container_width=True):
        st.session_state["in_temp"] = 13.0
        st.session_state["in_humid"] = 85.0
        st.session_state["in_moist"] = 85.0
        st.session_state["in_solar"] = 150.0

# 4개 슬라이더
s1, s2, s3, s4 = st.columns(4)
with s1:
    in_temp = st.slider("현재 온도 (°C)", 0.0, 50.0, 
                        value=float(st.session_state.get("in_temp", t_mid + 2.0)), step=0.5, key="in_temp")
with s2:
    in_humid = st.slider("현재 습도 (%)", 10.0, 100.0, 
                         value=float(st.session_state.get("in_humid", h_mid)), step=1.0, key="in_humid")
with s3:
    in_moist = st.slider("토양 수분 (%)", 0.0, 100.0, 
                         value=float(st.session_state.get("in_moist", m_mid - 7.0)), step=1.0, key="in_moist")
with s4:
    in_solar = st.slider("일조량 (W/m²)", 0.0, 1200.0, 
                         value=float(st.session_state.get("in_solar", 600.0)), step=10.0, key="in_solar")
st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------------------------------
# 5. 퍼지 추론 연산 및 단순 On-Off 제어값 계산
# -------------------------------------------------------------
fuzzy_sys = FuzzyGreenhouseSystem(selected_crop)
inference_res = fuzzy_sys.infer_and_defuzzify(in_temp, in_humid, in_moist, in_solar)

f_irr = inference_res["irrigation_val"]
f_vent = inference_res["ventilation_val"]
fuzzed = inference_res["fuzzed"]
active_rules = inference_res["active_rules"]
diagnoses, action_summary = fuzzy_sys.generate_expert_diagnosis(in_temp, in_humid, in_moist, in_solar, f_irr, f_vent)

# 단순 On-Off 기준값 제어 계산 (2학년 수준의 직관적 비교)
# 토양수분이 최저 권장치 미만이면 관수 100%(ON), 이상이면 0%(OFF)
onoff_irr = 100.0 if in_moist < m_min else 0.0
# 온도가 최고 권장치 초과이면 환기팬 100%(ON), 이하이면 0%(OFF)
onoff_vent = 100.0 if in_temp > t_max else 0.0


# -------------------------------------------------------------
# 6. 섹션 2: 제어 결과 및 퍼지 vs On-Off 비교 (가장 직관적인 비교)
# -------------------------------------------------------------
st.subheader("2. 제어 결과 비교 (퍼지 제어 vs 기존 On-Off 제어)")
st.caption("단일 기준값(On-Off) 방식과 퍼지 추론 방식의 출력 차이를 한눈에 비교할 수 있습니다.")

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.markdown('<div class="compare-card-fuzzy">', unsafe_allow_html=True)
    st.markdown("### 퍼지 추론 제어 (AI)")
    st.write("환경의 모호한 상태를 반영하여 부드러운 **비례 제어**를 수행합니다.")
    
    st.write(f"**관수 밸브 개방률:** {f_irr:.1f}%")
    st.progress(min(1.0, max(0.0, f_irr / 100.0)))
    
    st.write(f"**환기팬 작동 세기:** {f_vent:.1f}%")
    st.progress(min(1.0, max(0.0, f_vent / 100.0)))
    
    st.markdown("---")
    st.write("**현재 판단:**")
    st.write(action_summary)
    st.markdown('</div>', unsafe_allow_html=True)

with res_col2:
    st.markdown('<div class="compare-card-onoff">', unsafe_allow_html=True)
    st.markdown("### 기존 고정 방식 (On-Off)")
    st.write("단순 기준값을 기준으로 **0% 또는 100%**로만 작동합니다.")
    
    st.write(f"**관수 밸브 상태:** {'100% (열림)' if onoff_irr > 0 else '0% (닫힘)'}")
    st.progress(onoff_irr / 100.0)
    
    st.write(f"**환기팬 상태:** {'100% (가동)' if onoff_vent > 0 else '0% (정지)'}")
    st.progress(onoff_vent / 100.0)
    
    st.markdown("---")
    st.write("**동작 설명:**")
    if onoff_irr > 0:
        st.write("- 토양수분이 기준보다 낮아 밸브를 최대로 개방합니다 (과습 위험).")
    else:
        st.write("- 토양수분이 기준 이상이므로 관수를 완전히 정지합니다.")
    if onoff_vent > 0:
        st.write("- 온도가 기준치를 초과하여 환기팬을 최대로 회전합니다 (급랭 위험).")
    else:
        st.write("- 온도가 기준 이하이므로 환기팬을 끕니다.")
    st.markdown('</div>', unsafe_allow_html=True)

# 핵심 비교 요약 배너
st.info("""
**비교 요약**: 
기존 On-Off 방식은 수분이 조금만 부족해도 100% 전출력 관수를 하여 물 낭비나 토양 과습이 발생하기 쉽습니다. 
반면 **퍼지 추론 시스템**은 토양 건조 정도와 일조량에 따라 필요한 만큼만 비례 급수(예: 40 - 60%)하고 환기 세기를 조절하므로 안정적인 환경을 유지합니다.
""")

st.markdown("---")


# -------------------------------------------------------------
# 7. 섹션 3: 퍼지 추론 3단계 시각화 (과제의 핵심 내용)
# -------------------------------------------------------------
st.subheader("3. 퍼지 추론 과정 단계별 확인")
st.caption("퍼지화 ➔ 규칙 평가 ➔ 비퍼지화의 3단계 처리 과정을 그래프와 표로 확인합니다.")

viz_tab1, viz_tab2, viz_tab3 = st.tabs([
    "1단계: 소속함수 퍼지화",
    "2단계: IF-THEN 규칙 추론",
    "3단계: 무게중심법(COG) 비퍼지화"
])

# -----------------
# 탭 1: 소속함수
# -----------------
with viz_tab1:
    st.write("각 센서의 실제 측정값(크리스프 값)이 퍼지 집합에 속하는 정도(소속도, 0~1)를 나타냅니다.")

    mf_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"온도 소속함수 (현재: {in_temp:.1f} °C)",
            f"습도 소속함수 (현재: {in_humid:.1f} %)",
            f"토양수분 소속함수 (현재: {in_moist:.1f} %)",
            f"일조량 소속함수 (현재: {in_solar:.0f} W/m²)"
        ),
        vertical_spacing=0.15
    )

    def draw_mf(fig, var_name, x_range, row, col, cur_val, cur_fuzzed):
        x_vals = np.linspace(x_range[0], x_range[1], 250)
        colors = ["#2563eb", "#16a34a", "#dc2626"]
        
        for i, (term, (ftype, params)) in enumerate(fuzzy_sys.mfs[var_name].items()):
            if ftype == "trimf":
                y_vals = trimf(x_vals, params)
            else:
                y_vals = trapmf(x_vals, params)
            deg = cur_fuzzed[term]
            c = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="lines",
                name=f"{term} (소속도: {deg:.2f})",
                line=dict(color=c, width=2)
            ), row=row, col=col)
            
        fig.add_vline(x=cur_val, line=dict(color="#0f172a", width=2, dash="dash"), row=row, col=col)

    draw_mf(mf_fig, "temp", (5, 45), 1, 1, in_temp, fuzzed["temp"])
    draw_mf(mf_fig, "humid", (20, 100), 1, 2, in_humid, fuzzed["humid"])
    draw_mf(mf_fig, "moist", (20, 100), 2, 1, in_moist, fuzzed["moist"])
    draw_mf(mf_fig, "solar", (0, 1200), 2, 2, in_solar, fuzzed["solar"])

    mf_fig.update_layout(
        height=500, margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white", showlegend=True
    )
    st.plotly_chart(mf_fig, use_container_width=True)


# -----------------
# 탭 2: 규칙 추론
# -----------------
with viz_tab2:
    st.write("현재 센서 상태에 따라 발동된 전문가 IF-THEN 규칙 목록입니다.")
    if active_rules:
        table_rows = []
        for r in active_rules:
            actions = []
            if r["irr_consequence"]:
                actions.append(f"관수 ➔ {r['irr_consequence']}")
            if r["vent_consequence"]:
                actions.append(f"환기 ➔ {r['vent_consequence']}")
            table_rows.append({
                "규칙 번호": r["id"],
                "조건 (IF)": r["condition_str"],
                "발동 가중치 (Weight)": f"{r['weight']:.2f}",
                "결과 (THEN)": ", ".join(actions),
                "설명": r["desc"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.write("현재 조건에서 크게 활성화된 규칙이 없습니다.")


# -----------------
# 탭 3: 무게중심법(COG) 비퍼지화
# -----------------
with viz_tab3:
    st.write("추론된 출력 퍼지 집합의 무게중심(Center of Gravity)을 계산하여 최종 제어값(%)을 구합니다.")

    cog_fig = go.Figure()
    z = inference_res["output_z"]

    # 관수량 곡선
    cog_fig.add_trace(go.Scatter(
        x=z, y=inference_res["agg_irr"], mode="lines", fill="tozeroy",
        name=f"관수량 합성 면적 (최종값: {f_irr:.1f}%)",
        line=dict(color="#0284c7", width=2.5),
        fillcolor="rgba(2, 132, 199, 0.25)"
    ))

    # 환기량 곡선
    cog_fig.add_trace(go.Scatter(
        x=z, y=inference_res["agg_vent"], mode="lines", fill="tozeroy",
        name=f"환기량 합성 면적 (최종값: {f_vent:.1f}%)",
        line=dict(color="#16a34a", width=2.5),
        fillcolor="rgba(22, 163, 74, 0.2)"
    ))

    # 관수 / 환기 무게중심 수직선
    cog_fig.add_vline(x=f_irr, line=dict(color="#0284c7", width=2.5, dash="dot"))
    cog_fig.add_vline(x=f_vent, line=dict(color="#16a34a", width=2.5, dash="dash"))

    # 높이(y)를 완전히 분리하여 고온·건조 등 두 제어값이 가까울 때 글자가 겹치지 않도록 방지
    # 1) 관수 라벨: 상단(y=0.92)에 배치
    cog_fig.add_annotation(
        x=f_irr, y=0.92,
        text=f"관수: {f_irr:.1f}%",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#0284c7",
        ax=-55 if f_irr > 50 else 55, ay=0,
        bgcolor="rgba(30, 41, 59, 0.9)", bordercolor="#0284c7", borderwidth=1.5, borderpad=4,
        font=dict(size=12, color="#38bdf8")
    )

    # 2) 환기 라벨: 중하단(y=0.55)에 배치
    cog_fig.add_annotation(
        x=f_vent, y=0.55,
        text=f"환기: {f_vent:.1f}%",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#16a34a",
        ax=-55 if f_vent > 50 else 55, ay=0,
        bgcolor="rgba(30, 41, 59, 0.9)", bordercolor="#16a34a", borderwidth=1.5, borderpad=4,
        font=dict(size=12, color="#4ade80")
    )

    cog_fig.update_layout(
        title="출력 소속도 함수 및 무게중심(COG) 계산",
        xaxis_title="제어 출력 (0 ~ 100%)",
        yaxis_title="소속도",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 1.05]),
        height=380, margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(cog_fig, use_container_width=True)

st.markdown("---")


# -------------------------------------------------------------
# 8. 섹션 4: 간단한 1단계 환경 피드백 테스트
# -------------------------------------------------------------
st.subheader("4. 1단계 제어 적용 및 피드백 확인")
st.caption("현재 퍼지 제어값에 따라 온실 환경이 어떻게 변화하는지 1스텝 변화량을 확인합니다.")

fb_col1, fb_col2 = st.columns([1, 2])

with fb_col1:
    apply_btn = st.button("제어 적용 후 센서 변화 예측", type="primary", use_container_width=True)

with fb_col2:
    if apply_btn:
        # 간단한 1단계 변화량 물리 모델
        # 관수량이 50%이면 토양수분 약 +2.5% 상승, 증발은 -0.5%
        delta_moist = (f_irr / 100.0) * 5.0 - 0.5
        new_moist = min(100.0, max(0.0, in_moist + delta_moist))

        # 환기량이 50%이면 온도가 외기(약 22도) 방향으로 1도 냉각
        cooling_target = 22.0
        delta_temp = - (f_vent / 100.0) * max(0.0, in_temp - cooling_target) * 0.4
        new_temp = in_temp + delta_temp

        st.success(f"""
        **제어 적용 결과 예측**:
        - **토양수분**: {in_moist:.1f}% ➔ **{new_moist:.1f}%** ({'+' if delta_moist>=0 else ''}{delta_moist:.1f}%)
        - **내부온도**: {in_temp:.1f}°C ➔ **{new_temp:.1f}°C** ({delta_temp:.1f}°C)
        - 센서가 변화된 환경을 다시 읽어 다음 퍼지 제어에 피드백합니다.
        """)
    else:
        st.write("버튼을 누르면 현재 관수/환기 제어값에 따른 센서값 변화가 계산됩니다.")
