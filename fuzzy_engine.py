"""
퍼지 추론 기반 스마트 온실 환경 관리 시스템 - 퍼지 추론 엔진
논문: "퍼지 추론 기반 스마트 온실 관리 시스템" (전자공학부 인공지능 전공 이준현)
기능:
  - 소속함수 (Triangular, Trapezoidal Membership Functions)
  - 퍼지화 (Crisp Input -> Membership Degrees)
  - 전문가 IF-THEN 규칙 평가 (Mamdani Min Operator)
  - 무게중심법(Center of Gravity, COG) 비퍼지화
  - 작물별 최적 환경 프리셋 (토마토, 딸기, 파프리카, 상추)
"""

import numpy as np

def trimf(x, params):
    """삼각형 소속함수: params = [a, b, c]"""
    a, b, c = params
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    
    # Left slope
    idx1 = (a < x) & (x <= b)
    if b != a:
        y[idx1] = (x[idx1] - a) / (b - a)
    else:
        y[x == a] = 1.0
        
    # Right slope
    idx2 = (b < x) & (x < c)
    if c != b:
        y[idx2] = (c - x[idx2]) / (c - b)
    else:
        y[x == b] = 1.0
        
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapmf(x, params):
    """사다리꼴 소속함수: params = [a, b, c, d]"""
    a, b, c, d = params
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    
    # Left slope
    idx1 = (a < x) & (x < b)
    if b != a:
        y[idx1] = (x[idx1] - a) / (b - a)
    else:
        y[x == a] = 1.0
        
    # Flat top
    idx2 = (b <= x) & (x <= c)
    y[idx2] = 1.0
    
    # Right slope
    idx3 = (c < x) & (x < d)
    if d != c:
        y[idx3] = (d - x[idx3]) / (d - c)
    else:
        y[x == d] = 1.0
        
    return np.clip(y, 0.0, 1.0)


# 작물별 환경 프로필 프리셋 정의
CROP_PRESETS = {
    "토마토": {
        "description": "온대성 과채류로 주간 20 - 26°C, 토양수분 65 - 75% 유지가 적합합니다.",
        "temp_opt": (20.0, 26.0),
        "humid_opt": (60.0, 75.0),
        "moist_opt": (65.0, 75.0),
        "solar_opt": (400.0, 700.0),
        "mf": {
            "temp": {
                "Low": ("trapmf", [-10, 0, 16, 23]),
                "Optimal": ("trimf", [18, 23, 28]),
                "High": ("trapmf", [23, 28, 50, 60])
            },
            "humid": {
                "Low": ("trapmf", [-10, 0, 45, 60]),
                "Optimal": ("trimf", [50, 68, 82]),
                "High": ("trapmf", [68, 82, 100, 110])
            },
            "moist": {
                "Dry": ("trapmf", [-10, 0, 52, 70]),
                "Optimal": ("trimf", [58, 70, 82]),
                "Wet": ("trapmf", [70, 82, 100, 110])
            },
            "solar": {
                "Low": ("trapmf", [-10, 0, 150, 350]),
                "Moderate": ("trimf", [200, 500, 800]),
                "High": ("trapmf", [600, 800, 1200, 1300])
            }
        }
    },
    "딸기": {
        "description": "서늘한 기후를 선호하며 고온에 취약, 주간 18 - 23°C 및 과습 방지가 중요합니다.",
        "temp_opt": (18.0, 23.0),
        "humid_opt": (60.0, 70.0),
        "moist_opt": (50.0, 65.0),
        "solar_opt": (300.0, 600.0),
        "mf": {
            "temp": {
                "Low": ("trapmf", [-10, 0, 13, 19]),
                "Optimal": ("trimf", [15, 20.5, 25]),
                "High": ("trapmf", [21, 26, 50, 60])
            },
            "humid": {
                "Low": ("trapmf", [-10, 0, 40, 58]),
                "Optimal": ("trimf", [50, 65, 78]),
                "High": ("trapmf", [68, 80, 100, 110])
            },
            "moist": {
                "Dry": ("trapmf", [-10, 0, 40, 58]),
                "Optimal": ("trimf", [48, 58, 68]),
                "Wet": ("trapmf", [58, 70, 100, 110])
            },
            "solar": {
                "Low": ("trapmf", [-10, 0, 100, 280]),
                "Moderate": ("trimf", [180, 450, 720]),
                "High": ("trapmf", [550, 750, 1200, 1300])
            }
        }
    },
    "파프리카": {
        "description": "따뜻한 고온 다습 환경을 선호하며 주간 22 - 27°C 유지가 중요합니다.",
        "temp_opt": (22.0, 27.0),
        "humid_opt": (65.0, 80.0),
        "moist_opt": (65.0, 80.0),
        "solar_opt": (450.0, 750.0),
        "mf": {
            "temp": {
                "Low": ("trapmf", [-10, 0, 18, 23]),
                "Optimal": ("trimf", [20, 25, 29]),
                "High": ("trapmf", [25, 29, 50, 60])
            },
            "humid": {
                "Low": ("trapmf", [-10, 0, 50, 65]),
                "Optimal": ("trimf", [58, 72, 85]),
                "High": ("trapmf", [72, 85, 100, 110])
            },
            "moist": {
                "Dry": ("trapmf", [-10, 0, 55, 72]),
                "Optimal": ("trimf", [62, 72, 84]),
                "Wet": ("trapmf", [72, 85, 100, 110])
            },
            "solar": {
                "Low": ("trapmf", [-10, 0, 200, 380]),
                "Moderate": ("trimf", [250, 550, 850]),
                "High": ("trapmf", [680, 880, 1200, 1300])
            }
        }
    },
    "상추": {
        "description": "저온성 채소로 15 - 20°C의 시원한 온도와 높은 토양수분을 선호합니다.",
        "temp_opt": (15.0, 20.0),
        "humid_opt": (60.0, 70.0),
        "moist_opt": (70.0, 85.0),
        "solar_opt": (250.0, 500.0),
        "mf": {
            "temp": {
                "Low": ("trapmf", [-10, 0, 10, 16]),
                "Optimal": ("trimf", [13, 17.5, 22]),
                "High": ("trapmf", [18, 23, 50, 60])
            },
            "humid": {
                "Low": ("trapmf", [-10, 0, 45, 58]),
                "Optimal": ("trimf", [50, 65, 78]),
                "High": ("trapmf", [68, 80, 100, 110])
            },
            "moist": {
                "Dry": ("trapmf", [-10, 0, 60, 78]),
                "Optimal": ("trimf", [68, 78, 88]),
                "Wet": ("trapmf", [78, 88, 100, 110])
            },
            "solar": {
                "Low": ("trapmf", [-10, 0, 100, 220]),
                "Moderate": ("trimf", [150, 380, 620]),
                "High": ("trapmf", [480, 680, 1200, 1300])
            }
        }
    }
}

# 하위 호환성을 위해 영문 이름 포함 키도 매핑
CROP_PRESETS["토마토 (Tomato)"] = CROP_PRESETS["토마토"]
CROP_PRESETS["딸기 (Strawberry)"] = CROP_PRESETS["딸기"]
CROP_PRESETS["파프리카 (Paprika)"] = CROP_PRESETS["파프리카"]
CROP_PRESETS["상추/엽채류 (Lettuce)"] = CROP_PRESETS["상추"]


class FuzzyGreenhouseSystem:
    """
    논문 기반 스마트 온실 퍼지 추론 시스템
    - 입력: 온도(°C), 습도(%), 토양수분(%), 일조량(W/m²)
    - 출력: 관수 제어값(0~100%), 환기 제어값(0~100%)
    - 비퍼지화: 무게중심법(Center of Gravity, COG)
    """

    def __init__(self, crop_name="토마토"):
        self.crop_name = crop_name
        self.crop_data = CROP_PRESETS.get(crop_name, CROP_PRESETS.get("토마토", list(CROP_PRESETS.values())[0]))
        self.mfs = self.crop_data["mf"]
        
        # 출력 변수 소속함수 정의 (0 ~ 100%)
        # 비퍼지화 정확도를 위해 -20부터 120까지 균일 격자 생성 후 [0, 100] 클리핑
        self.output_z = np.linspace(-20, 120, 701)
        self.irrigation_mfs = {
            "Off": trimf(self.output_z, [-20, 0, 20]),
            "Low": trimf(self.output_z, [10, 30, 50]),
            "Medium": trimf(self.output_z, [35, 55, 75]),
            "High": trapmf(self.output_z, [60, 85, 120, 140])
        }
        
        # 환기량 (Ventilation): Closed (0), Low (25), Medium (50), High (80+)
        self.ventilation_mfs = {
            "Closed": trimf(self.output_z, [-20, 0, 20]),
            "Low": trimf(self.output_z, [10, 30, 50]),
            "Medium": trimf(self.output_z, [35, 55, 75]),
            "High": trapmf(self.output_z, [60, 85, 120, 140])
        }

    def evaluate_mf(self, var_name, term, val):
        """특정 입력 변수의 특정 언어적 값(term)에 대한 소속도 계산"""
        func_type, params = self.mfs[var_name][term]
        if func_type == "trimf":
            return float(trimf(val, params))
        elif func_type == "trapmf":
            return float(trapmf(val, params))
        return 0.0

    def fuzzify(self, temp, humid, moist, solar):
        """크리스프 센서 입력값을 퍼지 소속도로 변환"""
        return {
            "temp": {
                "Low": self.evaluate_mf("temp", "Low", temp),
                "Optimal": self.evaluate_mf("temp", "Optimal", temp),
                "High": self.evaluate_mf("temp", "High", temp)
            },
            "humid": {
                "Low": self.evaluate_mf("humid", "Low", humid),
                "Optimal": self.evaluate_mf("humid", "Optimal", humid),
                "High": self.evaluate_mf("humid", "High", humid)
            },
            "moist": {
                "Dry": self.evaluate_mf("moist", "Dry", moist),
                "Optimal": self.evaluate_mf("moist", "Optimal", moist),
                "Wet": self.evaluate_mf("moist", "Wet", moist)
            },
            "solar": {
                "Low": self.evaluate_mf("solar", "Low", solar),
                "Moderate": self.evaluate_mf("solar", "Moderate", solar),
                "High": self.evaluate_mf("solar", "High", solar)
            }
        }

    def infer_and_defuzzify(self, temp, humid, moist, solar):
        """
        퍼지화 -> 전문가 IF-THEN 규칙 추론 -> COG 비퍼지화 수행
        반환값:
            - irrigation_val (float): 최종 관수량 제어값 (0~100%)
            - ventilation_val (float): 최종 환기량 제어값 (0~100%)
            - rule_results (list): 활성화된 규칙 및 발동 강도(firing strength)
            - aggregated_irr (ndarray): 비퍼지화 전 통합 관수 소속함수
            - aggregated_vent (ndarray): 비퍼지화 전 통합 환기 소속함수
            - fuzzed (dict): 각 센서별 소속도
        """
        fuzzed = self.fuzzify(temp, humid, moist, solar)
        
        t_l, t_o, t_h = fuzzed["temp"]["Low"], fuzzed["temp"]["Optimal"], fuzzed["temp"]["High"]
        h_l, h_o, h_h = fuzzed["humid"]["Low"], fuzzed["humid"]["Optimal"], fuzzed["humid"]["High"]
        m_d, m_o, m_w = fuzzed["moist"]["Dry"], fuzzed["moist"]["Optimal"], fuzzed["moist"]["Wet"]
        s_l, s_m, s_h = fuzzed["solar"]["Low"], fuzzed["solar"]["Moderate"], fuzzed["solar"]["High"]

        # 전문가 IF-THEN 규칙 정의 (논문 표 1 및 스마트 온실 도메인 지식 반영)
        rules = [
            # 논문 핵심 규칙 R1 ~ R4
            {
                "id": "R1a",
                "condition_str": "IF 토양수분 = 건조(Dry) AND 일조량 = 강함(High)",
                "weight": min(m_d, s_h),
                "irr_consequence": "High",
                "vent_consequence": None,
                "desc": "강한 일사 및 토양 건조 시 급격한 증발 방지를 위해 최대 관수"
            },
            {
                "id": "R1b",
                "condition_str": "IF 토양수분 = 건조(Dry) AND (일조량 = 보통 OR 낮음)",
                "weight": min(m_d, max(s_l, s_m)),
                "irr_consequence": "Medium",
                "vent_consequence": None,
                "desc": "일조가 완만할 때 토양 건조 시 적정 관수로 과습 방지"
            },
            {
                "id": "R2",
                "condition_str": "IF 토양수분 = 습함(Wet)",
                "weight": m_w,
                "irr_consequence": "Off",
                "vent_consequence": None,
                "desc": "토양이 과습하므로 관수 즉시 중단(뿌리 부패 예방)"
            },
            {
                "id": "R3",
                "condition_str": "IF 온도 = 높음(High) AND 습도 = 높음(High)",
                "weight": min(t_h, h_h),
                "irr_consequence": None,
                "vent_consequence": "High",
                "desc": "고온다습으로 인한 병해충 및 열해 방지(환기량 최대화)"
            },
            {
                "id": "R4",
                "condition_str": "IF 온도 = 적정(Optimal) AND 습도 = 적정(Optimal)",
                "weight": min(t_o, h_o),
                "irr_consequence": None,
                "vent_consequence": "Low",
                "desc": "생육 적정 환경이므로 최소 환기로 에너지 및 온실 내 온습도 보존"
            },
            # 복합 환경 대응 규칙 (R5 ~ R10)
            {
                "id": "R5",
                "condition_str": "IF 온도 = 높음(High) AND 일조량 = 강함(High)",
                "weight": min(t_h, s_h),
                "irr_consequence": None,
                "vent_consequence": "High",
                "desc": "강한 태양열과 고온 누적 시 환기팬 최대 가동"
            },
            {
                "id": "R6",
                "condition_str": "IF 온도 = 낮음(Low)",
                "weight": t_l,
                "irr_consequence": None,
                "vent_consequence": "Closed",
                "desc": "저온 시 냉해 방지를 위한 온실 밀폐 및 보온"
            },
            {
                "id": "R7",
                "condition_str": "IF 온도 = 높음(High) AND 습도 = 낮음(Low)",
                "weight": min(t_h, h_l),
                "irr_consequence": None,
                "vent_consequence": "Medium",
                "desc": "고온건조 완화를 위한 적정 환기"
            },
            {
                "id": "R8",
                "condition_str": "IF 온도 = 적정(Optimal) AND 습도 = 높음(High)",
                "weight": min(t_o, h_h),
                "irr_consequence": None,
                "vent_consequence": "Medium",
                "desc": "적정 온도를 유지하며 다습 상태 배출을 위한 보통 환기"
            },
            {
                "id": "R9",
                "condition_str": "IF 토양수분 = 적정(Optimal)",
                "weight": m_o,
                "irr_consequence": "Off",
                "vent_consequence": None,
                "desc": "토양수분이 최적 적정 범위 내에 있으므로 관수 밸브 잠금"
            },
            {
                "id": "R10",
                "condition_str": "IF 온도 = 낮음(Low) AND 습도 = 높음(High)",
                "weight": min(t_l, h_h),
                "irr_consequence": None,
                "vent_consequence": "Low",
                "desc": "저온 다습 시 결로 및 잿빛곰팡이 방지를 위한 간헐적 최소 환기"
            }
        ]

        # Mamdani 결합 (출력 소속도 Clipping & Union(Max))
        agg_irr = np.zeros_like(self.output_z)
        agg_vent = np.zeros_like(self.output_z)

        active_rules = []
        for r in rules:
            w = r["weight"]
            if w > 0.001:
                active_rules.append(r)
                
            if r["irr_consequence"] is not None and w > 0:
                clipped = np.minimum(w, self.irrigation_mfs[r["irr_consequence"]])
                agg_irr = np.maximum(agg_irr, clipped)
                
            if r["vent_consequence"] is not None and w > 0:
                clipped = np.minimum(w, self.ventilation_mfs[r["vent_consequence"]])
                agg_vent = np.maximum(agg_vent, clipped)

        # COG (Center of Gravity, 무게중심법) 비퍼지화 계산
        # 식: z* = ∫ z * mu(z) dz / ∫ mu(z) dz
        irr_sum = np.sum(agg_irr)
        if irr_sum > 1e-9:
            irr_val = float(np.clip(np.sum(self.output_z * agg_irr) / irr_sum, 0.0, 100.0))
        else:
            irr_val = 0.0

        vent_sum = np.sum(agg_vent)
        if vent_sum > 1e-9:
            vent_val = float(np.clip(np.sum(self.output_z * agg_vent) / vent_sum, 0.0, 100.0))
        else:
            vent_val = 0.0

        return {
            "irrigation_val": irr_val,
            "ventilation_val": vent_val,
            "active_rules": active_rules,
            "all_rules": rules,
            "agg_irr": agg_irr,
            "agg_vent": agg_vent,
            "output_z": self.output_z,
            "fuzzed": fuzzed
        }

    def generate_expert_diagnosis(self, temp, humid, moist, solar, irr_val, vent_val):
        """농업 전문가 관점의 현재 생육 상태 및 AI 제어 진단 메시지 생성"""
        diagnoses = []
        
        # 온도 진단
        t_min, t_max = self.crop_data["temp_opt"]
        if temp < t_min:
            diagnoses.append(f"[온도 낮음] 현재 기온({temp:.1f}℃)이 적정 하한({t_min}℃)보다 낮아 보온이 필요합니다.")
        elif temp > t_max:
            diagnoses.append(f"[온도 높음] 현재 기온({temp:.1f}℃)이 적정 상한({t_max}℃)보다 높아 환기 냉각이 필요합니다.")
        else:
            diagnoses.append(f"[온도 적정] 현재 기온({temp:.1f}℃)은 생육 적정 범위({t_min}~{t_max}℃) 내에서 양호합니다.")

        # 습도 진단
        h_min, h_max = self.crop_data["humid_opt"]
        if humid < h_min:
            diagnoses.append(f"[습도 건조] 공기 습도({humid:.1f}%)가 건조하여 기공 개폐 스트레스가 우려됩니다.")
        elif humid > h_max:
            diagnoses.append(f"[습도 과습] 공기 습도({humid:.1f}%)가 과습하여 곰팡이병 발생 위험이 있습니다.")
        else:
            diagnoses.append(f"[습도 적정] 공기 습도({humid:.1f}%)는 적정선({h_min}~{h_max}%)입니다.")

        # 토양수분 진단
        m_min, m_max = self.crop_data["moist_opt"]
        if moist < m_min:
            diagnoses.append(f"[수분 건조] 토양수분({moist:.1f}%)이 건조 상태입니다. 관수 밸브를 {irr_val:.1f}% 개방합니다.")
        elif moist > m_max:
            diagnoses.append(f"[수분 과습] 토양수분({moist:.1f}%)이 과습 상태입니다. 관수를 차단하고 통기를 유도합니다.")
        else:
            diagnoses.append(f"[수분 적정] 토양수분({moist:.1f}%)이 최적 상태({m_min}~{m_max}%)입니다.")

        # 종합 액션 요약
        action_summary = f"[제어 상태] 관수 밸브 {irr_val:.1f}% 개방 | 환기창/팬 {vent_val:.1f}% 가동"
        return diagnoses, action_summary
