"""
스마트 온실 환경 피드백 시뮬레이션 및 On-Off 제어기 비교 모듈
논문 3장 "피드백 및 기대 효과" 모델 구현:
- 센서 측정 -> 제어 출력 -> 온실 물리 환경 변화 -> 재측정 피드백 루프
- 퍼지 추론 제어기 vs 전통 고정 임계값(On-Off) 벤치마크 제어기
- 평가 지표: 물 사용량, 목표 토양수분 오차(RMSE), 최적 생육 환경 유지 시간(%), 제어 헌팅(변동성)
"""

import numpy as np
import pandas as pd
from fuzzy_engine import FuzzyGreenhouseSystem, CROP_PRESETS

class SimpleOnOffController:
    """전통 고정 임계값(On-Off) 온실 제어기"""
    def __init__(self, crop_name="토마토 (Tomato)"):
        preset = CROP_PRESETS.get(crop_name, CROP_PRESETS["토마토 (Tomato)"])
        self.t_min, self.t_max = preset["temp_opt"]
        self.m_min, self.m_max = preset["moist_opt"]
        self.irr_state = False
        self.vent_state = False

    def step(self, temp, humid, moist, solar):
        # 토양수분이 하한 이하로 떨어지면 100% 펌프 가동, 상한 도달 시 정지
        if moist < self.m_min:
            self.irr_state = True
        elif moist >= self.m_max:
            self.irr_state = False
        irr_val = 100.0 if self.irr_state else 0.0

        # 온도가 상한 초과 시 100% 환기팬 가동, 최적 중간값 이하로 내려가면 정지
        t_target = (self.t_min + self.t_max) / 2.0
        if temp > self.t_max:
            self.vent_state = True
        elif temp <= t_target:
            self.vent_state = False
        vent_val = 100.0 if self.vent_state else 0.0

        return irr_val, vent_val


def generate_weather_profile(hours=24, scenario="맑은 날 (Sunny)", dt_minutes=10):
    """
    24~48시간 외부 기상 프로파일 생성 (일조량, 외기온도, 외기습도)
    시나리오:
      - '맑은 날 (Sunny)': 강한 일사, 주간 고온, 주간 건조
      - '흐린 날 (Cloudy)': 약한 일사, 온화한 온도, 다습
      - '폭염/건조 (Heatwave)': 매우 강한 일사, 극고온, 극건조
    """
    n_steps = int(hours * 60 / dt_minutes)
    times = [i * dt_minutes / 60.0 for i in range(n_steps)]
    
    t_out = np.zeros(n_steps)
    solar = np.zeros(n_steps)
    h_out = np.zeros(n_steps)
    
    for i, t in enumerate(times):
        day_hour = t % 24.0
        
        # 일조량: 06시~19시 해가 뜸, 12.5시에 피크
        if 6.0 <= day_hour <= 19.0:
            sin_val = np.sin((day_hour - 6.0) / 13.0 * np.pi)
            if scenario == "맑은 날 (Sunny)":
                solar[i] = max(0.0, sin_val * 850.0 + np.random.normal(0, 15))
            elif scenario == "흐린 날 (Cloudy)":
                solar[i] = max(0.0, sin_val * 350.0 + np.random.normal(0, 20))
            elif scenario == "폭염/건조 (Heatwave)":
                solar[i] = max(0.0, sin_val * 1050.0 + np.random.normal(0, 15))
        else:
            solar[i] = 0.0

        # 외기온도: 14시경 최고, 05시경 최저
        temp_cycle = -np.cos((day_hour - 5.0) / 24.0 * 2 * np.pi)  # -1 at 5am, +1 at 17pm
        if scenario == "맑은 날 (Sunny)":
            t_base, t_amp = 18.0, 10.0  # 8°C ~ 28°C
        elif scenario == "흐린 날 (Cloudy)":
            t_base, t_amp = 16.0, 5.0   # 11°C ~ 21°C
        elif scenario == "폭염/건조 (Heatwave)":
            t_base, t_amp = 26.0, 11.0  # 15°C ~ 37°C
        t_out[i] = t_base + t_amp * (temp_cycle + 1) / 2.0 + np.random.normal(0, 0.3)

        # 외기습도: 새벽에 높고 낮에 낮음
        humid_cycle = np.cos((day_hour - 5.0) / 24.0 * 2 * np.pi)
        if scenario == "맑은 날 (Sunny)":
            h_base, h_amp = 45.0, 35.0  # 45% ~ 80%
        elif scenario == "흐린 날 (Cloudy)":
            h_base, h_amp = 65.0, 25.0  # 65% ~ 90%
        elif scenario == "폭염/건조 (Heatwave)":
            h_base, h_amp = 30.0, 30.0  # 30% ~ 60%
        h_out[i] = np.clip(h_base + h_amp * (humid_cycle + 1) / 2.0 + np.random.normal(0, 0.5), 15.0, 95.0)

    return times, t_out, h_out, solar


def run_greenhouse_simulation(crop_name="토마토 (Tomato)", hours=24, scenario="맑은 날 (Sunny)", dt_minutes=10):
    """
    퍼지 제어기와 On-Off 제어기를 동일 환경에서 동시 시뮬레이션
    피드백 루프: 센서 측정 -> 제어값 산출 -> 온실 상태 업데이트
    """
    times, t_out_arr, h_out_arr, solar_arr = generate_weather_profile(hours, scenario, dt_minutes)
    n_steps = len(times)
    preset = CROP_PRESETS.get(crop_name, CROP_PRESETS["토마토 (Tomato)"])
    
    target_temp = (preset["temp_opt"][0] + preset["temp_opt"][1]) / 2.0
    target_moist = (preset["moist_opt"][0] + preset["moist_opt"][1]) / 2.0
    target_humid = (preset["humid_opt"][0] + preset["humid_opt"][1]) / 2.0

    # 초기 상태 (동일하게 시작)
    init_temp = float(t_out_arr[0] + 2.0)
    init_humid = float(h_out_arr[0] - 5.0)
    init_moist = float(target_moist - 8.0)  # 약간 건조한 상태에서 시작

    fuzzy_sys = FuzzyGreenhouseSystem(crop_name)
    onoff_sys = SimpleOnOffController(crop_name)

    # 데이터 저장소
    fuzzy_records = []
    onoff_records = []

    # 상태 변수
    f_temp, f_humid, f_moist = init_temp, init_humid, init_moist
    o_temp, o_humid, o_moist = init_temp, init_humid, init_moist

    dt_hr = dt_minutes / 60.0

    for step in range(n_steps):
        t_now = times[step]
        tout = t_out_arr[step]
        hout = h_out_arr[step]
        sol = solar_arr[step]

        # 1. Fuzzy Controller Step
        f_res = fuzzy_sys.infer_and_defuzzify(f_temp, f_humid, f_moist, sol)
        f_irr = f_res["irrigation_val"]
        f_vent = f_res["ventilation_val"]

        fuzzy_records.append({
            "time": t_now,
            "temp": f_temp,
            "humid": f_humid,
            "moist": f_moist,
            "solar": sol,
            "irrigation": f_irr,
            "ventilation": f_vent,
            "tout": tout,
            "hout": hout
        })

        # 2. On-Off Controller Step
        o_irr, o_vent = onoff_sys.step(o_temp, o_humid, o_moist, sol)

        onoff_records.append({
            "time": t_now,
            "temp": o_temp,
            "humid": o_humid,
            "moist": o_moist,
            "solar": sol,
            "irrigation": o_irr,
            "ventilation": o_vent,
            "tout": tout,
            "hout": hout
        })

        # 3. Dynamic Greenhouse Physics Step (논문 3장 피드백 모델)
        # 물리 파라미터 (시간당 변화율 계수)
        # 관수 효과: 관수 100% 1시간 지속 시 토양수분 약 +25% 상승
        k_irr = 0.28 * dt_hr
        # 증발산: 온도와 일조량에 비례하여 수분 손실
        evap_rate_base = 0.04 * dt_hr
        
        # 토양수분 업데이트
        f_evap = evap_rate_base * (f_temp / 20.0) * (1.0 + sol / 400.0)
        f_moist = np.clip(f_moist + k_irr * f_irr - f_evap * 15.0, 0.0, 100.0)

        o_evap = evap_rate_base * (o_temp / 20.0) * (1.0 + sol / 400.0)
        o_moist = np.clip(o_moist + k_irr * o_irr - o_evap * 15.0, 0.0, 100.0)

        # 온도 업데이트: 일사열 흡수, 외기와의 전도, 환기 냉각
        solar_gain = (sol / 1000.0) * 12.0 * dt_hr
        loss_coef = 0.45 * dt_hr
        vent_cool_coef = 1.1 * dt_hr

        # Fuzzy temp update
        f_cooling = (f_vent / 100.0) * vent_cool_coef * max(0.0, f_temp - tout)
        f_temp = f_temp + solar_gain + loss_coef * (tout - f_temp) - f_cooling

        # OnOff temp update
        o_cooling = (o_vent / 100.0) * vent_cool_coef * max(0.0, o_temp - tout)
        o_temp = o_temp + solar_gain + loss_coef * (tout - o_temp) - o_cooling

        # 습도 업데이트: 환기 배출, 증산작용 증가
        f_transp = (f_moist / 100.0) * (0.8 + sol / 500.0) * 5.0 * dt_hr
        f_h_vent = (f_vent / 100.0) * 1.2 * dt_hr * (f_humid - hout)
        f_humid = np.clip(f_humid + f_transp - f_h_vent, 20.0, 98.0)

        o_transp = (o_moist / 100.0) * (0.8 + sol / 500.0) * 5.0 * dt_hr
        o_h_vent = (o_vent / 100.0) * 1.2 * dt_hr * (o_humid - hout)
        o_humid = np.clip(o_humid + o_transp - o_h_vent, 20.0, 98.0)

    df_fuzzy = pd.DataFrame(fuzzy_records)
    df_onoff = pd.DataFrame(onoff_records)

    # 4. 성능 지표(KPI) 산출
    # 1) 물 사용량 (총 관수 제어값 적분, % * 시간)
    water_fuzzy = float(df_fuzzy["irrigation"].sum() * dt_hr)
    water_onoff = float(df_onoff["irrigation"].sum() * dt_hr)
    water_saving_pct = ((water_onoff - water_fuzzy) / max(water_onoff, 1e-5)) * 100.0

    # 2) 목표 토양수분과의 오차 (RMSE)
    rmse_fuzzy = float(np.sqrt(np.mean((df_fuzzy["moist"] - target_moist) ** 2)))
    rmse_onoff = float(np.sqrt(np.mean((df_onoff["moist"] - target_moist) ** 2)))

    # 3) 목표 환경 범위 내에서 유지된 시간 비율 (%)
    t_min, t_max = preset["temp_opt"]
    m_min, m_max = preset["moist_opt"]
    
    fuzzy_in_range = ((df_fuzzy["temp"] >= t_min) & (df_fuzzy["temp"] <= t_max) & 
                      (df_fuzzy["moist"] >= m_min) & (df_fuzzy["moist"] <= m_max)).mean() * 100.0
    onoff_in_range = ((df_onoff["temp"] >= t_min) & (df_onoff["temp"] <= t_max) & 
                      (df_onoff["moist"] >= m_min) & (df_onoff["moist"] <= m_max)).mean() * 100.0

    # 4) 환기팬/밸브 스위칭 헌팅(변동성, 총 절대 변화량)
    hunting_fuzzy = float(np.abs(np.diff(df_fuzzy["irrigation"])).sum() + np.abs(np.diff(df_fuzzy["ventilation"])).sum())
    hunting_onoff = float(np.abs(np.diff(df_onoff["irrigation"])).sum() + np.abs(np.diff(df_onoff["ventilation"])).sum())

    metrics = {
        "water_fuzzy": water_fuzzy,
        "water_onoff": water_onoff,
        "water_saving_pct": water_saving_pct,
        "rmse_fuzzy": rmse_fuzzy,
        "rmse_onoff": rmse_onoff,
        "rmse_improvement_pct": ((rmse_onoff - rmse_fuzzy) / rmse_onoff) * 100.0,
        "fuzzy_in_range_pct": fuzzy_in_range,
        "onoff_in_range_pct": onoff_in_range,
        "hunting_fuzzy": hunting_fuzzy,
        "hunting_onoff": hunting_onoff,
        "target_temp": target_temp,
        "target_moist": target_moist,
        "crop_name": crop_name,
        "scenario": scenario
    }

    return df_fuzzy, df_onoff, metrics
