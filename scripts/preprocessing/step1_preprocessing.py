"""
Step 1: 파생변수 생성 및 전처리
=================================
대한민국 시군구 인구이동 네트워크 분석 프로젝트
분석 계획서 Step 1 구현

생성 파생변수:
  - 시차변수 (lag1): net_rate, in_rate, out_rate, pagerank, betweenness, closeness
  - 1차 차분 (d_): net_rate, pagerank
  - 이동평균 (ma3_): net_rate (3년 이동평균)
  - 더미변수: covid_dummy, metro_dummy, urban_dummy
  - 상호작용항: pagerank_x_covid, metro_x_pagerank
  - 로그변환: ln_pop_density, ln_biz_count, ln_worker_count
  - 표준화: 분석 트랙별 서브셋에서 별도 처리

결측 처리 전략:
  - 핵심변수 (결측률 ≤5%): 지역 내 선형 보간 (interpolate)
  - 확장변수 (결측률 5~30%): 지역 내 평균 대체 (within-group mean)
  - 고결측 변수 (>30%): 원본 유지 (ML 트랙 한정 활용)

출력 파일:
  - analysis_dataset_preprocessed.csv  : 전처리 완료 전체 데이터
  - track_A_2008_2025.csv              : 네트워크 기술 분석용 (2008~2025)
  - track_B_2009_2024.csv              : 패널 계량모형용 (2009~2024)
  - track_C_2017_2024.csv              : ML 확장 분석용 (2017~2024)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 0. 데이터 로드
# ─────────────────────────────────────────
print("=" * 60)
print("Step 1: 파생변수 생성 및 전처리")
print("=" * 60)

df = pd.read_csv('data/source/analysis_dataset_FINAL_v4.csv')
print(f"원본 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")
print(f"지역 수: {df['region_code'].nunique()}개 | 연도 범위: {df['year'].min()}~{df['year'].max()}")

# 정렬 (지역코드 → 연도 순)
df = df.sort_values(['region_code', 'year']).reset_index(drop=True)

# ─────────────────────────────────────────
# 1. 시차변수 생성 (Lag Variables)
# ─────────────────────────────────────────
print("\n[1] 시차변수 생성...")

lag_vars = ['net_rate', 'in_rate', 'out_rate',
            'pagerank', 'betweenness', 'closeness',
            'in_deg_cent', 'out_deg_cent', 'deg_cent',
            'fiscal_indep', 'employ_rate', 'aging_ratio',
            'youth_ratio', 'pop_density', 'doctor_per1000']

for var in lag_vars:
    if var in df.columns:
        df[f'{var}_lag1'] = df.groupby('region_code')[var].shift(1)

print(f"  생성된 시차변수: {len([v for v in lag_vars if v in df.columns])}개")

# ─────────────────────────────────────────
# 2. 1차 차분 (First Difference)
# ─────────────────────────────────────────
print("[2] 1차 차분 생성...")

diff_vars = ['net_rate', 'pagerank', 'betweenness', 'fiscal_indep',
             'employ_rate', 'aging_ratio', 'youth_ratio', 'pop_density']

for var in diff_vars:
    if var in df.columns:
        df[f'd_{var}'] = df.groupby('region_code')[var].diff(1)

print(f"  생성된 차분변수: {len([v for v in diff_vars if v in df.columns])}개")

# ─────────────────────────────────────────
# 3. 이동평균 (Moving Average, 3년)
# ─────────────────────────────────────────
print("[3] 3년 이동평균 생성...")

ma_vars = ['net_rate', 'pagerank', 'fiscal_indep']

for var in ma_vars:
    if var in df.columns:
        df[f'ma3_{var}'] = (df.groupby('region_code')[var]
                              .transform(lambda x: x.rolling(window=3, min_periods=2).mean()))

print(f"  생성된 이동평균변수: {len([v for v in ma_vars if v in df.columns])}개")

# ─────────────────────────────────────────
# 4. 더미변수 생성
# ─────────────────────────────────────────
print("[4] 더미변수 생성...")

# 코로나 더미 (2020~2022)
df['covid_dummy'] = df['year'].between(2020, 2022).astype(int)

# 수도권 더미 (서울·경기·인천)
metro_sgg = df[df['sido'].isin(['서울특별시', '경기도', '인천광역시'])]['region_code'].unique()
df['metro_dummy'] = df['region_code'].isin(metro_sgg).astype(int)

# 광역시 더미 (5대 광역시: 부산·대구·광주·대전·울산)
major_city_sgg = df[df['sido'].isin(['부산광역시', '대구광역시', '광주광역시',
                                      '대전광역시', '울산광역시'])]['region_code'].unique()
df['major_city_dummy'] = df['region_code'].isin(major_city_sgg).astype(int)

# 도시 규모 분류 (인구 기준)
# 대도시: 50만 이상, 중도시: 10~50만, 소도시: 10만 미만
pop_mean = df.groupby('region_code')['population'].mean()
df['city_size'] = df['region_code'].map(pop_mean).apply(
    lambda x: 'large' if x >= 500000 else ('medium' if x >= 100000 else 'small')
)
df['large_city_dummy'] = (df['city_size'] == 'large').astype(int)
df['small_city_dummy'] = (df['city_size'] == 'small').astype(int)

# 코로나 전후 기간 더미
df['pre_covid'] = (df['year'] < 2020).astype(int)
df['post_covid'] = (df['year'] > 2022).astype(int)

print(f"  수도권 지역 수: {df[df['metro_dummy']==1]['region_code'].nunique()}개")
print(f"  광역시 지역 수: {df[df['major_city_dummy']==1]['region_code'].nunique()}개")
print(f"  대도시(50만+): {df[df['large_city_dummy']==1]['region_code'].nunique()}개")
print(f"  소도시(10만-): {df[df['small_city_dummy']==1]['region_code'].nunique()}개")

# ─────────────────────────────────────────
# 5. 상호작용항 (Interaction Terms)
# ─────────────────────────────────────────
print("[5] 상호작용항 생성...")

# PageRank × COVID (팬데믹 기간 네트워크 효과 변화)
df['pagerank_x_covid'] = df['pagerank'] * df['covid_dummy']

# PageRank × 수도권 (수도권 네트워크 효과 이질성)
df['pagerank_x_metro'] = df['pagerank'] * df['metro_dummy']

# PageRank 시차 × COVID (선행 네트워크 위치의 팬데믹 효과)
df['pagerank_lag1_x_covid'] = df['pagerank_lag1'] * df['covid_dummy']

# 고령화율 × 재정자립도 (지역 역량 상호작용)
if 'aging_ratio' in df.columns and 'fiscal_indep' in df.columns:
    df['aging_x_fiscal'] = df['aging_ratio'] * df['fiscal_indep']

print("  상호작용항 생성 완료: pagerank_x_covid, pagerank_x_metro, pagerank_lag1_x_covid, aging_x_fiscal")

# ─────────────────────────────────────────
# 6. 로그변환
# ─────────────────────────────────────────
print("[6] 로그변환 생성...")

log_vars = {
    'pop_density': 'ln_pop_density',
    'population': 'ln_population',
    'doctor_per1000': 'ln_doctor',
    'area_km2': 'ln_area',
}

for src, dst in log_vars.items():
    if src in df.columns:
        df[dst] = np.log1p(df[src].clip(lower=0))

# 사업체수·종사자수 (결측 많지만 있는 값은 변환)
if 'biz_count' in df.columns:
    df['ln_biz_count'] = np.log1p(df['biz_count'].clip(lower=0))
if 'worker_count' in df.columns:
    df['ln_worker_count'] = np.log1p(df['worker_count'].clip(lower=0))

print(f"  로그변환 변수: {len(log_vars)}개 + biz/worker")

# ─────────────────────────────────────────
# 7. 결측 처리
# ─────────────────────────────────────────
print("\n[7] 결측 처리...")

# 결측률 계산
miss_rate = df.isnull().mean().sort_values(ascending=False)

# 핵심변수 (결측률 ≤5%): 지역 내 선형 보간
core_vars = miss_rate[miss_rate <= 0.05].index.tolist()
# 식별자 및 더미는 제외
exclude = ['region_code', 'year', 'sido', 'sgg_name', 'name', 'city_size',
           'covid_dummy', 'metro_dummy', 'major_city_dummy',
           'large_city_dummy', 'small_city_dummy', 'pre_covid', 'post_covid']
core_vars = [v for v in core_vars if v not in exclude and df[v].dtype in [float, int, 'float64', 'int64']]

interp_count = 0
for var in core_vars:
    before = df[var].isnull().sum()
    df[var] = df.groupby('region_code')[var].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    after = df[var].isnull().sum()
    if before > after:
        interp_count += 1

print(f"  선형 보간 적용 변수: {interp_count}개 (결측률 ≤5%)")

# 확장변수 (결측률 5~30%): 지역 내 평균 대체
ext_vars = miss_rate[(miss_rate > 0.05) & (miss_rate <= 0.30)].index.tolist()
ext_vars = [v for v in ext_vars if v not in exclude and df[v].dtype in [float, int, 'float64', 'int64']]

mean_fill_count = 0
for var in ext_vars:
    before = df[var].isnull().sum()
    df[var] = df.groupby('region_code')[var].transform(
        lambda x: x.fillna(x.mean())
    )
    after = df[var].isnull().sum()
    if before > after:
        mean_fill_count += 1

print(f"  지역 내 평균 대체 변수: {mean_fill_count}개 (결측률 5~30%)")
print(f"  원본 유지 변수: 결측률 >30% 변수 (ML 트랙 한정)")

# ─────────────────────────────────────────
# 8. 전처리 완료 데이터 저장
# ─────────────────────────────────────────
print("\n[8] 전처리 완료 데이터 저장...")

df.to_csv('data/derived/analysis_dataset_preprocessed.csv', index=False, encoding='utf-8-sig')
print(f"  저장: analysis_dataset_preprocessed.csv ({df.shape[0]:,}행 × {df.shape[1]}열)")

# ─────────────────────────────────────────
# 9. 3트랙 서브셋 생성
# ─────────────────────────────────────────
print("\n[9] 3트랙 서브셋 생성...")

# Track A: 네트워크 기술 분석 (2008~2025, 전체)
track_a = df[(df['year'] >= 2008) & (df['year'] <= 2025)].copy()
track_a.to_csv('data/derived/track_A_2008_2025.csv', index=False, encoding='utf-8-sig')
print(f"  Track A (2008~2025): {track_a.shape[0]:,}행 × {track_a.shape[1]}열")

# Track B: 패널 계량모형 (2009~2024, 핵심변수 완비)
# 핵심 통제변수 목록 (결측 없어야 하는 변수)
core_required = ['net_rate', 'pagerank', 'fiscal_indep', 'aging_ratio',
                 'youth_ratio', 'doctor_per1000', 'fertility',
                 'nat_increase', 'seoul_dist_km', 'population']
core_required = [v for v in core_required if v in df.columns]

track_b = df[(df['year'] >= 2009) & (df['year'] <= 2024)].copy()
track_b_clean = track_b.dropna(subset=core_required)
track_b_clean.to_csv('data/derived/track_B_2009_2024.csv', index=False, encoding='utf-8-sig')
print(f"  Track B (2009~2024): {track_b_clean.shape[0]:,}행 × {track_b_clean.shape[1]}열")
print(f"    (전체 {track_b.shape[0]:,}행 중 핵심변수 완비: {track_b_clean.shape[0]:,}행, "
      f"제외: {track_b.shape[0]-track_b_clean.shape[0]}행)")

# Track C: ML 확장 분석 (2017~2024, 확장변수 포함)
ext_required = core_required + ['employ_rate', 'biz_count', 'childcare_pk',
                                  'culture_facility_count', 'road_pavement_rate']
ext_required = [v for v in ext_required if v in df.columns]

track_c = df[(df['year'] >= 2017) & (df['year'] <= 2024)].copy()
track_c.to_csv('data/derived/track_C_2017_2024.csv', index=False, encoding='utf-8-sig')
print(f"  Track C (2017~2024): {track_c.shape[0]:,}행 × {track_c.shape[1]}열")

# ─────────────────────────────────────────
# 10. 전처리 요약 통계
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("전처리 완료 요약")
print("=" * 60)

new_vars = [c for c in df.columns if any(c.startswith(p) for p in
            ['net_rate_lag', 'in_rate_lag', 'out_rate_lag',
             'pagerank_lag', 'betweenness_lag', 'closeness_lag',
             'd_', 'ma3_', 'covid_', 'metro_', 'major_city_', 'large_city_',
             'small_city_', 'pre_covid', 'post_covid',
             'pagerank_x_', 'aging_x_', 'pagerank_lag1_x_',
             'ln_pop', 'ln_doctor', 'ln_area', 'ln_biz', 'ln_worker',
             'city_size', 'ln_population'])]

print(f"\n신규 생성 변수 수: {len(new_vars)}개")
print(f"최종 전체 변수 수: {df.shape[1]}개")
print(f"\n최종 결측률 현황 (상위 10개):")
final_miss = df.isnull().mean().sort_values(ascending=False).head(10)
for var, rate in final_miss.items():
    if rate > 0:
        print(f"  {var:<35}: {rate*100:.1f}%")

print(f"\n트랙별 데이터 규모:")
print(f"  Track A (네트워크 기술): {track_a.shape[0]:,}행 (229개 × 18년)")
print(f"  Track B (패널 계량모형): {track_b_clean.shape[0]:,}행")
print(f"  Track C (ML 확장):       {track_c.shape[0]:,}행")
print("\n전처리 완료!")
