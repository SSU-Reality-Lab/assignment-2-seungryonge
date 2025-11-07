import os, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import features

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def save_heatmap(array, title, filename, cmap='jet'):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_keypoints(image, keypoints, filename):
    vis = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), 2, (0,255,0), -1)
    os.makedirs('results', exist_ok=True)
    cv2.imwrite(filename, vis)

# -------------------------------------------------------------------
# 0️⃣ Load Images
# -------------------------------------------------------------------
img1 = cv2.imread('resources/yosemite1.jpg')
img2 = cv2.imread('resources/yosemite2.jpg')

gray1 = cv2.cvtColor(img1.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------
# 1️⃣ Feature Computation (TODO1~6)
# -------------------------------------------------------------------
HKD = features.HarrisKeypointDetector()
SFD = features.SimpleFeatureDescriptor()
MFD = features.MOPSFeatureDescriptor()

# TODO1
a1, b1 = HKD.computeHarrisValues(gray1)
a2, b2 = HKD.computeHarrisValues(gray2)

# TODO3
d1 = HKD.detectKeypoints(img1)
d2 = HKD.detectKeypoints(img2)

# Filter weak keypoints
d1 = [kp for kp in d1 if kp.response > 0.01]
d2 = [kp for kp in d2 if kp.response > 0.01]

# TODO4~6
desc_simple_1 = SFD.describeFeatures(img1, d1)
desc_simple_2 = SFD.describeFeatures(img2, d2)
desc_mops_1 = MFD.describeFeatures(img1, d1)
desc_mops_2 = MFD.describeFeatures(img2, d2)

# -------------------------------------------------------------------
# 2️⃣ Visualization (TODO1, TODO3)
# -------------------------------------------------------------------
save_heatmap(a1, "Image1 - TODO1 Harris Strength", "results/img1_TODO1_harris_strength.png")
save_heatmap(a2, "Image2 - TODO1 Harris Strength", "results/img2_TODO1_harris_strength.png")

save_keypoints(img1, d1, "results/img1_TODO3_detected_keypoints.png")
save_keypoints(img2, d2, "results/img2_TODO3_detected_keypoints.png")

print("✅ Saved TODO1 & TODO3 visualizations.")

# -------------------------------------------------------------------
# 3️⃣ Matching (TODO7 - SSD, TODO8 - Ratio)
# -------------------------------------------------------------------
matcher_ssd = features.SSDFeatureMatcher()
matcher_ratio = features.RatioFeatureMatcher()

# ------------------------------
# TODO7 - SSD matching
# ------------------------------
# Step 1. SSD matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ssd = matcher_ssd.matchFeatures(desc_mops_1,desc_mops_2)

# Step 2. 거리(distance)를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ssd = sorted(matches_ssd, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ssd_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ssd, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO7_SSD_matches.png", ssd_vis)
print("✅ TODO7 (SSD) match result saved → results/TODO7_SSD_matches.png")

# ------------------------------
# TODO8 - Ratio matching
# ------------------------------
# Step 1. Ratio matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ratio = matcher_ratio.matchFeatures(desc_mops_1,desc_mops_2)

# Step 2. distance를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ratio = sorted(matches_ratio, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ratio_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ratio, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO8_Ratio_matches.png", ratio_vis)
print("✅ TODO8 (Ratio) match result saved → results/TODO8_Ratio_matches.png")

print("🎯 All TODO1–8 visualizations done! Files saved in 'results/'")

# [TODO 8단계 매칭이 잘된 이유]
#
# features.py의 `SSDFeatureMatcher` (TODO 7)는 `matchFeatures` 함수 내에서 `np.argmin(dist[i])`를 사용해 1번 이미지의 각 특징점(i)에 대해 2번 이미지에서 SSD 거리가 '가장 작은 단 하나의 인덱스(min_dist)'를 찾습니다.
# 그리고 `match.distance`에 그 최소 거리(SSD) 값을 저장합니다.이 방식은 1순위와 2순위의 거리 차이가 0.01이든 100.0이든 상관없이 '무조건 1순위'만 선택하므로,1순위와 2순위가 비등비등한(모호한) 매칭도 결과에 포함시킵니다.
#
# 반면, `RatioFeatureMatcher` (TODO 8)는 `matchFeatures` 함수 내에서 np.argsort(dist[i])`를 사용해 1번 특징점(i)에 대한 '모든 거리 순위'를 계산합니다.
# 이를 통해 1순위 인덱스(`sort_Idx[0]`)와 2순위 인덱스(`sort_Idx[1]`)를 모두 가져와, 각각의 거리 `SSD1`과 `SSD2`를 얻습니다.
#
# 핵심은 `match.distance = SSD1 / (SSD2 * 1.0)` 코드입니다. TODO 8은 `match.distance`에 '최소 거리(SSD)'가 아닌, '1순위와 2순위의 거리 비율(Ratio)'을 저장합니다.
#
# `tests.py`에서 이 `match.distance`를 기준으로 `sorted`를 수행하면, TODO 7은 '단순히 SSD 거리가 가까운 순'으로 정렬되지만, TODO 8은 '1순위가 2순위보다 압도적으로 좋은(즉, Ratio가 낮은) 순'으로 정렬됩니다.
#
# 따라서 TODO 8의 방식이 1, 2순위가 비슷한 모호한 매칭(Ratio가 1에 가까움)을 효과적으로 걸러내고, 신뢰할 수 있는(Ratio가 낮음) 매칭만 상위로 보여주므로 훨씬 더 정확한 결과를 제공합니다.