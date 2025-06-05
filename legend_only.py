import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 감정 레이블과 색상
colors = ['red','blue','green','yellow','brown','indigo','black']
labels = ['ang','anx','emb','hap','hur','neu','sad']

# 범례 항목 만들기
legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]

# 새로운 그림과 축 생성
fig, ax = plt.subplots(figsize=(6, 1))
ax.axis('off')  # 축 제거

# 가로로 범례 배치
ax.legend(handles=legend_elements, loc='center', ncol=len(labels), fontsize=10, frameon=False)

# 저장
plt.savefig('legend_only.png', dpi=300, bbox_inches='tight')
plt.close()
