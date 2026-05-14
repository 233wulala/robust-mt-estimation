import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.colorbar as mcbar
import matplotlib.colors as mcolors

# ── 读取数据 ──────────────────────────────────────────────
df_Z = pd.read_csv("Z_impedance_70_5.csv")
df_w_x = pd.read_csv("weights_out_s/weights_0.005524_x_s_70_5.csv")
df_w_y = pd.read_csv("weights_out_s/weights_0.005524_y_s_70_5.csv")

numeric_cols = ["freq", "period", "Zxx_real", "Zxx_imag", "Zxy_real", "Zxy_imag",
                "Zyx_real", "Zyx_imag", "Zyy_real", "Zyy_imag"]
for col in numeric_cols:
    df_Z[col] = pd.to_numeric(df_Z[col], errors='coerce')
df_Z = df_Z.dropna(subset=["freq"])

# 取目标频率数据
TARGET_FREQ = 0.005524
df_f = df_Z[np.isclose(df_Z["freq"], TARGET_FREQ, atol=1e-6)].reset_index(drop=True)

# 提取权重
w_x = df_w_x["weight"].values
w_y = df_w_y["weight"].values
n = min(len(df_f), len(w_x), len(w_y))
df_f = df_f.iloc[:n]
wxx = w_x[:n]
wxy = w_x[:n]
wyx = w_y[:n]
wyy = w_y[:n]

print(f"数据点数: {n}")
print(f"权重范围 - Zx: [{wxy.min():.3f}, {wxy.max():.3f}]")
print(f"权重范围 - Zy: [{wyx.min():.3f}, {wyx.max():.3f}]")

# 提取复数阻抗
Zxx = df_f["Zxx_real"].values + 1j * df_f["Zxx_imag"].values
Zxy = df_f["Zxy_real"].values + 1j * df_f["Zxy_imag"].values
Zyx = df_f["Zyx_real"].values + 1j * df_f["Zyx_imag"].values
Zyy = df_f["Zyy_real"].values + 1j * df_f["Zyy_imag"].values
freq = df_f["freq"].values
# period = 1 / freq

# 计算幅值和相位
amp_xx_raw = np.abs(Zxx)
amp_xy_raw = np.abs(Zxy)
amp_yx_raw = np.abs(Zyx)
amp_yy_raw = np.abs(Zyy)
# ⭐ 让 Zxy 显示在与 Zyx 相反的相位上（相位差 180°）
# 等价于把 Zxy 取负后再求相位，使 phi_xy = phi(Zxy) + 180°（自动归一化到 [-180, 180]）
phi_xx = np.angle(-Zxx, deg=True)
phi_xy = np.angle(-Zxy, deg=True)
phi_yx = np.angle(Zyx, deg=True)
phi_yy = np.angle(Zyy, deg=True)

# ── ⭐ Log 幅值 ────────────────────────────────────────────
all_amp_abs = np.concatenate([amp_xx_raw, amp_xy_raw, amp_yx_raw, amp_yy_raw])

# 根据数据的98/2百分位确定 log 坐标上下界
log_amp_max_val = np.percentile(all_amp_abs, 98)
log_amp_min_val = max(np.percentile(all_amp_abs, 2), 1e-3)

log_max = np.ceil(np.log10(log_amp_max_val))    # e.g. 3 → 最大圆对应 10^3
log_min = np.floor(np.log10(log_amp_min_val))   # e.g. 0 → 最小圆对应 10^0

print(f"log 幅值范围: 10^{log_min:.0f} ~ 10^{log_max:.0f}")

# ⭐ 绘图半径 = log10(amp) 归一化到 [0, 1]
def to_plot_r(amp_abs, log_min, log_max):
    log_a = np.log10(np.clip(amp_abs, 10**log_min, 10**log_max))
    return (log_a - log_min) / (log_max - log_min)

r_xx = to_plot_r(amp_xx_raw, log_min, log_max)
r_xy = to_plot_r(amp_xy_raw, log_min, log_max)
r_yx = to_plot_r(amp_yx_raw, log_min, log_max)
r_yy = to_plot_r(amp_yy_raw, log_min, log_max)
r_plot_max = 1.0

# 极坐标 → 直角
def polar_to_cartesian(r, phase_deg):
    phase_rad = np.deg2rad(phase_deg)
    return r * np.cos(phase_rad), r * np.sin(phase_rad)

x_xx, y_xx = polar_to_cartesian(r_xx, phi_xx)
x_xy, y_xy = polar_to_cartesian(r_xy, phi_xy)
x_yx, y_yx = polar_to_cartesian(r_yx, phi_yx)
x_yy, y_yy = polar_to_cartesian(r_yy, phi_yy)

# ── 权重 → 透明度 ─────────────────────────────────────────
def weight_to_alpha(weights, min_alpha=0.5, max_alpha=1.0):
    w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    return min_alpha + (max_alpha - min_alpha) * w_norm

alpha_xx = weight_to_alpha(wxx)
alpha_xy = weight_to_alpha(wxy)
alpha_yx = weight_to_alpha(wyx)
alpha_yy = weight_to_alpha(wyy)

def create_rgba(base_color, alphas):
    c = np.zeros((len(alphas), 4))
    c[:, :3] = base_color
    c[:, 3] = alphas
    return c

colors_xx = create_rgba((0.85, 0.15, 0.15), alpha_xx)
colors_xy = create_rgba((0.85, 0.15, 0.15), alpha_xy)
colors_yx = create_rgba((0.15, 0.25, 0.85), alpha_yx)
colors_yy = create_rgba((0.15, 0.25, 0.85), alpha_yy)

# ── 绘图布局 ─────────────────────────────────────────────
fig = plt.figure(figsize=(10, 9), facecolor='white')
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.07], figure=fig)
gs.update(left=0.05, right=0.88, top=0.93, bottom=0.07, wspace=0.02)

ax = fig.add_subplot(gs[0])
ax.set_facecolor('white')
ax.set_aspect('equal')

# ── 极坐标网格（Log 刻度）────────────────────────────────
# 每半个 decade 一圈：整 decade 实线，半 decade 虚线
tick_logs = np.arange(log_min, log_max + 0.01, 0.5)
tick_logs = tick_logs[(tick_logs >= log_min) & (tick_logs <= log_max)]

for tl in tick_logs:
    r_plot = (tl - log_min) / (log_max - log_min)
    lw = 1.3 if (tl == log_min or tl == log_max) else 0.8
    ls = '-' if (tl % 1 == 0) else '--'
    circle = Circle((0, 0), r_plot, fill=False, color='#FFAAAA',
                    linewidth=lw, linestyle=ls, zorder=1)
    ax.add_patch(circle)

# 辐射线（每30度）
for angle_deg in range(0, 360, 30):
    angle_rad = np.deg2rad(angle_deg)
    ax.plot([0, r_plot_max * np.cos(angle_rad)],
            [0, r_plot_max * np.sin(angle_rad)],
            color='#FFCCCC', linewidth=0.7, zorder=1)

# 角度标签
label_r = r_plot_max * 1.1
for angle_deg in [0, 30, 60, 90, 120, 150, 180, -150, -120, -90, -60, -30]:
    angle_rad = np.deg2rad(angle_deg)
    ax.text(label_r * np.cos(angle_rad), label_r * np.sin(angle_rad),
            f'{angle_deg}°', ha='center', va='center', fontsize=8.5, color='#555')

# ⭐ 幅值标签：显示 log10 值（如 "2" 代表 10^2=100）
label_angle_rad = np.deg2rad(52)
for tl in tick_logs:
    r_plot = (tl - log_min) / (log_max - log_min)
    x_lbl = r_plot * np.cos(label_angle_rad)
    y_lbl = r_plot * np.sin(label_angle_rad)
    lbl = f'{tl:.0f}' if tl == int(tl) else f'{tl:.1f}'
    ax.text(x_lbl, y_lbl, lbl, fontsize=7.5, color='#AA4444',
            ha='center', va='bottom',
            bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.7))

# log10(ρ) 单位标注
ax.text(r_plot_max * np.cos(label_angle_rad) * 1.04,
        r_plot_max * np.sin(label_angle_rad) * 1.04,
        r'log$_{10}$($\rho$)', fontsize=8, color='#AA4444',
        ha='center', va='bottom')

# ── 散点绘制（按权重排序，低权重先画）───────────────────
order_xx = np.argsort(alpha_xx)
order_xy = np.argsort(alpha_xy)
order_yx = np.argsort(alpha_yx)
order_yy = np.argsort(alpha_yy)

ax.scatter(x_xx[order_xx], y_xx[order_xx],
           c=colors_xx[order_xx], s=18, marker='^',
           edgecolors='none', zorder=3)
ax.scatter(x_xy[order_xy], y_xy[order_xy],
           c=colors_xy[order_xy], s=18, marker='s',
           edgecolors='none', zorder=3)
ax.scatter(x_yx[order_yx], y_yx[order_yx],
           c=colors_yx[order_yx], s=18, marker='s',
           edgecolors='none', zorder=3)
ax.scatter(x_yy[order_yy], y_yy[order_yy],
           c=colors_yy[order_yy], s=18, marker='^',
           edgecolors='none', zorder=3)


# ── 坐标轴设置 ────────────────────────────────────────────
margin = r_plot_max * 1.22
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(f'Apparent Resistivity  |  f = {TARGET_FREQ} Hz  |  T = {1/TARGET_FREQ:.1f} s',
             fontsize=13, pad=14, color='#333')

# 图例（Zxy / Zyx 标记 + 加权均值）
legend_elements = [
    Line2D([0], [0], marker='^', color='w',
           markerfacecolor=(0.85, 0.15, 0.15, 0.9), markersize=10, label='Zxx'),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor=(0.85, 0.15, 0.15, 0.9), markersize=10, label='Zxy'),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor=(0.15, 0.25, 0.85, 0.9), markersize=10, label='Zyx'),
    Line2D([0], [0], marker='^', color='w',
           markerfacecolor=(0.15, 0.25, 0.85, 0.9), markersize=10, label='Zyy'),
]
ax.legend(handles=legend_elements, loc='lower right',
          fontsize=10, framealpha=0.92, edgecolor='#CCC')

# ── ⭐ 双色条（红/蓝权重，并排）─────────────────────────
cmap_red  = mcolors.LinearSegmentedColormap.from_list('wred',  ['#FFFFFF', '#CC2222'])
cmap_blue = mcolors.LinearSegmentedColormap.from_list('wblue', ['#FFFFFF', '#2233CC'])
norm = Normalize(vmin=0.5, vmax=1)

cax_r = fig.add_axes([0.895, 0.20, 0.022, 0.56])   # 红色条
cax_b = fig.add_axes([0.928, 0.20, 0.022, 0.56])   # 蓝色条

cb_r = mcbar.ColorbarBase(cax_r, cmap=cmap_red,  norm=norm, orientation='vertical')
cb_b = mcbar.ColorbarBase(cax_b, cmap=cmap_blue, norm=norm, orientation='vertical')

# 红色条（Zxy）刻度——放在左侧，避免与右侧蓝色条标签重叠
cb_r.set_ticks([])
cb_r.ax.yaxis.set_ticks_position('left')
cb_r.outline.set_linewidth(0.5)

# 蓝色条（Zyx）刻度——放在右侧，并将 'Weight' 标签放在最右侧
cb_b.ax.tick_params(labelsize=8.5, colors='#444')
cb_b.set_ticks([0.5, 0.625, 0.75, 0.875, 1.0])
# cb_b.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb_b.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])
cb_b.ax.yaxis.set_ticks_position('right')
cb_b.set_label('Weight', fontsize=10, color='#444', labelpad=4)
cb_b.ax.yaxis.set_label_position('right')
cb_b.outline.set_linewidth(0.5)

# 色条底部颜色归属标注
fig.text(0.906, 0.17, 'Zx', ha='center', va='top',
         fontsize=9, color='#CC2222', fontweight='bold')
fig.text(0.939, 0.17, 'Zy', ha='center', va='top',
         fontsize=9, color='#2233CC', fontweight='bold')


# ── 保存 ─────────────────────────────────────────────────
plt.savefig('polar/Z_polar_log_0.005524Hz_70_5_s.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('polar/Z_polar_log_0.005524Hz_70_5_s.pdf',
            dpi=200, bbox_inches='tight', facecolor='white', format='pdf')
print("✓ 图像已保存")
plt.close()