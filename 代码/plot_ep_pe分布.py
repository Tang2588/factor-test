# -*- coding: utf-8 -*-
r"""EP / PE 分布直方图
数据: 中间结果\ep_raw.parquet（原始 EP = TTM归母净利 / 总市值）
      中间结果\pe_raw.parquet（原始 PE = 1/EP）
输出: 图形\ep_原始分布.png（池化 + 最新截面）
      图形\pe_分布.png   （PE = 1/EP，同双面板）
只读，不改动任何管道中间结果。
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r'D:\因子计算\市盈率\EP因子_交付版'
OUT = os.path.join(BASE, '图形')
os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

EP_LO, EP_HI = -0.5, 0.5        # EP 显示窗口
PE_LO, PE_HI = -100.0, 300.0    # PE 显示窗口


def stats_line(v, unit=''):
    v = v.dropna()
    q = v.quantile([.01, .25, .50, .75, .99])
    return (f'n={len(v):,}  负值 {100 * (v < 0).mean():.2f}%  '
            f'中位 {q[.50]:.4f}{unit}\n'
            f'分位 1%={q[.01]:.4f}  25%={q[.25]:.4f}  '
            f'75%={q[.75]:.4f}  99%={q[.99]:.4f}  极值 [{v.min():.3f}, {v.max():.3f}]')


# ---- 读数据 ----
df = pd.read_parquet(os.path.join(BASE, '中间结果', 'ep_raw.parquet'))
ep = df['signal']
pe_df = pd.read_parquet(os.path.join(BASE, '中间结果', 'pe_raw.parquet'))
if not df.index.equals(pe_df.index):
    raise ValueError('ep_raw 和 pe_raw 的索引不一致')
pe = pe_df['signal']
dates = df.index.get_level_values('date')
last_day = dates.max()
ep_all = ep.dropna()
ep_cs = ep[dates == last_day].dropna()

# PE = 1/EP；零 EP 在 pe_raw 中已按无定义处理为 NaN。
pe_all = pe.dropna()
pe_cs = pe[dates == last_day].dropna()

# ---- 控制台统计（供处理记录登记） ----
print('== EP 原始值 ==')
print('全样本池化:', stats_line(ep_all))
print(f'最新截面 {last_day.date()}:', stats_line(ep_cs))
print(f'EP 显示窗口 [{EP_LO}, {EP_HI}] 之外: '
      f'{100 * ((ep_all < EP_LO) | (ep_all > EP_HI)).mean():.3f}%')
print('\n== PE = 1/EP ==')
print('全样本池化:', stats_line(pe_all, '倍'))
print(f'最新截面 {last_day.date()}:', stats_line(pe_cs, '倍'))
print(f'PE 显示窗口 [{PE_LO:.0f}, {PE_HI:.0f}] 之外: '
      f'{100 * ((pe_all < PE_LO) | (pe_all > PE_HI)).mean():.3f}%')
print(f'正 PE 中位数: 池化 {pe_all[pe_all > 0].median():.2f}  '
      f'截面 {pe_cs[pe_cs > 0].median():.2f}')


def draw_hist(ax, v, lo, hi, nbins, title, note, logy=False):
    inside = v[(v >= lo) & (v <= hi)]
    outside = 1 - len(inside) / len(v)
    ax.hist(inside, bins=np.linspace(lo, hi, nbins),
            color='#3b6fb6', edgecolor='white', linewidth=.3)
    if logy:
        ax.set_yscale('log')
    med = v.median()
    ax.axvline(0, color='gray', lw=.8, ls=':')
    ax.axvline(med, color='#c0392b', lw=1.2, ls='--', label=f'中位数 {med:.4g}')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('EP（TTM归母净利/总市值）' if 'EP' in title else 'PE = 1/EP（倍）')
    ax.set_ylabel('样本数（对数）' if logy else '样本数')
    ax.text(.01, .98, f'{note}\n窗口外未显示 {100 * outside:.2f}%',
            transform=ax.transAxes, va='top', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#bbb', alpha=.9))


# ---- 图 1: EP ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
d0, d1 = dates.min().date(), dates.max().date()
draw_hist(axes[0], ep_all, EP_LO, EP_HI, 161,
          f'EP 全样本池化（{d0} ~ {d1}）',
          f'有效 {len(ep_all):,} 条\n负值 {100 * (ep_all < 0).mean():.2f}%')
draw_hist(axes[1], ep_cs, EP_LO, EP_HI, 161,
          f'EP 最新截面（{last_day.date()}）',
          f'有效 {len(ep_cs):,} 条\n负值 {100 * (ep_cs < 0).mean():.2f}%')
fig.suptitle('EP 原始分布（MAD/标准化之前）', fontsize=13)
fig.tight_layout()
f1 = os.path.join(OUT, 'ep_原始分布.png')
fig.savefig(f1, dpi=150)
plt.close(fig)

# ---- 图 2: PE ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
draw_hist(axes[0], pe_all, PE_LO, PE_HI, 150,
          f'PE 全样本池化（{d0} ~ {d1}）',
          f'负 PE（亏损）{100 * (pe_all < 0).mean():.2f}%\n'
          f'正 PE 中位 {pe_all[pe_all > 0].median():.1f} 倍', logy=True)
draw_hist(axes[1], pe_cs, PE_LO, PE_HI, 150,
          f'PE 最新截面（{last_day.date()}）',
          f'负 PE（亏损）{100 * (pe_cs < 0).mean():.2f}%\n'
          f'正 PE 中位 {pe_cs[pe_cs > 0].median():.1f} 倍', logy=True)
fig.suptitle('PE = 1/EP 分布（亏损股为负值，近零 EP 使尾部极长，窗口外截断显示）',
             fontsize=13)
fig.tight_layout()
f2 = os.path.join(OUT, 'pe_分布.png')
fig.savefig(f2, dpi=150)
plt.close(fig)

print('\n输出:', f1)
print('输出:', f2)
