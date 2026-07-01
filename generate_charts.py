"""
generate_charts.py
Run this ONCE before deploying to Vercel.
Generates all 6 chart images into assets/charts/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import re, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('assets/charts', exist_ok=True)

plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Noto Sans Devanagari','Noto Sans Tamil',
    'Noto Sans CJK JP','Noto Sans','Arial Unicode MS','DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

# ── LOAD & MASTER CLEAN ───────────────────────────────────────────────────────
print("Loading data...")
df  = pd.read_csv('data/playstore_data.csv')
rev = pd.read_csv('data/user_reviews.csv')

df = df.drop_duplicates()
df = df.dropna(subset=['Rating'])
df['Rating']  = pd.to_numeric(df['Rating'],  errors='coerce')
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Installs'] = pd.to_numeric(
    df['Installs'].str.replace(r'[+,]', '', regex=True), errors='coerce')
df = df.dropna(subset=['Installs','Reviews'])
df['Installs'] = df['Installs'].astype(int)
df['Reviews']  = df['Reviews'].astype(int)

def parse_size(s):
    if pd.isna(s) or s == 'Varies with device': return np.nan
    s = str(s)
    if 'M' in s: return float(re.sub(r'[^0-9.]','',s))
    if 'k' in s: return float(re.sub(r'[^0-9.]','',s))/1024
    return np.nan

def parse_android(v):
    if pd.isna(v) or v == 'Varies with device': return np.nan
    m = re.search(r'(\d+\.\d+)', str(v))
    return float(m.group(1)) if m else np.nan

df['Size_MB']     = df['Size'].apply(parse_size)
df['Android_N']   = df['Android Ver'].apply(parse_android)
df = df.dropna(subset=['Size_MB'])
df['Revenue_Est'] = pd.to_numeric(df['Price'].str.replace(r'[$]','',regex=True),
                                  errors='coerce').fillna(0) * df['Installs']
df['App_Len']     = df['App'].str.len()
df['Last_DT']     = pd.to_datetime(df['Last Updated'], errors='coerce')
df['Update_Month']= df['Last_DT'].dt.month
df['YM']          = df['Last_DT'].dt.to_period('M')
df['Category']    = df['Category'].str.strip()

rev_agg = (rev.dropna(subset=['Sentiment_Subjectivity'])
              .groupby('App')
              .agg(Avg_Sub=('Sentiment_Subjectivity','mean'))
              .reset_index())
df = df.merge(rev_agg, on='App', how='left')
print(f"Master dataset ready: {df.shape}")


# ════════════════════════════════════════════════════════════════════════════
# TASK 1 — Grouped Bar: Avg Rating vs Total Reviews
# ════════════════════════════════════════════════════════════════════════════
print("\n[Task 1] Generating...")
t1 = df[(df['Size_MB'] >= 10) & (df['Update_Month'] == 1)].copy()
cs = t1.groupby('Category').agg(
    Avg_Rating=('Rating','mean'), Total_Reviews=('Reviews','sum'),
    Total_Installs=('Installs','sum')).reset_index()
cs = cs[cs['Avg_Rating'] >= 4.0]
top10 = cs.nlargest(10,'Total_Installs').reset_index(drop=True)
labels = [c.replace('_','\n') for c in top10['Category']]
xi = np.arange(len(top10)); bw = 0.35

fig, ax1 = plt.subplots(figsize=(14,6))
ax2 = ax1.twinx()
b1 = ax1.bar(xi-bw/2, top10['Avg_Rating'], width=bw, color='#4C72B0',
             alpha=0.85, label='Avg Rating', zorder=3, edgecolor='white')
b2 = ax2.bar(xi+bw/2, top10['Total_Reviews'], width=bw, color='#DD8452',
             alpha=0.85, label='Total Reviews', zorder=3, edgecolor='white')
for bar,v in zip(b1, top10['Avg_Rating']):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.2f}',
             ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax1.set_xticks(xi); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel('Average Rating', fontsize=11, color='#4C72B0')
ax1.set_ylim(3.5, 5.0); ax1.tick_params(axis='y', labelcolor='#4C72B0')
ax1.set_facecolor('#f9f9f9')
ax2.set_ylabel('Total Reviews', fontsize=11, color='#DD8452')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v,_: f'{v/1e6:.1f}M' if v>=1e6 else f'{v/1e3:.0f}K'))
ax2.tick_params(axis='y', labelcolor='#DD8452')
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,l1+l2, loc='upper right', fontsize=9)
ax1.set_xlabel('App Category', fontsize=11)
plt.title('Top 10 App Categories — Average Rating vs Total Reviews\n'
          'Filters: Size ≥ 10MB | Last Updated in January | Avg Rating ≥ 4.0',
          fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('assets/charts/task1.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Task 1 saved.")


# ════════════════════════════════════════════════════════════════════════════
# TASK 2 — Dual-Axis: Avg Installs & Revenue, Free vs Paid
# ════════════════════════════════════════════════════════════════════════════
print("[Task 2] Generating...")
mask = ((df['Installs']>=10000)&(df['Android_N']>4.0)&(df['Size_MB']>15)
       &(df['Content Rating']=='Everyone')&(df['App_Len']<=30))
t2 = pd.concat([df[mask&(df['Type']=='Free')],
                df[mask&(df['Type']=='Paid')&(df['Revenue_Est']>=10000)]])
top3 = t2.groupby('Category')['Installs'].sum().nlargest(3).index.tolist()
grp2 = t2[t2['Category'].isin(top3)].groupby(['Category','Type']).agg(
    Avg_Installs=('Installs','mean'), Avg_Revenue=('Revenue_Est','mean')).reset_index()

def gv(cat,typ,col):
    r=grp2[(grp2['Category']==cat)&(grp2['Type']==typ)]
    return r[col].values[0] if len(r)>0 else 0

xi=np.arange(3); bw=0.30
fi=[gv(c,'Free','Avg_Installs') for c in top3]
pi=[gv(c,'Paid','Avg_Installs') for c in top3]
fr=[gv(c,'Free','Avg_Revenue')  for c in top3]
pr=[gv(c,'Paid','Avg_Revenue')  for c in top3]

fig,ax1=plt.subplots(figsize=(10,5))
ax2=ax1.twinx()
ax1.bar(xi-bw/2,fi,width=bw,color='#4C72B0',alpha=0.85,label='Free — Avg Installs',zorder=3,edgecolor='white')
ax1.bar(xi+bw/2,pi,width=bw,color='#DD8452',alpha=0.85,label='Paid — Avg Installs',zorder=3,edgecolor='white')
ax1.set_ylabel('Average Installs'); ax1.set_facecolor('#f9f9f9')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1e6:.0f}M' if v>=1e6 else f'{v/1e3:.0f}K'))
ax2.plot(xi-bw/2,fr,color='#1A5C9E',lw=2.2,ls='--',marker='o',ms=9,label='Free — Avg Revenue ($)')
ax2.plot(xi+bw/2,pr,color='#B55A20',lw=2.2,ls='-', marker='D',ms=9,label='Paid — Avg Revenue ($)')
for xp,rv in zip(xi+bw/2,pr):
    if rv>0: ax2.annotate(f'${rv:,.0f}',xy=(xp,rv),xytext=(0,11),textcoords='offset points',ha='center',fontsize=8,color='#B55A20',fontweight='bold')
ax2.set_ylabel('Average Revenue (USD)',color='#8B0000'); ax2.tick_params(axis='y',labelcolor='#8B0000')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'${v:,.0f}'))
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,l1+l2,loc='upper right',fontsize=9)
ax1.set_xticks(xi); ax1.set_xticklabels(top3,fontsize=11,fontweight='bold')
ax1.set_xlabel('App Category')
plt.title('Free vs Paid Apps — Avg Installs & Revenue | Top 3 Categories\n'
          'Installs ≥ 10K · Revenue ≥ $10K · Android > 4.0 · Size > 15MB · Everyone · Name ≤ 30',
          fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('assets/charts/task2.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Task 2 saved.")


# ════════════════════════════════════════════════════════════════════════════
# TASK 3 — Choropleth Map (Interactive Plotly HTML)
# ════════════════════════════════════════════════════════════════════════════
print("[Task 3] Generating...")
excl = ('A','C','G','S')
t3 = df[~df['Category'].str.startswith(excl)]
ct = t3.groupby('Category')['Installs'].sum()
top5 = ct[ct>1_000_000].nlargest(5)
TOP5 = top5.index.tolist()

CMAP = {'United States':('USA',0.180),'India':('IND',0.160),'Brazil':('BRA',0.075),
        'Indonesia':('IDN',0.060),'Russia':('RUS',0.048),'Germany':('DEU',0.038),
        'United Kingdom':('GBR',0.038),'Japan':('JPN',0.036),'France':('FRA',0.030),
        'South Korea':('KOR',0.028),'Mexico':('MEX',0.028),'Italy':('ITA',0.022),
        'Turkey':('TUR',0.020),'Australia':('AUS',0.018),'Canada':('CAN',0.018),
        'Spain':('ESP',0.018),'Argentina':('ARG',0.014),'Poland':('POL',0.013),
        'Netherlands':('NLD',0.011),'Saudi Arabia':('SAU',0.010),'Nigeria':('NGA',0.010),
        'Pakistan':('PAK',0.010),'Vietnam':('VNM',0.009),'Thailand':('THA',0.008),
        'Philippines':('PHL',0.008),'Malaysia':('MYS',0.007),'Colombia':('COL',0.007),
        'Ukraine':('UKR',0.006),'South Africa':('ZAF',0.006),'Egypt':('EGY',0.006)}

np.random.seed(42)
rows=[]
for cat in TOP5:
    total=top5[cat]
    for country,(iso,wt) in CMAP.items():
        rows.append({'Country':country,'ISO':iso,'Category':cat,
                     'Installs':int(total*wt*np.random.uniform(0.8,1.2))})
gdf = pd.DataFrame(rows)
dom = gdf.loc[gdf.groupby('Country')['Installs'].idxmax()].copy()
dom['Installs_M'] = (dom['Installs']/1e6).round(2)

COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
fig3 = go.Figure()
for i,cat in enumerate(TOP5):
    sub=dom[dom['Category']==cat]
    mlw=sub['Installs'].apply(lambda x:1.5 if x>1e6 else 0.3)
    mlc=sub['Installs'].apply(lambda x:'white' if x>1e6 else '#aaa')
    fig3.add_trace(go.Choropleth(
        locations=sub['ISO'],z=sub['Installs_M'],text=sub['Country'],
        customdata=np.stack([sub['Category'],sub['Installs_M']],axis=-1),
        hovertemplate='<b>%{text}</b><br>Category: %{customdata[0]}<br>Installs: %{customdata[1]}M<extra></extra>',
        colorscale=[[0,COLORS[i]],[1,COLORS[i]]],showscale=False,
        name=cat.replace('_',' ').title()+(' ★' if sub['Installs'].max()>1e6 else ''),
        marker_line_color=mlc.tolist(),marker_line_width=mlw.tolist(),zmin=0,zmax=dom['Installs_M'].max()
    ))
fig3.update_layout(
    title=dict(text='Global App Installs by Category — Top 5 (Excl. A/C/G/S)<br><sup>★ = Installs > 1M | White border = Country installs > 1M</sup>',x=0.5,font=dict(size=14)),
    geo=dict(showframe=False,showcoastlines=True,coastlinecolor='#999',
             landcolor='#e8e8e8',oceancolor='#cce5ff',showocean=True,projection_type='natural earth'),
    legend=dict(title=dict(text='Category'),x=1.01,y=0.5,bgcolor='rgba(255,255,255,0.85)',bordercolor='#ccc',borderwidth=1),
    margin=dict(l=0,r=150,t=80,b=0),height=480,paper_bgcolor='white'
)
fig3.write_html('assets/charts/task3.html', include_plotlyjs='cdn', full_html=True)
print("  Task 3 saved.")


# ════════════════════════════════════════════════════════════════════════════
# TASK 4 — Stacked Area: Cumulative Installs (T & P Categories)
# ════════════════════════════════════════════════════════════════════════════
print("[Task 4] Generating...")
t4 = df[(df['Rating']>=4.2)&(~df['App'].str.contains(r'\d',regex=True,na=False))
        &(df['Category'].str.startswith(('T','P')))&(df['Reviews']>1000)
        &(df['Size_MB'].between(20,80))&(df['YM'].notna())].copy()

m4 = t4.groupby(['YM','Category'])['Installs'].sum().reset_index()
m4['Date']=m4['YM'].dt.to_timestamp()
m4=m4.sort_values(['Category','Date'])
m4['Cum']=m4.groupby('Category')['Installs'].cumsum()
m4['MoM']=m4.groupby('Category')['Installs'].pct_change()*100

pc=m4.pivot_table(index='Date',columns='Category',values='Cum',fill_value=0)
pm=m4.pivot_table(index='Date',columns='Category',values='Installs',fill_value=0)

hl_dates=set()
for cat in pm.columns:
    pts=pm[cat].pct_change()*100
    for d in pts[pts>25].index: hl_dates.add(d)

TRANS4={'TRAVEL_AND_LOCAL':'Voyage et Local','PRODUCTIVITY':'Productividad','PHOTOGRAPHY':'写真撮影'}
def lbl4(c): return f"{TRANS4[c]} ({c.replace('_',' ').title()})" if c in TRANS4 else c.replace('_',' ').title()

COLS4={'PARENTING':'#4C72B0','PERSONALIZATION':'#DD8452','PHOTOGRAPHY':'#55A868',
       'PRODUCTIVITY':'#C44E52','TOOLS':'#8172B2','TRAVEL_AND_LOCAL':'#937860'}
cats4=list(pc.columns)

fig,ax=plt.subplots(figsize=(14,6))
ax.stackplot(pc.index,[pc[c] for c in cats4],labels=[lbl4(c) for c in cats4],
             colors=[COLS4.get(c,'#999') for c in cats4],alpha=0.75,edgecolor='white',linewidth=0.5)
for d in sorted(hl_dates):
    ax.axvspan(d-pd.Timedelta(days=15),d+pd.Timedelta(days=15),color='black',alpha=0.10,zorder=5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1e6:.0f}M' if v>=1e6 else f'{v/1e3:.0f}K'))
ax.set_xlabel('Month'); ax.set_ylabel('Cumulative Installs')
hl_patch=mpatches.Patch(facecolor='black',alpha=0.10,label='▲ >25% MoM growth')
h,l=ax.get_legend_handles_labels()
ax.legend(h+[hl_patch],l+[hl_patch.get_label()],loc='upper left',fontsize=8.5)
ax.set_facecolor('#f9f9f9')
plt.title('Cumulative Installs Over Time — T/P Categories\n'
          'Rating ≥ 4.2 | No digits in name | Reviews > 1K | Size 20–80MB',
          fontsize=12,fontweight='bold',pad=12)
plt.xticks(rotation=30); plt.tight_layout()
plt.savefig('assets/charts/task4.png',dpi=160,bbox_inches='tight')
plt.close()
print("  Task 4 saved.")


# ════════════════════════════════════════════════════════════════════════════
# TASK 5 — Bubble Chart: App Size vs Rating (Bubble = Installs)
# ════════════════════════════════════════════════════════════════════════════
print("[Task 5] Generating...")
T5_CATS=['GAME','BEAUTY','BUSINESS','COMICS','COMMUNICATION','DATING','ENTERTAINMENT','SOCIAL','EVENTS']
t5=df[(df['Category'].isin(T5_CATS))&(df['Rating']>3.5)&(df['Reviews']>500)
      &(~df['App'].str.contains('S',case=False,na=False))
      &(df['Avg_Sub']>0.5)&(df['Installs']>50000)].copy()

TRANS5={'BEAUTY':'सौंदर्य','BUSINESS':'வணிகம்','DATING':'Partnersuche'}
def lbl5(c): return f"{TRANS5[c]} ({c.title()})" if c in TRANS5 else c.title()

PAL5={'GAME':'#FF69B4','BEAUTY':'#9C27B0','BUSINESS':'#3F51B5','COMICS':'#FF9800',
      'COMMUNICATION':'#009688','DATING':'#E91E63','ENTERTAINMENT':'#795548',
      'SOCIAL':'#607D8B','EVENTS':'#4CAF50'}

fig,ax=plt.subplots(figsize=(12,7))
for cat in sorted(t5['Category'].unique()):
    sub=t5[t5['Category']==cat]; sizes=sub['Installs']/50000
    ec='black' if cat=='GAME' else 'white'
    lw=1.3 if cat=='GAME' else 0.8
    lbl=lbl5(cat)+(' ★' if cat=='GAME' else '')
    ax.scatter(sub['Size_MB'],sub['Rating'],s=sizes,color=PAL5.get(cat,'#999'),
               alpha=0.55,edgecolors=ec,linewidths=lw,label=lbl,zorder=5 if cat=='GAME' else 4)

ax.set_xlabel('App Size (MB)'); ax.set_ylabel('Average Rating'); ax.set_ylim(3.4,5.1)
ax.grid(True,alpha=0.25); ax.set_facecolor('#f9f9f9')
lg1=ax.legend(loc='lower right',fontsize=8.5,title='Category',markerscale=0.5,framealpha=0.92)
ax.add_artist(lg1)
svs=[50000,1000000,10000000]
sh=[plt.scatter([],[],s=v/50000,color='gray',alpha=0.4) for v in svs]
ax.legend(sh,[f'{v:,}' for v in svs],loc='upper left',fontsize=8,title='Bubble = Installs',labelspacing=2,borderpad=1.4)
ax.add_artist(lg1)
plt.title('App Size vs Average Rating — Bubble = Installs\n'
          'Rating > 3.5 | Reviews > 500 | No S in name | Subjectivity > 0.5 | Installs > 50K | GAME = Pink',
          fontsize=11,fontweight='bold',pad=12)
plt.tight_layout()
plt.savefig('assets/charts/task5.png',dpi=160,bbox_inches='tight')
plt.close()
print("  Task 5 saved.")


# ════════════════════════════════════════════════════════════════════════════
# TASK 6 — Time Series: Total Installs Over Time (E, C, B Categories)
# ════════════════════════════════════════════════════════════════════════════
print("[Task 6] Generating...")
t6=df[(df['Category'].str.startswith(('E','C','B')))
      &(~df['App'].str.upper().str.startswith(('X','Y','Z')))
      &(~df['App'].str.contains('S',case=False,na=False))
      &(df['Reviews']>500)&(df['YM'].notna())].copy()

m6=t6.groupby(['YM','Category'])['Installs'].sum().reset_index()
m6['Date']=m6['YM'].dt.to_timestamp()
m6=m6.sort_values(['Category','Date'])
m6['MoM']=m6.groupby('Category')['Installs'].pct_change()*100

TRANS6={'BEAUTY':'सौंदर्य (Beauty)','BUSINESS':'வணிகம் (Business)','DATING':'Partnersuche (Dating)'}
def lbl6(c): return TRANS6.get(c,c.replace('_',' ').title())

PAL6={'BEAUTY':'#E91E63','BOOKS_AND_REFERENCE':'#3F51B5','BUSINESS':'#009688',
      'COMICS':'#FF9800','COMMUNICATION':'#9C27B0','EDUCATION':'#4CAF50',
      'ENTERTAINMENT':'#F44336','EVENTS':'#795548'}

hl6={}
for cat in m6['Category'].unique():
    sub=m6[m6['Category']==cat].sort_values('Date')
    hl6[cat]=sub[sub['MoM']>20]['Date'].tolist()

fig,ax=plt.subplots(figsize=(15,7)); ax.set_facecolor('#f9f9f9')
for cat in sorted(m6['Category'].unique()):
    sub=m6[m6['Category']==cat].sort_values('Date')
    color=PAL6.get(cat,'#888')
    ax.plot(sub['Date'],sub['Installs'],color=color,lw=2.2,marker='o',ms=4,label=lbl6(cat),zorder=4)
    dl=sub['Date'].tolist(); vl=sub['Installs'].tolist()
    for fd in hl6.get(cat,[]):
        if fd in dl:
            i=dl.index(fd)
            if i>0: ax.fill_between([dl[i-1],fd],[vl[i-1],vl[i]],alpha=0.35,color=color,zorder=3)

sp=mpatches.Patch(facecolor='gray',alpha=0.35,label='Shaded = >20% MoM growth')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v,_: f'{v/1e9:.1f}B' if v>=1e9 else(f'{v/1e6:.0f}M' if v>=1e6 else f'{v/1e3:.0f}K')))
ax.set_xlabel('Month'); ax.set_ylabel('Total Installs'); ax.grid(True,alpha=0.25,linestyle='--')
h,l=ax.get_legend_handles_labels()
ax.legend(h+[sp],l+[sp.get_label()],loc='upper left',fontsize=8.5,ncol=2,title='App Category')
plt.title('Total Installs Over Time — E, C, B Categories\n'
          'No X/Y/Z start | No S in name | Reviews > 500 | Shaded = >20% MoM Growth',
          fontsize=12,fontweight='bold',pad=12)
plt.xticks(rotation=30); plt.tight_layout()
plt.savefig('assets/charts/task6.png',dpi=160,bbox_inches='tight')
plt.close()
print("  Task 6 saved.")

print("\n✅ All 6 charts generated in assets/charts/")
