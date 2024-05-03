import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


df = pd.read_csv('Weather.csv', sep=',')
df.columns = ['CET', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
              'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC', 'Max_Humidity',
              'Mean_Humidity', 'Min_Humidity', 'Max_Sea_Level_PressurehPa',
              'Mean_Sea_Level_PressurehPa', 'Min_Sea_Level_PressurehPa',
              'Max_VisibilityKm', 'Mean_VisibilityKm', 'Min_VisibilityKm',
              'Max_Wind_SpeedKm/h', 'Mean_Wind_SpeedKm/h', 'Max_Gust_SpeedKm/h',
              'Precipitationmm', 'CloudCover', 'Events', 'WindDirDegrees']
print(f'\n\nЗавдання 1\n{df.head()}')


print(f'\n\nЗавдання 2\n')
print(f'Записів: {df.shape[0]}, полів: {df.shape[1]}')


M = 2  #  23 лютого 2003р. 
N = 1000
nrows, ncols = df.shape
print(f'\n\nЗавдання 3\nПерші 5 :\n{df.iloc[M-1:M+5-1]}')
range_N = pd.Index([*range(N, nrows+1, N)]) - 1
print(f'\nN`ті записи\n{df.iloc[range_N]}')


print('\n\nЗавдання 4\nТипи колонок:\n')
print(df.dtypes)


df[['Year', 'Month', 'Day']] = df['CET'].str.split('-', expand=True)
df['Month'] = df['Month'].apply(lambda x: x.zfill(2))
df['Day'] = df['Day'].apply(lambda x: x.zfill(2))

df = df.drop(['CET'], axis=1)
print('\n\nЗавдання 5\n', df.head())


print('\n\nЗавдання 6A\n', df['Events'].isna().sum())

print('\nЗавдання 6B')
min_mean_Humidity = df['Mean_Humidity'].min()
print(df[['Year', 'Month', 'Day', 'Mean_Humidity', 'Max_Wind_SpeedKm/h', 'Mean_Wind_SpeedKm/h', ]]\
      .loc[df['Mean_Humidity']==min_mean_Humidity])

print('\nЗавдання 6C')
data = df[(df['Mean_TemperatureC'] >= 0) & (df['Mean_TemperatureC'] <= 5)]
data['date'] = df['Year'] + '_' + df['Month']
print(data.date.unique())


print('\n\nЗавдання 7A\n')
print(df.groupby('Day')['Max_TemperatureC'].mean())

print('\nЗавдання 7B\n', df.groupby('Year').Events.apply(lambda x: x.str.contains('Fog').sum()))


print('\n\nЗавдання 8\n')
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 6]})
fig.suptitle('Bar charts `Events`')

df['Events'].value_counts(dropna=True).plot(kind='bar', color='royalblue')
print(df['Events'].value_counts(dropna=True))
events_counts = df['Events'].value_counts(dropna=True)

ax[1].set_xlabel('Events')
ax[1].set_ylabel('Quantity events')
ax[1].grid(axis='y')


for i in range(len(events_counts.index)):
    ax[1].text(i, events_counts[i] + 15, events_counts[i], ha = 'center')

na_counts = df['Events'].isna().sum()
d = {'Nan Values': na_counts}
na_val_8 = pd.Series(data=d)
print('\nКількість значень Nan =', na_counts)


ax[0].set_ylabel('Quantity events')
ax[0].grid(axis='y')
ax[0].bar('Nan values', na_counts, color='maroon', width=0.1)

for i in range(len(na_val_8.index)):
    ax[0].text(i, na_val_8[i] + 15, na_val_8[i], ha = 'center')

plt.xticks(rotation = 50)
plt.tight_layout()
plt.show()


print('\n\nЗавдання 9\n')
print('Мінімальне значення у колонці = ', df['WindDirDegrees'].min())
print('Максимальне значення у колонці = ', df['WindDirDegrees'].max())

fig, ax = plt.subplots()
plt.title('Діаграма напрямків вітру')

labels = ['ПН', 'ПН-ЗХ', 'ЗХ', 'ПД-ЗХ', 'ПД', 'ПД-СХ', 'СХ', 'ПН-СХ']
dirs = [0, 0, 0, 0, 0, 0, 0, 0]
explode_list = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
colors = ['gray', 'royalblue', 'sandybrown', 'darkorange', 'gold', 'darkorange', 'sandybrown', 'royalblue']


for dir in df['WindDirDegrees']:
    if dir >= -22.5 and dir < 22.5:
        dirs[0] += 1
    elif dir >= 22.5 and dir < 67.5:
        dirs[1] += 1
    elif dir >= 67.5 and dir < 112.5:
        dirs[2] += 1
    elif dir >= 112.5 and dir < 157.5:
        dirs[3] += 1
    elif dir >= 157.5 and dir < 202.5:
        dirs[4] += 1
    elif dir >= 202.5 and dir < 247.5:
        dirs[5] += 1
    elif dir >= 247.5 and dir < 292.5:
        dirs[6] += 1
    elif dir >= 292.5 and dir < 337.5:
        dirs[7] += 1
    elif dir >= 337.5 and dir <= 360:
        dirs[0] += 1

ax.pie(dirs, labels=labels, autopct='%1.1f%%', startangle=22.5, explode=explode_list, colors=colors)
plt.tight_layout()
plt.show()


print('\n\nЗавдання 10\n')
max_tem = df.groupby(['Year', 'Month'], as_index=False)['Max_TemperatureC'].mean()
dew_p = df.groupby(['Year', 'Month'], as_index=False)['Min_DewpointC'].mean()
df['Year_month_view'] = df[['Year', 'Month']].apply('_'.join, axis=1)

Year_month_view_list = []
all_Year_month_view_list = []
count = 0
for i in df['Year_month_view'].unique():
    count += 1
    all_Year_month_view_list.append(i)
    if i.endswith('01') == True:
        Year_month_view_list.append(i)
    else:
        Year_month_view_list.append('')

plt.plot(all_Year_month_view_list, max_tem['Max_TemperatureC'], color='black', label='Max_TemperatureC')
plt.plot(all_Year_month_view_list, dew_p['Min_DewpointC'], color='red', label='Min_DewpointC')

plt.xlabel('Місяці', labelpad=5)
plt.ylabel('Температура C', labelpad=5)
plt.legend(bbox_to_anchor=(1, 0.5))
plt.gca().set_xticks(list(range(count)))
plt.gca().set_xticklabels(Year_month_view_list)
plt.xticks(rotation=45)

print('Середня максимальна температура за місяць:\n', max_tem)
print('Мінімальна температура = ', max_tem['Max_TemperatureC'].min())
print('Максимальна температура = ', max_tem['Max_TemperatureC'].max())

print('\nСередня точка роси за місяць:\n', dew_p)
print('Мінімальна точка роси = ', dew_p['Min_DewpointC'].min())
print('Максимальна точка роси = ', dew_p['Min_DewpointC'].max())
plt.tight_layout()
plt.show()
