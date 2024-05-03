import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PATH_TO_FILE = 'Weather.csv'  #  Відносний шлях до файлу (має бути в тій самій директорії)


# ---  1  ---
df = pd.read_csv(PATH_TO_FILE,
                 sep=',',
                 )
df.columns = ['CET', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
              'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC', 'Max_Humidity',
              'Mean_Humidity', 'Min_Humidity', 'Max_Sea_Level_PressurehPa',
              'Mean_Sea_Level_PressurehPa', 'Min_Sea_Level_PressurehPa',
              'Max_VisibilityKm', 'Mean_VisibilityKm', 'Min_VisibilityKm',
              'Max_Wind_SpeedKm/h', 'Mean_Wind_SpeedKm/h', 'Max_Gust_SpeedKm/h',
              'Precipitationmm', 'CloudCover', 'Events', 'WindDirDegrees']
              #  Перейменуємо колонки, щоб прибрати пробіли
print(f'\n\n\nTask 1\nFirst 5 row from dataset:\n\n{df.head()}')



# ---  2  ---
nrows, ncols = df.shape   #  Розмірність датафрейму
print(f'\n\n\nTask 2\n\nRows number = {nrows}\nColumns number = {ncols}\n')
# df.info()  #  Для більш детального опису дататфрейму



# ---  3  ---
M = 6  #  12 червня 2003р. 
N = 3_000  #  500 * 6 = 3000
print(f'\n\n\nTask 3\n\nFirst 5 :\n{df.iloc[M-1:M+5-1]}')  #  5 записів починаючи з М-го
range_N = pd.Index([*range(N, nrows+1, N)]) - 1  #  сторення індексів, N-их строк
print(f'\nN`s rows\n{df.iloc[range_N]}')  #  Вивід N-тих строк



# ---  4  ---
print('\n\n\nTask 4\nColumn`s types:\n')
print(df.dtypes)  #  Функція виведення типів колонок
# Або див. пунк 2 ( df.info() )



# ---  5  ---
df[['Year', 'Month', 'Day']] = df['CET'].str.split('-', expand=True)  #  На основі колонки 'CET' робимо 3 колонки
df['Month'] = df['Month'].apply(lambda x: x.zfill(2))  #  Перетворення у двоцифровий формат
df['Day'] = df['Day'].apply(lambda x: x.zfill(2))  #  Перетворення у двоцифровий формат

df = df.drop(['CET'], axis=1)  #  Видаляємо колонку 'CET'
print('\n\n\nTask 5\nDivide `CET` into 3 columns\n\n', df.head())  #  виводимо змінений датафрейм



# ---  6  ---
    # ---  A  ---
print('\n\n\nTask 6\n\nSubtask A\nNan values in "Events" column =',\
      df['Events'].isna().sum())  #  Кількість пропусків у 'Events'
# print(df.keys())

    # ---  B  ---
print('\nSubtask B')
min_mean_Humidity = df['Mean_Humidity'].min()  #  15
print('A day when average humidity was minimal + wind speed')
print(df[['Year', 'Month', 'Day', 'Mean_Humidity', 'Max_Wind_SpeedKm/h', 'Mean_Wind_SpeedKm/h', ]]\
      .loc[df['Mean_Humidity']==min_mean_Humidity])  #  День коли сердня вологість була мінімальною + швидкості вітру

    # ---  С  ---
print('\nSubtask C')
print('Months when the average temperature is from zero to five degrees:\n')

data = df[(df['Mean_TemperatureC'] >= 0) & (df['Mean_TemperatureC'] <= 5)]  #  робимо новий дф по фільтру
data['date'] = df['Year'] + '_' + df['Month']  #  створюємо нову колонку
print(data.date.unique())  #  Місяці, коли середня температура від нуля до п'яти градусів



# ---  7  ---
print('\n\n\nTask 7\n')
print('Average maximum temperature for each day for all years:\n',\
      df.groupby('Day')['Max_TemperatureC'].mean())

print('\nNumber of days in each year with fog:\n',\
      df.groupby('Year').Events.apply(lambda x: x.str.contains('Fog').sum()))



# ---  8  ---
print('\n\n\nTask 8\n')
print('Check the new window with visualization')  #  вивід у консоль повідомлення про що завдання

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 6]})  #  Створення фігури
fig.suptitle('Bar charts `Events`')  #  Назва фігури

df['Events'].value_counts(dropna=True).plot(kind='bar',
                                            color='royalblue',
                                            )  # малюнок за параметрами
print('Number of values without Nan values\n')
print(df['Events'].value_counts(dropna=True))
events_counts = df['Events'].value_counts(dropna=True)  #  кількість значень без Nan значень

ax[1].set_xlabel('Events')  #  Підпис осі Х
ax[1].set_ylabel('Quantity events')  #  Підпис осі У
ax[1].grid(axis='y')  #  сітка по вісі У


for i in range(len(events_counts.index)):
    ax[1].text(i, events_counts[i] + 15, events_counts[i], ha = 'center')  #  Підпис зверху над стовпцями їх значень
# -------
# побудова лівої частини графіку - Nan значень
na_counts = df['Events'].isna().sum()  #  сума Nan значень
d = {'Nan Values': na_counts}
na_val_8 = pd.Series(data=d)  #  створення пандас серії
print('\nNumber of Nan values =', na_counts)

# ax[0].set_xlabel('Nan values')
ax[0].set_ylabel('Quantity events')  #  Підпис осі У
ax[0].grid(axis='y')  #  сітка по вісі У
ax[0].bar('Nan values',
          na_counts,
          color='maroon',
          width=0.1,
          )  #  Побудова графіку

for i in range(len(na_val_8.index)):
    ax[0].text(i, na_val_8[i] + 15, na_val_8[i], ha = 'center')  #  Підпис зверху над стовпцями їх значень

plt.xticks(rotation = 50)  #  поворот підписів на кут 50 градусів для кращого сприйняття
plt.show()



# ---  9  ---
print('\n\n\nTask 9\n')
print('Check the new window with visualization')
print('Min value in column = ', df['WindDirDegrees'].min())  #  мінімальне значення кута вітру
print('Max value in column = ', df['WindDirDegrees'].max())  #  максимальне значення кута вітру

fig, ax = plt.subplots()
plt.title('Diagram of wind angles')

labels = ['ПН', 'ПН-ЗХ', 'ЗХ', 'ПД-ЗХ', 'ПД', 'ПД-СХ', 'СХ', 'ПН-СХ']  #  підписи секторів
dirs = [0, 0, 0, 0, 0, 0, 0, 0]  #  кількість значень по секторах
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

ax.pie(dirs,
      labels=labels,
      autopct='%1.1f%%',
      startangle=22.5,
      explode=explode_list,
      colors=colors
      )
plt.show()



# ---  10  ---
print('\n\n\nTask 10\n')
print('Check the new window with visualization')

max_tem = df.groupby(['Year', 'Month'], as_index=False)['Max_TemperatureC'].mean()  #  Середнє знач максимальної температури
dew_p = df.groupby(['Year', 'Month'], as_index=False)['Min_DewpointC'].mean()  #  Середнє значення мінімальної точки роси

df['Year_month_view'] = df[['Year', 'Month']].apply('_'.join, axis=1)  #  Створення колонки з датою у зручному форматі 
                                                                                                        # (рік_місяць)

Year_month_view_list = []  #  дати, які виводимо на вісі х
all_Year_month_view_list = []  #  дати по яких будуємо графік (точки)
count = 0
for i in df['Year_month_view'].unique():
    count += 1
    all_Year_month_view_list.append(i)
    if i.endswith('01') == True:
        Year_month_view_list.append(i)
    else:
        Year_month_view_list.append('')

plt.plot(all_Year_month_view_list,
         max_tem['Max_TemperatureC'],
         color='royalblue',
         label='Max_TemperatureC',
         )  #  графік температури
plt.plot(all_Year_month_view_list,
         dew_p['Min_DewpointC'],
         color='maroon',
         label='Min_DewpointC',
         )  #  графік точки роси

plt.xlabel('Month', labelpad=5)  #  підпис осі Х
plt.ylabel('Temperature C', rotation=0, labelpad=50)  #  підпис осі У
plt.title('Plot average lines')  #  заголовок
plt.legend()  #  виведення  легенди графіку на основі label in plt.plot()
plt.gca().set_xticks(list(range(count)))  #  gca замість ax
plt.gca().set_xticklabels(Year_month_view_list)  #  gca замість ax
plt.xticks(rotation=45)  #  нахил підписів на осі Х

print('List of all months and years:\n', df['Year_month_view'].unique(), '\n')  #  Значення усіх місяців

print('Average max temperature ever month:\n', max_tem)  #  Середні значення максимальних температур по місяцях
print('Min temperature = ', max_tem['Max_TemperatureC'].min())  #  мінімальна середня температура серед усіх
print('Max temperature = ', max_tem['Max_TemperatureC'].max())  #  максимальна середня температура серед усіх

print('\nAverage dew points ever month:\n', dew_p)  #  Середні значення максимальних точок роси по місяцях
print('Min dew points = ', dew_p['Min_DewpointC'].min())  #  мінімальна середня точка роси серед усіх
print('Max dew points = ', dew_p['Min_DewpointC'].max())  #  максимальна середня точка роси серед усіх

plt.show()
