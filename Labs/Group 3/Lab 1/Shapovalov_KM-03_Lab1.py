import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


M = 6
N = 3000


data = pd.read_csv("Weather.csv")


print('# Завдання 2\n')
print(f'Data rows: {len(data)}\n'
      f'Data cols: {len(data.columns)}\n')


print('# Завдання 3\n')
print(f"Записи M - M + 5:")
print(data.iloc[M:M+5])
print()
print(f"Кожен N-й запис:")
print(data.iloc[::N])
print('\n')


print('# Завдання 4\n')
print('Типи полів кожного запису:')
print(data.dtypes)
print('\n')


data['Year'] = data['CET'].apply(lambda string: string.split('-')[0])
data['Month'] = data['CET'].apply(lambda string: string.split('-')[1].zfill(2))
data['Day'] = data['CET'].apply(lambda string: string.split('-')[2].zfill(2))
del data['CET']


print('# Завдання 6\n')
print('a. Кількість днів з порожнім полем Events:')
print(data[" Events"].isnull().sum())
print()
print('b. День з мінімальною середньою вологістю:')
print(data.iloc[data[' Mean Humidity'].idxmin()][['Year', 'Month', 'Day', ' Max Wind SpeedKm/h', ' Mean Wind SpeedKm/h', ' Max Gust SpeedKm/h']])
print('\n')
print()
print('c. Місяці з середньою температурою від 0 до 5 градусів:')
for month, temp in data.groupby(["Year", "Month"])["Mean TemperatureC"].mean().iteritems():
    if 0 <= temp <= 5:
        print(f'{month[0]}.{month[1]}: {temp}')
print('\n')


print('# Завдання 7\n')
print('a. Середня температура по кожному дню за всі роки:')
print(data.groupby('Year')['Max TemperatureC'].mean())
print()
fog_days = []
for i in range(0, len(data)):
    events = data[' Events'][i]
    if type(events) is str and 'Fog' in events:
        fog_days.append(i)
print('b. Кількість днів у кожному році з туманом:')
print(data.iloc[fog_days].groupby('Year')[' Events'].count())
print('\n')


events_counts = data.groupby(' Events')[' Events'].count()
plt.figure(1)
plt.bar(events_counts.index, events_counts)
plt.xticks(rotation=45)
plt.title("Завдання 8. Стовпчикова діаграма кількості Events")


winds = {'ПН': 0, 'ПН-СХ': 0, 'СХ': 0, 'ПД-СХ': 0, 'ПД': 0, 'ПД-ЗХ': 0, 'ЗХ': 0, 'ПН-ЗХ': 0}
for row in data['WindDirDegrees']:
    if 337.5 < row or row < 22.5:
        winds['ПН'] += 1
    elif 22.5 < row < 67.5:
        winds['ПН-СХ'] += 1
    elif 67.5 < row < 112.5:
        winds['СХ'] += 1
    elif 112.5 < row < 157.5:
        winds['ПД-СХ'] += 1
    elif 157.5 < row < 202.5:
        winds['ПД'] += 1
    elif 202.5 < row < 247.5:
        winds['ПД-ЗХ'] += 1
    elif 247.5 < row < 292.5:
        winds['ЗХ'] += 1
    else:
        winds['ПН-ЗХ'] += 1
plt.figure(2)
plt.pie(winds.values(), labels=winds.keys())
plt.title("Завдання 9. Кругова діаграма напрямків вітру")


plt.figure(3)
min_dew_points = data.groupby(['Year', 'Month'])['Min DewpointC'].mean()
max_temps = data.groupby(['Year', 'Month'])['Max TemperatureC'].mean()
xpoints = np.arange(0, len(max_temps))
plt.plot(xpoints, min_dew_points, color='blue')
plt.plot(xpoints, max_temps, color='red')
xlabels = []
for month in max_temps.index:
    if month[1] == '01':
        xlabels.append(month[1] + '.' + month[0])
    elif month[1] in ['03', '05', '07', '09', '11']:
        xlabels.append(month[1])
    else:
        xlabels.append('')
plt.xticks(xpoints, xlabels, rotation=90)
plt.title("Завдання 10. Середньомісячна мінімальна точка роси та максимальна температура")


plt.show()
