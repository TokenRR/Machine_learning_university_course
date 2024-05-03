import matplotlib.pyplot as plt
import pandas as pd


filepath = 'Vehicle_Sales.csv'   # шлях до файлу 'Vehicle_Sales.csv'

    
# 1

df = pd.read_csv(filepath)   # зчитування файлу

print(f'# 1\n\nInitial dataframe:\n{df}\n')   # вивід вмісту файлу


# 2

nrows, ncols = df.shape   # розмірність таблички (кількість рядочків та кількість стовпчиків)

print(f'# 2\n\nRows number: {nrows}\nColumns number: {ncols}\n')


# 3

K = 5   # магічне число K

print(f'# 3\n\nFirst K+7:\n{df.head(K+7)}\n')   # вивід перших К+7 записів
print(f'Last 5K-3:\n{df.tail(5*K-3)}\n')        # вивід останніх 5K-3 записи


# 4

print(f'# 4\n\nTypes:\n{df.dtypes}\n')   # вивід типів кожного поля


# 5

df['Total Sales New'] = df['Total Sales New'].apply(lambda x: int(x[1:]))     # приведення поля Total Sales New  до числового вигляду
df['Total Sales Used'] = df['Total Sales Used'].apply(lambda x: int(x[1:]))   # приведення поля Total Sales Used до числового вигляду

print(f'# 5\n\nUpdated dataframe:\n{df}\n')
print(f'Updated types:\n{df.dtypes}\n')


# 6

df['Cars'] = df['New'] + df['Used']                                   # введення поля 'Cars' (сумарний обсяг продаж автомобілів)
df['Total Income'] = df['Total Sales New'] + df['Total Sales Used']   # введення поля 'Total Income' (сумарних дохід від продажу автомобілів)
df['Diff'] = abs(df['New'] - df['Used'])                              # введення поля 'Diff' (різниця в обсязі продаж нових та б/в автомобілів)

print(f'# 6\n\nAdded new columns:\n{df}\n')


# 7

df = df[['Year', 'Month', 'Total Income', 'Total Sales New', 'Total Sales Used', 'Cars', 'New', 'Used', 'Diff']]   # зміна порядку полів

print(f'# 7\n\nChanged columns order:\n{df}\n')


# 8

a = df[df['New'] < df['Used']][['Year', 'Month']]                           # рік та місяць, у які нових автомобілів було продано менше за б/в
b = df[df['Total Income'] == df['Total Income'].min()][['Year', 'Month']]   # рік та місяць, коли сумарний дохід був мінімальним
c = df[df['Used'] == df['Used'].max()][['Year', 'Month']]                   # рік та місяць, коли було продано найбільше б/в авто

print(f'# 8\n\na:\n{a}\n')
print(f'b:\n{b}\n')
print(f'c:\n{c}\n')


# 9

a = df.groupby('Year')['Cars'].sum()   # групування за роком та підрахунок обсягу продаж автомобілів за кожен рік

M = 5   # номер у підгрупі
b = df[df['Month'] == df['Month'][M-1]].groupby('Month')['Total Sales Used'].mean()   # відфільтровування даних за потрібним місяцем, групування, підрахунок середнього значення доходу від продаж б/в автомобілів

print(f'# 9\n\na:\n{a}\n')
print(f'b:\n{b}\n')


# 10

year = int('20' + str(17-M))
data = df[df['Year'] == year][['Month', 'New']]   # відфільтровування даних за потрібним роком, 
                                                  # вибір потрібних полів ('Month' та 'New')

fig, ax = plt.subplots()              # створення фігури
ax.set_title(f'New cars in {year}')   # встановлення заголовка
ax.bar(data['Month'], data['New'])    # побудова гістограми обсягу продав нових авто у відповідному році

plt.show()   # вивід графіка


# 11

data = df.groupby('Year')['Cars'].sum()   # групування за роком та підрахунок обсягу продаж автомобілів за кожен рік

fig, ax = plt.subplots()
ax.set_title('Cars count per year')
ax.pie(data, labels=data.index, autopct='%1.2f%%')   # побудова кругової діаграми сумарного обсягу продаж за кожен рік

plt.show()


# 12

fig, ax = plt.subplots()
ax.set_title('Total sales')
ax.plot(df.index, df['Total Sales New'], marker='.', label='Total Sales New')     # побудова графіка доходів від продаж нових авто
ax.plot(df.index, df['Total Sales Used'], marker='.', label='Total Sales Used')   # побудова графіка доходів від продаж б/в авто
ax.legend()

plt.show()
