import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


print('\n\n#1. Відкрити та зчитати файл з даними.')
df = pd.read_csv('vehicle_sales.csv')
print(df.head())


print('\n\n#2. Визначити та вивести кількість записів та кількість полів у кожному записі.')
print(f'Записів: {df.shape[0]}, полів: {df.shape[1]}')


K = int(input('Введіть коефіцієнт К  '))
print('\n\n#3. Вивести К+7 перших')
print(df.head(K+7))


print('\n#3. Вивести 5К-3 останніх записі')
print(df.tail(5*K-3))


print('\n\n#4. Визначити та вивести тип полів кожного запису')
print(df.info())


print('\n\n#5. Привести поля, що відповідають обсягам продаж, до числового вигляду(показати, що це виконано).')
df['Total_Sales_New'] = df['Total Sales New'].apply(lambda x :x.replace('$','')).astype(int)
df['Total_Sales_Used'] = df['Total Sales Used'].apply(lambda x :x.replace('$','')).astype(int)
print(df.info())


print('\n\n#6. Ввести нові поля: ')
df['Total_used+new'] = df['New'] + df['Used'] #a
df['Total_sales_used+new'] = df['Total_Sales_New'] + df['Total_Sales_Used'] #b
df['Difference_used-new'] = df['Used'] - df['New'] #c
print(df[['Total_used+new','Total_sales_used+new','Difference_used-new']].head())


print('\n\n#7. Змінити порядок розташування полів певним чином')
df_new= df[['Year','Month','Total_sales_used+new','Total_Sales_New','Total_Sales_Used','Total_used+new','New','Used','Difference_used-new']]
print(df_new.head())


print('\n\n#8. Визначити та вивести:')

print('\n#a. Рік та місяць, у які нових автомобілів було продано менше за б/в;')
print(df_new.query('New<Used')[['Year','Month']])

print('\n#b. Рік та місяць, коли сумарний дохід був мінімальни')
print(df_new[df_new['Total_sales_used+new']==df_new['Total_sales_used+new'].min()][['Year','Month']])

print('\n#c. Рік та місяць, коли було продано найбільше б/в авто')
print(df_new[df_new['Used']==df_new['Used'].max()][['Year','Month']])


print('\n\n#9. Визначити та вивести:')
M = 14  #  APR
print('\n#a. Сумарний обсяг продажу транспортних засобів за кожен рік')
print(df_new.groupby('Year').sum()['Total_used+new'])
# М – це порядковий номер у списку підгрупи за абеткою.
print('\n#b. Середній дохід від продажу б/в транспортних засобів в місяці April,')
print(df_new.query('Month=="APR"').mean()['Total_Sales_Used'])


print('\n\n#10. Визначити та вивести:')
print(' Побудувати стовпчикову діаграму обсягу продаж нових авто уроці 2014')
df_2014 = df.query('Year==2014')
df_2014
fig, ax = plt.subplots()
ax.bar(df_2014['Month'],df_2014['New'], color='green')
ax.set_ylabel('Counts')
ax.set_title('Oбсяг продаж нових авто у 2014 році')
plt.show()


print('\n\n#11. Побудувати кругову діаграму сумарного обсягу продаж за кожен рік.')
df_pie = df[['Year','Total_sales_used+new']].groupby('Year').sum()
fig1, ax1 = plt.subplots()
colors = plt.get_cmap('Greens')(np.linspace(0.2, 0.7, len(df_pie.index)))
ax1.pie(df_pie['Total_sales_used+new'],labels = df_pie.index,colors = colors, autopct='%1.2f%%')
ax1.set_title('Сумарний обсяг продаж за кожен рік')
plt.show()


print('\n\n#12. Побудувати на одному графіку:a. Сумарний дохід від продажу нових авто; b. Сумарний дохід від продажу старих авто.')
df_last = df[['Total_Sales_New','Total_Sales_Used']].sum()
fig2, ax2 = plt.subplots()
ax2.bar(df_last.index,df_last, color='Green')
ax2.set_ylabel('Counts in 10^11')
ax2.set_title('Сумарний дохід від продажу старих\нових авто')
plt.show()