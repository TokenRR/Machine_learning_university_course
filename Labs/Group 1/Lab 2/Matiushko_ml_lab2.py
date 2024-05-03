import pandas as pd
import sklearn.metrics as sm
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)


#3a
def number_3a(y, y_pred):
    res_list = []

    for threshold in np.arange(0, 1, 0.1):
        binarizy = Binarizer(threshold=threshold).fit(y_pred.values.reshape(1, -1))
        binarized = binarizy.transform(y_pred.values.reshape(1, -1))
        binarized.astype(int)

        prs, recs, _ = sm.precision_recall_curve(y, y_pred)
        fprs, tprs, thr = sm.roc_curve(y, y_pred)

        res_list.append([threshold, \
                         sm.accuracy_score(y, binarized.T), \
                         sm.precision_score(y, binarized.T), \
                         sm.recall_score(y, binarized.T), \
                         sm.f1_score(y, binarized.T), \
                         sm.matthews_corrcoef(y, binarized.T), \
                         sm.balanced_accuracy_score(y, binarized.T), \
                         np.max(tprs - fprs), \
                         sm.auc(recs, prs), \
                         sm.auc(fprs, tprs)])


    df_res = pd.DataFrame(res_list, columns=['Threashold', \
                                             'Accuracy_score', \
                                             'Precision_score', \
                                             'Recall_score', \
                                             'F1_score', \
                                             'Matthews_corrcoef', \
                                             'Balanced_accuracy_score', \
                                             'Youden’s  J-statistics', \
                                             'Recall_prec under ', \
                                             'Roc under'])

    return df_res



#3b
def number_3b(df,name):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    list1 = []
    list1 = np.arange(0, 1, 0.1)

    for i in df.columns:
        if i == "Threashold":
            continue
        ax.plot(list1, df[i].values, linewidth=2, label=i)

    ax.set_xlabel('Threashold')
    ax.set_ylabel('Score')
    ax.set_title(f'3b: Metrics {name}')
    plt.legend(loc='lower right')
    plt.show()


#3c
def number_3c(y, y_pred,name):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    list1 = []
    list1 = np.arange(0, 1, 0.1)
    res_c1 = []
    res_c0 = []

    for threshold in np.arange(0, 1, 0.1):
        binarizy = Binarizer(threshold=threshold).fit(y_pred.values.reshape(1, -1))
        binarized = binarizy.transform(y_pred.values.reshape(1, -1))
        binarized.astype(int)

        c1 = 0
        c0 = 0

        for a in binarized[0]:
            if a == 1:
                c1 += 1
            else:
                c0 += 1

        res_c1.append(c1)
        res_c0.append(c0)

    ax.plot(np.arange(0, 1, 0.1), res_c1, linewidth=2, label='1', marker="|")
    ax.plot(np.arange(0, 1, 0.1), res_c0, linewidth=2, label='0', marker="|")
    plt.xlabel('Thrashold')
    plt.ylabel('Count')
    plt.title(f'3c: {name}')

    plt.legend(loc='lower right')
    plt.show()



#3d
def number_3d(y, y_pred,name):
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    # prs recs
    prs, recs, _ = sm.precision_recall_curve(y, y_pred.values.reshape(1, -1).T)

    ax.plot(recs, prs, marker='o')
    ax.set_ylim([0, 1.1])
    ax.set_xlim([0, 1.1])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'3d: PR_curve of {name}')
    #plt.show()

    # roc curve
    fprs, tprs, thr = sm.roc_curve(y, y_pred.values.reshape(1, -1).T)

    ax1.plot(fprs, tprs, marker='o')
    ax1.set_ylim([0, 1.1])
    ax1.set_xlim([0, 1.1])
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.set_title(f'3d: ROC_curve of {name}')
    plt.show()





#----------------------------------------------------------------

#1.Відкрити та зчитати дані з наданого файлу. Файл містить три стовпчики
input('Enter to show number 1:')
df = pd.read_csv("KM-01-1.csv")
print(df.head())
input('Enter to show number 2:')


#2. Визначити  збалансованість  набору  даних.  Вивести  кількість  об’єктів кожного класу.
print(df['GT'].value_counts())
input('Enter to show number 3a:')


#3a
df_res_m1 = pd.DataFrame()
df_res_m2 = pd.DataFrame()
df_res_m1 = number_3a(df['GT'], df['Model_1'])
print(f'3a: Model_1 :\n')
print(df_res_m1)
df_res_m2 = number_3a(df['GT'], df['Model_2'])
print(f'3a: Model_2 : \n')
print(df_res_m2)
input('Enter to show nubmer 3b:')


# 3b
number_3b(df_res_m1,'Model 1')
number_3b(df_res_m2,'Model 2')
input('Enter to show number 3c:')


# 3c
number_3c(df.GT, df.Model_1, 'Model_1')
number_3c(df.GT, df.Model_2,'Model_2')
input('Enter to show number 3d:')


#3d
number_3d(df.GT,df.Model_1,'Model_1')
number_3d(df.GT,df.Model_2, 'Model_2')
input('Enter to show number 4:')


#4
#висновки
print('Можемо побачити ,що у першій моделі компоновані метрики вищі ніж у другої ,а також точність. Проте у другої моделі більше площа під рок кривою.\n'
      'Ябільш схиляюся до 1ї моделі')
input('Enter to show number 5:')

#5
df_1 = df[df['GT'] == 1].sample(frac=0.6)
df_5aufgabe = df.drop(labels =df_1.index ,axis = 0)
print('Мій місяць 4. Тому прибираємо 60 %')
input('Enter to show number 6:')



#6
print(f'Кільксть занчень кожного типу : \n{df_5aufgabe.GT.value_counts()}')
print(f'Procent % of 0 :{round(df_5aufgabe.GT.value_counts()[0] / 3500, 2)}')
print(f'Procent % of 1 :{round(df_5aufgabe.GT.value_counts()[1] / 3500, 2)}')
input('Enter to show number 7. 3a_sub:')



#7

#3a_sub
df_res_m1 = pd.DataFrame()
df_res_m2 = pd.DataFrame()
df_res_m1 = number_3a(df_5aufgabe['GT'], df_5aufgabe['Model_1'])
print(f'3a: Model_1_sub :\n')
print(df_res_m1)
df_res_m2 = number_3a(df_5aufgabe['GT'], df_5aufgabe['Model_2'])
print(f'3a: Model_2_sub : \n')
print(df_res_m2)
input('Enter to show number 7. 3b_sub:')

# 3b_sub
number_3b(df_res_m1,'Model_1_sub')
number_3b(df_res_m2,'Model_2_sub')
input('Enter to show number 7. 3c_sub:')

# 3c_sub
number_3c(df_5aufgabe.GT, df_5aufgabe.Model_1, 'Model_1_sub')
number_3c(df_5aufgabe.GT, df_5aufgabe.Model_2,'Model_2_sub')
input('Enter to show number 7. 3d_sub:')

#3d_sub
number_3d(df_5aufgabe.GT,df_5aufgabe.Model_1,'Model_1_sub')
number_3d(df_5aufgabe.GT,df_5aufgabe.Model_2, 'Model_2_sub')
input('Enter to show number 8:')


#8
print('я більш схиляюся до 1 моделі')
input('Enter to show number 9:')


#9
print('Там де вибірка менш збалансовама має сенс дивитися на рекол_прес криву, а вона більше у другої моделі \n'
      'Проте інші показники вищі у першої, к тому ж різниця під кривою не така ж і велика, тому я склоняюсь що перша кращя')







