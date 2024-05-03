import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as skm
import copy


# --- 1 --- #
df = pd.read_csv(r'KM-01-3.csv')
print(df)


# --- 2 --- #
print(df.groupby('GT')['GT'].count().reset_index(name="count"))


# --- 3 --- #
def task3(data, task):
    print(f"\n\n{task}.\n")
    print(f"{task}a")

    df = data
    step = 0.1
    range_step = np.arange(step, 1, step)

    metrics_m1 = {
        'Accuracy': {'value': np.array([]), 'func': skm.accuracy_score},
        'Precision': {'value': np.array([]), 'func': skm.precision_score},
        'Recall': {'value': np.array([]), 'func': skm.recall_score},
        'F-Scores': {'value': np.array([]), 'func': skm.f1_score},
        'Matthews Correlation Coefficient': {'value': np.array([]), 'func': skm.matthews_corrcoef},
        'Balanced Accuracy': {'value': np.array([]), 'func': skm.balanced_accuracy_score},
        'Youden’s J statistics': {'value': np.array([])},
        'Area Under Curve for Precision-Recall Curve': {'value': np.array([]), 'func': skm.average_precision_score},
        'Area Under Curve for Receiver Operation Curve': {'value': np.array([]), 'func': skm.roc_auc_score},

    }
    metrics_m2 = copy.deepcopy(metrics_m1)

    for stp in range_step:
        for key in metrics_m1.keys():
            if key == 'Youden’s J statistics':
                temp1 = metrics_m1['Balanced Accuracy']['value'][-1]
                metrics_m1[key]['value'] = np.append(metrics_m1[key]['value'], temp1 * 2 - 1)
                temp2 = metrics_m2['Balanced Accuracy']['value'][-1]
                metrics_m2[key]['value'] = np.append(metrics_m2[key]['value'], temp2 * 2 - 1)
                continue

            metrics_m1[key]['value'] = np.append(metrics_m1[key]['value'],
                                               metrics_m1[key]['func'](df['GT'], [int(i < stp) for i in df['Model_1_0']]))
            metrics_m2[key]['value'] = np.append(metrics_m2[key]['value'],
                                               metrics_m2[key]['func'](df['GT'], [int(i > stp) for i in df['Model_2_1']]))

    for key in metrics_m1.keys():
        print(key)
        print(np.around(metrics_m1[key]['value'], 3))
        print(np.around(metrics_m2[key]['value'], 3))
        print()

    print(f"\n{task}b")
    fig, ax = plt.subplots()

    xmax_arr = np.array([])
    ymax_arr = np.array([])
    xymax = {
        'm1': {},
        'm2': {},
    }  # для завдання 3c

    start = 0.4
    finish = 0.9
    opacity = np.arange(start, finish, (finish-start)/len(metrics_m1))
    op_ind = 0

    for key in metrics_m1.keys():
        plt.plot(range_step, metrics_m1[key]['value'], color='red', alpha=opacity[op_ind], label=f"m1_{key}")
        plt.plot(range_step, metrics_m2[key]['value'], color='blue', alpha=opacity[op_ind], label=f"m2_{key}")
        op_ind += 1

        # для завдання 3с
        xymax['m1'][key] = {'x': [], 'y': []}
        xymax['m2'][key] = {'x': [], 'y': []}
        #

        ymax = max(metrics_m1[key]['value'])
        xpos = np.where(metrics_m1[key]['value'] == ymax)
        xmax = range_step[xpos]

        for el in xmax:  # на випадок якщо буде кілька максимумів
            ymax_arr = np.append(ymax_arr, ymax)
            xymax['m1'][key]['y'].append(ymax)
            xymax['m1'][key]['x'].append(el)
        xmax_arr = np.append(xmax_arr, xmax)


        ymax = max(metrics_m2[key]['value'])
        xpos = np.where(metrics_m2[key]['value'] == ymax)
        xmax = range_step[xpos]

        for el in xmax:
            ymax_arr = np.append(ymax_arr, ymax)
            xymax['m2'][key]['y'].append(ymax)
            xymax['m2'][key]['x'].append(el)
        xmax_arr = np.append(xmax_arr, xmax)

    print('Максимуми:')
    print(f"m1: {xymax['m1']}")
    print(f"m2: {xymax['m2']}")

    scatter = plt.scatter(
        x=xmax_arr,
        y=ymax_arr,
        c='black',
        zorder=10
    )

    plt.grid()
    plt.xlabel("Величина порогу ")
    plt.ylabel("Значення метрики")
    plt.title("Метрики")
    plt.legend(bbox_to_anchor=(0.5, 0.5), loc='center right')
    plt.show()

    print(f"\n{task}c")
    count_m1 = [[], []]
    count_m2 = [[], []]
    for stp in range_step:
        if stp == 0:
            continue
        array1 = [int(i < stp) for i in df['Model_1_0']]
        array2 = [int(i > stp) for i in df['Model_2_1']]

        count_m1[0].append(len(df.values) - sum(array1))
        count_m1[1].append(sum(array1))

        count_m2[0].append(len(df.values) - sum(array2))
        count_m2[1].append(sum(array2))

    plt.subplot(121)
    plt.bar(range_step - 0.025/2, count_m1[0], width=0.025, color='blue', label='Class 0', alpha=0.5)
    plt.bar(range_step + 0.025/2, count_m1[1], width=0.025, color='red', label='Class 1', alpha=0.5)

    hs_ind = 1
    for key in metrics_m1.keys():
        plt.bar(-0.006+xymax['m1'][key]['x'][0]+hs_ind*0.003, 4400, width=0.003, label=key)
        hs_ind += 1

    plt.xticks(range_step)
    plt.legend()

    plt.subplot(122)
    plt.bar(range_step - 0.025/2, count_m2[0], width=0.025, color='blue', label='Class 0', alpha=0.5)
    plt.bar(range_step + 0.025/2, count_m2[1], width=0.025, color='red', label='Class 1', alpha=0.5)

    hs_ind = 1
    for key in metrics_m2.keys():
        plt.bar(-0.006+xymax['m2'][key]['x'][0]+hs_ind*0.003, 4400, width=0.003, label=key)
        hs_ind += 1

    plt.xticks(range_step)
    plt.show()

    print(f"\n{task}d")
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()

    df['Model_1_v2'] = 1 - df['Model_1_0']

    precision, recall, _ = skm.precision_recall_curve(df['GT'], df['Model_1_v2'])
    max_value = np.array((2 * precision * recall) / (precision + recall)).argmax()

    ax[0][0].plot(recall, precision)
    ax[0][0].plot(recall[max_value], precision[max_value], 'o', color='black')
    ax[0][0].set_title('PR крива, m1')
    ax[0][0].set_xlabel('Recall')
    ax[0][0].set_ylabel('Precision')

    precision, recall, _ = skm.precision_recall_curve(df['GT'], df['Model_2_1'])
    max_value = np.array((2 * precision * recall) / (precision + recall)).argmax()

    ax[1][0].plot(recall, precision)
    ax[1][0].plot(recall[max_value], precision[max_value], 'o', color='black')
    ax[1][0].set_title('PR крива, m2')
    ax[1][0].set_xlabel('Recall')
    ax[1][0].set_ylabel('Precision')

    fpr, tpr, _ = skm.roc_curve(df['GT'], df['Model_1_v2'])
    max_value = np.array(np.sqrt(tpr * (1 - fpr))).argmax()

    ax[0][1].plot(fpr, tpr)
    ax[0][1].plot(fpr[max_value], tpr[max_value], 'o', color='black')
    ax[0][1].set_title('ROC крива, m1')
    ax[0][1].set_xlabel('False Positive')
    ax[0][1].set_ylabel('True Positive')

    fpr, tpr, _ = skm.roc_curve(df['GT'], df['Model_2_1'])
    max_value = np.array(np.sqrt(tpr * (1 - fpr))).argmax()

    ax[1][1].plot(fpr, tpr)
    ax[1][1].plot(fpr[max_value], tpr[max_value], 'o', color='black')
    ax[1][1].set_title('ROC крива, m2')
    ax[1][1].set_xlabel('False Positive')
    ax[1][1].set_ylabel('True Positive')

    plt.tight_layout()
    plt.show()


task3(df, 3)


# --- 4 --- #
'''
Моделі є достатньо якісні, оскільки відносно добре передбачають клас
Перша модель краща, оскільки вона більш точно передбачає (видно з графіків)
'''

# --- 5 --- #
print('\n5.')
birthday = '2003-23'
month = birthday.split('-')
month = int(month[1])

K = month % 4
percent = (50 + 10*K)/100  # 0.8

indexes = df[df['GT'] == 1].sample(frac=percent).index
df = df.drop(indexes)


# --- 6 --- #
print('\n6.')
print(f"Відсоток видалення: {percent}")
print(f"Кількість:\n\tклас 0 - {len(df[df['GT'] == 0].values)}\n\tклас 1 - {len(df[df['GT'] == 1].values)}")


# --- 7 --- #
task3(df, 7)


# --- 8 --- #
'''
Друга модель стала кращою
'''


# --- 9 --- #
'''
Вплив замітний
'''


