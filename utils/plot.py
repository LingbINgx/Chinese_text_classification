import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


matplotlib.rcParams['font.sans-serif'] = [  'SimHei',
                                            'Microsoft YaHei',
                                            'Noto Sans CJK SC',
                                            'WenQuanYi Zen Hei'
                                        ]
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format='.3f', colorbar=False)
    plt.title('Confusion Matrix')
    #plt.show()
    fig.savefig(f"../image/{title}.png", dpi=300, bbox_inches="tight")
    

if __name__ == "__main__":
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]
    labels = ['一你好世界', '二再见', '三深度学习']
    plot_confusion_matrix(y_true, y_pred, labels)
