import seaborn as sns
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

def traingraph(x_axis_epoch,res_train_loss, res_train_score, res_val_loss, res_val_score, train_confusion, valid_confusion, settings):
    print('Making graph...')
    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gray')
    fig = plt.figure()


    ax1 = fig.add_subplot(221)
    ax1.plot(x_axis_epoch, res_train_loss, 'r', label='Train', marker='o')
    ax1.plot(x_axis_epoch, res_val_loss, 'b', label='Valid', marker='+')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel("nEpoch")
    ax1.set_ylabel("Loss")
    plt.legend(loc="upper left")

    ax2 = fig.add_subplot(222)
    ax2.plot(x_axis_epoch, res_train_score, 'r', label='Train', marker='o')
    ax2.plot(x_axis_epoch, res_val_score, 'b', label='Valid', marker='+')
    ax2.set_title('f1-score Curve')
    ax2.set_xlabel("nEpoch")
    ax2.set_ylabel("f1-score")
    plt.legend(loc="upper left")

    ax3 = fig.add_subplot(223)
    train_confusion = pd.DataFrame(train_confusion,
                                   index=settings.config["LabelNameList"]["TrueLabel"],
                                   columns=settings.config["LabelNameList"]["PredLabel"])
    sns.heatmap(train_confusion, linewidth=0.3,
                ax=ax3, annot=True, square=False, fmt=".1f")

    ax3.set_title('Training Confusion Matrix')
    ax3.set_ylabel("True Label")
    ax3.set_xlabel("Predicted Label")

    valid_confusion = pd.DataFrame(valid_confusion,
                                   index=settings.config["LabelNameList"]["TrueLabel"],
                                   columns=settings.config["LabelNameList"]["PredLabel"])
    ax4 = fig.add_subplot(224)
    sns.heatmap(valid_confusion, linewidth=0.3,
                ax=ax4, annot=True, square=False, fmt=".1f")

    ax4.set_title('Validation Confusion Matrix')
    ax4.set_ylabel("Ture Label")
    ax4.set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.savefig(settings.config["System"]["OutputFileDir"]+ '/' + str(datetime.datetime.now()).replace("-","").replace(" ","_").replace(":","").replace(".","_") + "_TrainingResult.png")
    plt.close(fig)

def testgraph(loss, score, test_confusion, settings):
    print('Making graph...')

    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gray')
    fig = plt.figure()
    ax1 = fig.add_subplot()
    train_confusion = pd.DataFrame(test_confusion,
                                   index=settings.config["LabelNameList"]["TrueLabel"],
                                   columns=settings.config["LabelNameList"]["PredLabel"])
    sns.heatmap(train_confusion, linewidth=0.3,
                ax=ax1, annot=True, square=False, fmt=".1f")

    ax1.set_title('Test Confusion Matrix[Score:{:.3f}, loss:{:.3f}]'.format(score, loss))
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")


    fig.savefig(settings.config["System"]["OutputFileDir"]+ '/' + str(datetime.datetime.now()).replace("-","").replace(" ","_").replace(":","").replace(".","_") + "_TestResult.png")


def predgraph(score, pred_confusion, settings):
    print('Making graph...')

    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gray')
    fig = plt.figure()
    ax1 = fig.add_subplot()
    pred_confusion = pd.DataFrame(pred_confusion,
                                  index=settings.config["LabelNameList"]["TrueLabel"],
                                  columns=settings.config["LabelNameList"]["PredLabel"])
    sns.heatmap(pred_confusion, linewidth=0.3,
                ax=ax1, annot=True, square=False, fmt=".1f")

    ax1.set_title('Pred Confusion Matrix[Score:{:.3f}]'.format(score))
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")


    fig.savefig(settings.config["System"]["OutputFileDir"]+ '/' + str(datetime.datetime.now()).replace("-","").replace(" ","_").replace(":","").replace(".","_") + "_PredResult.png")