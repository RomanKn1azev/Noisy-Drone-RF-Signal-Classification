import matplotlib.pyplot as plt


from src.utils.file import load_yml_file, join_path_file_name


def visualize_f1_score(
        data_path: str, 
        average: str, 
        save_path: str=None, 
        name: str=None, 
        type: str=None, 
        show: bool=True
        ):
    f1_scores = load_yml_file(data_path)

    plt.figure(figsize=(10, 5))

    plt.plot(f1_scores)

    plt.title(f'Multiclass F1-Score (average={average})')

    plt.xlabel('Batch Number')

    plt.ylabel('F1-Score')

    plt.grid()

    if save_path is not None:
        plt.savefig(join_path_file_name(save_path, name, type))

    if show:
        plt.show()


def visualize_multiclass_accuracy(
        data,
        save_path: str=None, 
        name: str=None, 
        type: str=None, 
        show: bool=True
        ):
    

    plt.figure(figsize=(10, 5))

    # plt.bar(range(len()), )

    plt.title('Multiclass Accuracy per Batch')

    plt.xlabel('Batch')

    plt.ylabel('Accuracy')

    plt.grid()

    if save_path is not None:
        plt.savefig(join_path_file_name(save_path, name, type))

    if show:
        plt.show()


def visualize_multiclass_auroc(
        AUC_data_path: str,
        MulticlassAUROC_data_path: str,
        num_classes: int,
        save_path: str=None, 
        name: str=None, 
        type: str=None, 
        show: bool=True
        ):
    # roc_curves = []
    # for class_index in range(num_classes):
    #     fpr, tpr, thresholds = metric.roc_curve(class_index)
    #     roc_curves.append((fpr, tpr))

    # # Plot the ROC curve
    # plt.figure(figsize=(10, 6))
    # for i, roc_curve in enumerate(roc_curves):
    #     plt.plot(roc_curve[0], roc_curve[1], label=f'Class {i}')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'MulticlassAUROC (AUC = {AUC:.3f})')
    # plt.legend()

    ...


def visualize_multiclass_recall(        
        data_path: str, 
        save_path: str=None, 
        name: str=None, 
        type: str=None, 
        show: bool=True
        ):
    rec = load_yml_file(data_path)

    plt.plot(rec)

    plt.xlabel('Batch')

    plt.ylabel('MulticlassRecall')

    plt.title('MulticlassRecall per Batch')

    plt.grid()

    if save_path is not None:
        plt.savefig(join_path_file_name(save_path, name, type))

    if show:
        plt.show()


def visualize_multiclass_prc():
    ...