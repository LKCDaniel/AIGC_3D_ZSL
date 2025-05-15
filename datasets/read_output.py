import glob
from os.path import join

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

test_class = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'nightstand', 'sofa', 'table', 'toilet']

# names = ['16_depth_black_42_60_210_PL8_CH20_comfyui',
#          '16_edge_black_42_60_210_PL8_CH20_comfyui_scribble',
#          '16_edge_black_42_60_210_PL8_CH20_comfyui_canny']
#
# name = names[0]
# labels = np.load(os.path.join(name, name + '_label.npy'))
# preds = np.load(os.path.join(name, name + '_pred.npy'))
# probs = np.load(os.path.join(name, name + '_prob.npy'))
#
# cm = confusion_matrix(labels, preds)
#
# accuracy = np.sum(labels == preds) / len(labels)
# label_classes = np.unique(labels)
# pred_classes = np.unique(preds)
#
# # plot the confusion matrix
# plt.figure(figsize=(10, 8))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title(f'Confusion Matrix: {name}, accuracy {accuracy:.4f}')
# plt.colorbar()
# plt.xticks(np.arange(len(test_class)), test_class, rotation=45)
# plt.yticks(np.arange(len(test_class)), test_class)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# for i in range(len(test_class)):
#     for j in range(len(test_class)):
#         plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
# plt.tight_layout()
# plt.show()


test_file_list = glob.glob('./ModelNet10/ModelNet10/*/test/*.off')
test_file_list.sort()

for output_dir in glob.glob('./ModelNet10/results/*'):
    if os.path.basename(output_dir) == 'plots': continue

    label_list = np.load(join(output_dir, 'label.npy'))
    pred_list = np.load(join(output_dir, 'pred.npy'))
    probs_list = np.load(join(output_dir, 'prob.npy'))
    filename_list = []
    output_name = os.path.basename(output_dir)


    for f_path in test_file_list:
        file_name = os.path.basename(f_path).split('.')[0]
        filename_list.append(file_name)

    with open(join(output_dir, 'result.txt'), 'w') as f:
        filename_width = 17
        label_width = 12
        tf_width = 6
        pred_width = 10

        f.write(f"{'file':<{filename_width}}{'prediction':<{label_width}}{'T/F':<{tf_width}}")
        for class_n in test_class:
            f.write(f"{class_n:<{pred_width}}  ")

        f.write('\n' + '-' * (filename_width + label_width + 60) + '\n')

        for i, label in enumerate(label_list):
            f.write(f"{filename_list[i]:<{filename_width}}")
            f.write(f"{test_class[pred_list[i]]:<{label_width}}")
            f.write(f"{'T' if label == pred_list[i] else 'F':<{tf_width}}")
            for j in probs_list[i]:
                f.write(f"{j:<{pred_width}}  ")
            f.write('\n')

    acc = np.sum(label_list == pred_list) / len(label_list)
    per_accuracy = []
    for i, cl in enumerate(test_class):
        cl_i = np.where(label_list == i)[0]
        per_accuracy.append(np.sum(pred_list[cl_i] == i) / len(cl_i))

    cm = confusion_matrix(label_list, pred_list)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix, acc {acc:.4f}, {output_name}')
    plt.colorbar()
    plt.xticks(np.arange(len(test_class)), test_class, rotation=45)
    y_labels = []
    for i, name in enumerate(test_class):
        y_labels.append(name + f'\n{per_accuracy[i]:.4f}')
    plt.yticks(np.arange(len(test_class)), y_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(test_class)):
        for j in range(len(test_class)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()

    if os.path.isfile(join(output_dir, 'result.png')):
        os.remove(os.path.join(output_dir, 'result.png'))

    plt.savefig(join(output_dir, output_name + '.png'))
    plt.savefig(join('ModelNet10', 'results', 'plots', output_name + '.png'))
