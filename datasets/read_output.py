import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

test_class = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'nightstand', 'sofa', 'table', 'toilet']

names = ['16_depth_black_42_60_210_PL8_CH20_comfyui',
         '16_edge_black_42_60_210_PL8_CH20_comfyui_scribble',
         '16_edge_black_42_60_210_PL8_CH20_comfyui_canny']

name = names[0]
labels = np.load(os.path.join(name, name + '_label.npy'))
preds = np.load(os.path.join(name, name + '_pred.npy'))
probs = np.load(os.path.join(name, name + '_prob.npy'))

cm = confusion_matrix(labels, preds)

accuracy = np.sum(labels == preds) / len(labels)
label_classes = np.unique(labels)
pred_classes = np.unique(preds)

# plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix: {name}, accuracy {accuracy:.4f}')
plt.colorbar()
plt.xticks(np.arange(len(test_class)), test_class, rotation=45)
plt.yticks(np.arange(len(test_class)), test_class)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(len(test_class)):
    for j in range(len(test_class)):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.show()
