import numpy as np
import matplotlib.pyplot as plt

def visualize_predictions(X, y_true, y_pred, class_labels, num_images=10):
    indices = np.random.choice(len(X), num_images, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.5)

    for i, idx in enumerate(indices):
        ax = axes[i // 5, i % 5]
        ax.imshow(X[idx])
        true_label = class_labels[y_true[idx]]
        pred_label = class_labels[y_pred[idx]]
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        ax.axis('off')

plot.show()
