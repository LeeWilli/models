import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1)
performance = np.array([[0.43,0.42,0.47,0.06],
                         [0.40,0.28,0.32,0.19],
                         [0.57,0.52,0.56,0.40],
                         [0.33,0.38,0.44,0],
                         [0.30,0.23,0.29,0.15],
                         [0.75,0.69,0.76,0.53]])
df = pd.DataFrame(performance,
                  index=['mAP', 'car', 'person', 'truck', 'boat', 'dog'],
                  columns=pd.Index(['yolo-coco', 'ssd_mobilenet','ssd_inception','yolo-voc'], name='model'))
df.plot.bar(ax=axes[0])
axes[0].set_xlabel("class", fontsize=12)
axes[0].set_ylabel("precision", fontsize=12)
efficience = np.array([1265,281,359,629])
data = pd.Series(efficience, index=list(['yolo-coco', 'ssd_mobilenet','ssd_inception','yolo-voc']))
data.plot.barh(ax=axes[1])
axes[1].set_xlabel("s", fontsize=12)
axes[1].set_ylabel("model", fontsize=12)

plt.show()