import pandas as pd
import matplotlib.pyplot as plt

# 读取Combined.csv文件
df_combined = pd.read_csv("Combinedact.csv")

# 读取predict23.csv文件
df_predict = pd.read_csv("predict23.csv")

import numpy as np
#add noise to predict
df_predict["Number of reported results"] = df_predict["Number of reported results"] * (1 + np.random.normal(0, 0.05, df_predict.shape[0]))

# remove the first row
df_predict = df_predict.drop(0)

# 将两个DataFrame对象连接起来
df_all = pd.concat([df_combined, df_predict], ignore_index=True)

# 绘制折线图
plt.plot(df_all["Date"], df_all["Number of reported results"], label="Combined")
plt.plot(df_predict["Date"], df_predict["Number of reported results"], label="Predict", linestyle="--")
plt.legend()
plt.title("Number of reported results over time")
plt.xlabel("Date")
plt.ylabel("Number of reported results")
x_ticks = df_all["Date"][::90]
plt.xticks(x_ticks)
plt.savefig('predict.png', dpi=300)
plt.show()