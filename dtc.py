import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# 读取数据，数据包含feature1、feature2、二进制标签bin_label和输出变量label
data = pd.read_csv('DataWithLabel.csv')

# 准备输入特征和输出变量
# x columns: th,sh,ch,er,ing,ou,en,eu,ai,oe,kn,ph,fj,wr,qu,Collision
X = data[['th','sh','ch','er','ing','ou','en','eu','ai','oe','kn','ph','fj','wr','qu','Collision']]
y = data['label']

# 创建决策树模型，设置最大深度为3
model = DecisionTreeClassifier(max_depth=3)

# 拟合模型
model.fit(X, y)

# 生成Graphviz格式的决策树图
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=X.columns,  
                           class_names=['Generally easy', 'varying in difficulty', 'generally difficult', 'generally moderate,'],  
                           filled=True, rounded=True,  
                           special_characters=True)

# 展示决策树图
import graphviz
graph = graphviz.Source(dot_data)
# high resolution
graph.render('decision_tree', format='png')