import numpy as np
import matplotlib.pyplot as plt

# 设置参数
epochs = 1000
start_loss = 10
end_loss = 1e-2
noise_scale = 0.15  # 噪声幅度
stable_epoch = 700  # 稳定的epoch

# 生成损失值
x = np.linspace(0, 1, epochs)
losses = start_loss * np.power(end_loss / start_loss, x)
losses[stable_epoch:] = end_loss  # 在稳定的epoch后，损失为一个常数
losses_with_noise = losses + np.random.normal(0, noise_scale, epochs)

# 绘制损失图
plt.plot(range(epochs), losses_with_noise)
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png', dpi=300)
plt.show()
