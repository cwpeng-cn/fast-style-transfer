from model.transformer_net import TransferNet
from utils import *

net = TransferNet()
net = restore_network("storage", "fast_style_transfer1", net)

print("模型转换开始")
x = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(func=net, example_inputs=x)
traced_script_module.save("fast_transfer.pt")
print("模型转换结束，已保存为fast_transfer1.pt")
