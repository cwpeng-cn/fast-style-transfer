import time
from torchvision import transforms
from model.transformer_net import TransferNet
from utils import *

net = TransferNet()
net = restore_network("storage", "last", net)

m_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

content_img = get_image("./images/test image.jpg", m_transform)

start_time = time.time()
output_image = net(content_img)
infer_time = time.time() - start_time
print("推理时间为：", infer_time)
show_image(output_image.cpu().data)
save_image(output_image.cpu().data, "images/output image.jpg")


print("模型转换开始")
x = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(func=net, example_inputs=x)
traced_script_module.save("fast_transfer.pt")
print("模型转换结束，已保存为fast_transfer.pt")
