import time
from torchvision import transforms
from model.transformer_net import TransferNet
from utils import *

net = TransferNet()
net = restore_network("storage", "fast_style_transfer3", net)

m_transform = transforms.Compose([
    transforms.Resize((640, 480)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

content_img = get_image("images/test image3.jpg", m_transform)

start_time = time.time()
output_image = net(content_img)
infer_time = time.time() - start_time
print("推理时间为：", infer_time)
show_image(output_image.cpu().data)
save_image(output_image.cpu().data, "images/output image.jpg")
