from torch import nn
from torch import optim
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from utils import *
from model.transformer_net import TransferNet
from model.loss_net import LossNet

LR = 0.001
EPOCH = 2
BATCH_SIZE = 4
IMAGE_SIZE = 224
STYLE_WEIGHTS = [i * 2 for i in [1e2, 1e4, 1e4, 5e3]]
DATASET = "../datasets/Intel_image_classification/seg_train/seg_train"

m_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_dataset = datasets.ImageFolder(DATASET, m_transform)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
style_img = get_image("./images/style image.jpg", m_transform).to(device)

# 定义网络
transferNet = TransferNet().to(device)
lossNet = LossNet().to(device)
# 定义损失
mse = nn.MSELoss()
# 定义优化器
optimizer = optim.Adam(transferNet.parameters(), LR)
style_feature = lossNet(style_img.repeat(BATCH_SIZE, 1, 1, 1))
style_target = [gram_matrix(f).detach() for f in style_feature]
# 生成网络可训练，损失网络固定
transferNet.train()
lossNet.eval()

step = 0
for i in range(EPOCH):
    for contents_imgs, _ in train_loader:
        contents_imgs = contents_imgs.cuda()
        optimizer.zero_grad()
        generate_imgs = transferNet(contents_imgs)
        generate_features = lossNet(generate_imgs)
        style_generate = [gram_matrix(f) for f in generate_features]
        content_generate = generate_features[1]
        content_features = lossNet(contents_imgs)
        content_target = content_features[1].detach()
        content_loss = mse(content_generate, content_target)

        style_loss = 0
        for j in range(len(STYLE_WEIGHTS)):
            style_loss += STYLE_WEIGHTS[j] * mse(style_generate[j], style_target[j])

        loss = content_loss + style_loss
        loss.backward()

        if step % 100 == 0:
            print(step, "  content loss:", content_loss.data, "    style loss:", style_loss)

        # if step % 600 == 0:
        #     show_image(contents_imgs.cpu().data, is_show=False)
        #     show_image(generate_imgs.cpu().data)

        if step % 1000 == 0:
            save_network("storage", transferNet, step)

        optimizer.step()

        step += 1
        if step > 13000:
            break
