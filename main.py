import torch
from torchvision.transforms import transforms
from data import ZSSRdataset, ZSSRsampler
from utils import *
from model import ZSSRnet
from train import train, final_output
import numpy as np
from PIL import Image

num_batches = 3000
s_factor = 1.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
lres_img = Image.open('./images/1.jpg')

dataset = ZSSRdataset.from_image(lres_img, s_factor)
data_sampler = ZSSRsampler(dataset)
print("dataset-len：")
print(dataset.__len__())

model = ZSSRnet()
model.to(device)

train(lres_img,
      model,
      data_sampler,
      num_batches,
      s_factor,
      device)

torch.save(model, './model/zssr1.pt')

hres_false = bicubic_resample(lres_img, s_factor)
hres_false.save('./result/zssr_x1.5_interpolated.png')

model = torch.load('./model/zssr1.pt')
final_zssr, outputs = final_output(lres_img, model, bicubic_resample, s_factor, back_projection)
final_zssr.save('./result/zssr_x1.5_net_out.png')


# model = torch.load('./model/zssr1.pt')
num_batches = 3000
s_factor = 2 / 1.5
# device = "cuda"
dataset2 = ZSSRdataset.from_image(final_zssr, s_factor)
data_sampler2 = ZSSRsampler(dataset2)
train(final_zssr,
      model,
      data_sampler2,
      num_batches,
      s_factor,
      device)
torch.save(model, './model/zssr1.pt')


hres_false = bicubic_resample(final_zssr, s_factor)
hres_false.save('./result/zssr_x2_interpolated.png')
model = torch.load('./model/zssr1.pt')
final_zssr2, outputs2 = final_output(final_zssr, model, bicubic_resample, s_factor, back_projection)
final_zssr2.save('./result/zssr_x2_net_out.png')


direct_zssr = Image.open('./result/zssr_x2_net_out.png')
print("direct zssr size:")
print(direct_zssr.size)
print("final zssr size:")
print(final_zssr2.size)

direct_zssr = direct_zssr.resize([final_zssr2.size[0], final_zssr2.size[1]], resample=Image.BICUBIC)


direct_zssr_tensor = transforms.ToTensor()(direct_zssr)
final_zssr2_tensor = transforms.ToTensor()(final_zssr2)

residual = torch.abs(direct_zssr_tensor - final_zssr2_tensor)
residual_img = transforms.ToPILImage()(residual)
residual_img.save('./result/zssr_x2_residual.png')



