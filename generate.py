# coding:utf8

import torch as t
import torchvision as tv
from model import NetG, NetD



class Config(object):

    """model: 定义判别器和生成器的初始化参数"""
    nz = 100        # 判别器的输入，即噪声的维度 (1*1*nz)
    ngf = 64        # 生成器feature map数
    ndf = 64        # 判别器feature map数
    
    """加载数据"""
    # 数据集存放路径(文件夹里只有图片)
    data_path = './faces'
    # 多进程加载数据所用的进程数('win'中，该值设置为0)  
    num_workers = 0     
    # 数据集中的图片尺寸，图片大小都是固定的，为3*96*96
    image_size = 96     
    # 小批量值
    batch_size = 256
    
    """训练所用到的超参数"""
    # 是否使用GPU
    gpu = True      
    # 训练迭代次数
    max_epoch = 300
    # 生成器和判别器的学习率
    lr1 = 2e-4      # 生成器的学习率
    lr2 = 2e-4      # 判别器的学习率
    # 优化器参数: Adam优化器的beta1参数
    beta1 = 0.5
    # 每1个batch训练一次判别器 
    d_every = 1
    # 每5个batch训练一次生成器
    g_every = 5
    
    """可视化生成器训练过程"""
    # 是否使用visdom可视化
    vis = False
    # visdom的env名字
    env = 'GAN'
    # 每间隔plot_every个batch，visdom画图一次
    plot_every = 20
    # 存在该文件则进入debug模式
    debug_file = '/tmp/debuggan'
    
    """保存训练过程中，判别器生成的图片 """
    # 保存图片的路径
    save_path = 'visdom'
    # 每save_every个epoch保存一次模型权重和生成的图片，
    # 权重文件默认保存在checkpoints, 生成图片默认保存在save_path
    save_every = 50

    """预训练模型"""
    # netd_path = None 
    # netg_path = None
    netd_path = 'checkpoints/netd_299.pth'
    netg_path = 'checkpoints/netg_299.pth'


    """使用训练好的生成器，生成图片"""
    # 从512张生成的图片中保存最好的64张
    gen_search_num = 512;gen_num = 64
    # 噪声均值和方差
    gen_mean = 0
    gen_std = 1  
    # result
    gen_img = './static/images/result.png'


opt = Config()

@t.no_grad()
def generate(**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device=t.device('cuda') if opt.gpu else t.device('cpu')
    print("device: ",device)

    netg, netd = NetG(opt).eval(), NetD(opt).eval()

    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).detach()

    # 挑选最好的某几张
    print("fake img")
    indexs = scores.topk(opt.gen_num)[1]
    print("indexs: ",indexs)
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    print("result")
    tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, value_range=(-1, 1))
    
if __name__ == '__main__':
    generate()