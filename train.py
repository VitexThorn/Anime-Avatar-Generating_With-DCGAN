# coding:utf8
import os
import ipdb
import torch as t
import torchvision as tv
import tqdm
from model import NetG, NetD
from torchnet.meter import AverageValueMeter


class Config(object):

    """model: 定义判别器和生成器的初始化参数"""
    nz = 100        # 判别器的输入，即噪声的维度 (1*1*nz)
    ngf = 64        # 生成器feature map数
    ndf = 64        # 判别器feature map数
    
    """加载数据"""
    # 数据集存放路径(文件夹里只有图片)
    data_path = './face/'
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
    max_epoch = 500
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
    gen_img = 'result.png'


opt = Config()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device=t.device('cuda') if opt.gpu else t.device('cpu')
    print("device: ",device)
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)

    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    print("dataset loader")
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )

    # 网络
    print("net")
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)


    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()


    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        print("epoch:" ,epoch)
        fix_fake_imgs=None
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            if ii % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                ## 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                ## 尽可能把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 根据噪声生成假图
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

        # if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
        if opt.vis:
            ## 可视化
            if os.path.exists(opt.debug_file):
                ipdb.set_trace()
            fix_fake_imgs = netg(fix_noises)
            vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
            vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
            vis.plot('errord', errord_meter.value()[0])
            vis.plot('errorg', errorg_meter.value()[0])
            

        if (epoch+1) % opt.save_every == 0:
            # 保存模型、图片
            if fix_fake_imgs is not None:
                tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,range=(-1, 1))
            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % (epoch+300))
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % (epoch+300))
            errord_meter.reset()
            errorg_meter.reset()

    
if __name__ == '__main__':
    # import fire
    # fire.Fire()
    train()
    