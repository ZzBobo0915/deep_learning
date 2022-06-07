import time
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import torchvision
import resnet50

# todo: 读取常用数据集
def load_data_fashion_mnist(batch_size, resize=None, root='./Datasets/'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    # 是否需要resize，默认插值方法为BILINEAR
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)  # 通过Compose将trans里的多个步骤合到一起

    # torchvision.datasets包含了目前流行的数据集，模型结构和图片转换工具，用这个可以快速读取数据
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    """
    torch.utils.data.DataLoader()用来输入数据和标签，常用参数如下：
        dataset:表示Dataset类，决定了读取的数据
        batch_size:每次处理的数据批量大小，一般为2的次方，如2,4,8,16,32,64等等
        shuffle:是否随机读入数据，在训练集的时候一般随机读入，在验证集的时候一般不随机读入
        num_works:多线程传入数据，设置的数字即使传入的线程数，可以加快数据的读取
        drop_last:如果数据集的大小不能被批大小整除，当样本数不能被batch_size整除时，是否舍弃最后一批数据
    """
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #print(train_iter)

    return train_iter, test_iter

# todo: 转换自己的数据集
# 需要继承torch.utils.data.Dataset，并且重写__getitem__()和__len__()类方法，传入resize后的tensor数据
class MyDataset(torch.utils.data.Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

# todo: 读取自己的数据集
def load_data_MyDataset(data_tensor, target_tensor, batch_size, train_or_test='train', num_workers=0):
    my_dataset = MyDataset(data_tensor, target_tensor)
    if train_or_test == 'train':
        iter = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif train_or_test == 'test':
        iter = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        print("check your param : train_or_test!")
    return iter

# todo: 自己设定损失函数，需要继承nn.Module
# 锐哥这部分你好好研究一下，感觉贺也得猛问，看完了记得删除这行
class cross_entropy_loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(cross_entropy_loss, self).__init__()
        self.reduction = reduction  # 用来指定损失结果返回的是mean、sum
    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

# todo: 计算测试集准确率
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            # 因为FashionMNIST输入为单通道图片，需要转换为三通道
            X = np.array(X)
            X = X.transpose((1, 0, 2, 3))  # array 转置
            X = np.concatenate((X, X, X), axis=0)
            X = X.transpose((1, 0, 2, 3))  # array 转置回来
            X = torch.tensor(X)  # 将 numpy 数据格式转为 tensor

            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else:
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# todo: 训练函数
def train(net, train_iter, test_iter, optimizer, device, num_epochs):
    print("training on : ", device)
    # 保存精度用来绘图
    Train_acc, Test_acc = [0], [0]
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n----------------------")
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            # 因为FashionMNIST输入为单通道图片，需要转换为三通道
            X = np.array(X)
            X = X.transpose((1, 0, 2, 3))  # array 转置
            X = np.concatenate((X, X, X), axis=0)  # 维度拼接
            X = X.transpose((1, 0, 2, 3))  # array 转置回来
            X = torch.tensor(X)  # 将 numpy 数据格式转为 tensor
            # 将数据移到gpu上
            X = X.to(device)
            y = y.to(device)
            # 得到预测结果
            y_hat = net(X)
            # 计算损失
            l = loss(y_hat, y)
            optimizer.zero_grad()  # 梯度清零
            l.backward()  # 计算反向传播
            optimizer.step()  # 梯度下降，参数更新
            # cpu()函数作用是将数据从GPU上复制到memory上，item()返回的是一个数值而非tensor，想要返回得到tensor要用cpu().data
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        # print("train loss : %.4f, train acc : %.3f" %(train_l_sum / batch_count, train_acc_sum / n))
        # 每个epoch的结果输出到控制台并保存数据以便最后绘制精度曲线图像/损失曲线图像
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        Train_acc.append(train_acc_sum / n)
        Test_acc.append(test_acc)
        if epoch == num_epochs-1:
            torch.save(net.state_dict(), "./last_model.pth")  # 权重保存

    # 保存精度与迭代次数图像
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xlim(0, 10)
    plt.plot(np.arange(len(Train_acc)), Train_acc, label='train_acc')
    plt.plot(np.arange(len(Test_acc)), Test_acc, label='test_acc')
    plt.savefig('./acc_result.png')
    print("Done!")

# 使用GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 网络Resnet50，FashionMNIST为10类
net = resnet50.ResNet50(num_classes=10).to(device)
# 交叉熵损失函数
#loss = torch.nn.CrossEntropyLoss()
loss = cross_entropy_loss()
# 批量大小
batch_size = 64
# 训练和测试数据集划分
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
# 学习率和迭代轮次
lr, num_epochs = 0.0001, 10
# 优化器采用Adam
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#开始训练
train(net, train_iter, test_iter, optimizer, device, num_epochs)