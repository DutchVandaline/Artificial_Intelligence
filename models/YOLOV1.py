from torch import nn
import torch.nn.functional as F

class YOLOV1(nn.Module):
  def __init__(self, input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()
    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size= 7,stride=2,padding=3),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )

    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units,out_channels = 192 ,kernel_size = 3,padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
    )

    self.block_3 = nn.Sequential(
        nn.Conv2d(in_channels = 192,out_channels = 128,kernel_size = 1,padding = 0),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 3,padding = 1,),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=256,out_channels = 256,kernel_size = 1,padding = 0),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size = 3,padding = 1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size =2,stride = 2)
    )

    self.block_4 = nn.Sequential(
    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.LeakyReLU(),

    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.LeakyReLU(),

    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.LeakyReLU(),

    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.LeakyReLU(),

    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0),
    nn.LeakyReLU(),

    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)


    self.block_5 = nn.Sequential(
      nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
      nn.LeakyReLU(),

      nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
      nn.LeakyReLU(),

      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
      nn.LeakyReLU(),

      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2),
      nn.LeakyReLU(),
  )
    self.block_6 = nn.Sequential(
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=0,),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1,),
        nn.LeakyReLU(),
    )

    self.flatten = nn.Flatten()

    self.block_7 = nn.Linear(in_features=1024 * 5 * 5, out_features=4096)

    self.block_8 = nn.Linear(in_features=4096, out_features=output_shape)




  def forward(self, x:torch.Tensor):
    x = self.block_1(x)
    x = self.block_2(x)
    x = self.block_3(x)
    x = self.block_4(x)
    x = self.block_5(x)
    x = self.block_6(x)

    x = self.flatten(x)

    x = self.block_7(x)
    x = self.block_8(x)

    x = F.softmax(x, dim=1)

    return x

