import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 매개변수와 DataLoaders
input_size = 5
output_size = 2

batch_size = 20
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 랜덤으로 데이터를 던져주는 데이터셋 생성
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


# 간단한 모델 작성하기
class Model(nn.Module):
    # 우리의 모델

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


# 데이터와 모델 병렬 구현
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)


for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())