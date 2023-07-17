import random
import time
import torchvision.transforms as tvt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

class DemoDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        # define a folder
        self.folder = Path('.') / path
        self.filenames = [] # keep filename
        for filename in self.folder.iterdir():
            # filter some images out
            if filename.name[:3] == 'PXL':
                self.filenames.append(filename)

        self.augment = tvt.Compose([
            tvt.Resize([640, 360]), # keep 16:9 ratio
            # reduce brightness and saturation upto half
            tvt.ColorJitter(brightness=(0.5, 1), saturation =(0.5, 1)),
            tvt.transforms.RandomHorizontalFlip(0.5), # 50% chance to flip
            tvt.RandomCrop(size=(256, 256)), # crop to 256x256
            tvt.RandomPerspective(distortion_scale=0.2), # add distrotion
            tvt.ToTensor(),
            ])
        
    def __len__(self):
        # assuming there are 1000 samples
        return 1000

    def __getitem__(self, index):
        index = index % len(self.filenames)
        image = Image.open(self.filenames[0])
        tensor = self.augment(image)
        target = random.randint(0, 10)
        return tensor, target

dataset = DemoDataset('hw2')
print(len(dataset))
# get a sample and visualize
data = dataset[9][0]
tvt.ToPILImage()(data).save(Path('.') / 'temp.png')

# get a batch of 4 samples and save them
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(dataloader))
for id, sample in enumerate(batch[0]):
    tvt.ToPILImage()(sample).save(Path('.') / f'temp{id}.png')

# time loading data with dataset
start_time = time.time()
for index in range(1000):
    data = dataset[index]
print(f'Loading 1,000 samples by calling dataset takes {(time.time()-start_time):.2f} seconds')

# test performace with different parameters
for worker in [1, 2, 4, 8]:
    for batch_size in [1, 4, 16, 64]:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=worker, shuffle=True)
        start_time = time.time()
        for tensors, targets in dataloader:
            pass
        print('Loading 1,000 samples by dataload batch size = {} '
        'and number workers = {} takes {:.3f} seconds'
        .format(batch_size, worker, time.time()-start_time))