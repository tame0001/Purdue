import argparse
import torch
import network
import utils
import json
from pathlib import Path
from dataloader import HW7Dataset
from torch.utils.data import DataLoader

LETENT = 100
BETAS = (0.5, 0.9)
LR = 1e-5
'''
--------------------------------------------------------------------------------
'''
# Choose numbers of iteration when execution
parser = argparse.ArgumentParser(description='ECE60146 HW7.')
parser.add_argument(
    '--epoch', '-n', type=int, default=500,
    help='number of training iterations'
)

parser.add_argument(
    '--pretrain', '-p', action='store_true',
    help='use previous train weight'
)

args = vars(parser.parse_args())
path = Path('/home/tam/git/ece60146/data') # Define dataset location.
# Use cuda 1 because other people in my lab usually use cuda 0
if torch.cuda.is_available():
    device = "cuda:1"
num_workers = 2
batch_size = 64
# Checking message
print(f"Using {device} device with {num_workers} workers")
# Load training dataset
train_dataset = HW7Dataset(path, 'train')
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
) 
utils.show_pizzas(next(iter(train_dataloader))['image'], 'training sample')
m, s = utils.get_real_pizza_features(train_dataset, device)
'''
--------------------------------------------------------------------------------
BCE-GAN
'''
netD = network.Critic().to(device)
network.initialize_weights(netD)
netG = network.Generator(LETENT).to(device)
network.initialize_weights(netG)
# Optimizers
optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=BETAS)
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=BETAS)
# Training
records = {
    'real': [], # loss on real pizza classification
    'fake': [], # loss on fake pizza classification
    'total': [], # loss on  classification
    'gen': [], # generation loss
    'fid': [],
    'epoch': []
}
for epoch in range(args['epoch']):
    real_pred_loss_accumulate = 0
    fake_pred_loss_accumulate = 0
    gen_loss_accumulate = 0
    records['epoch'].append(epoch+1)
    for batch, data in enumerate(train_dataloader):
        # Discrimation phase
        netD.zero_grad()
        real_pizzas = data['image'].to(device) # Classify real pizzas
        output = netD(real_pizzas).view(-1) # Ideally should be vector of 1
        real_pred_loss = torch.nn.BCELoss()(output, torch.full_like(output, 1))
        real_pred_loss_accumulate += real_pred_loss.item()
        real_pred_loss.backward()
        # Generate fake pizzas
        noise = torch.randn(batch_size, LETENT, 1, 1, device=device)
        fake_pizzas = netG(noise)
        output = netD(fake_pizzas).view(-1) # Ideally should be vector of 0
        fake_pred_loss = torch.nn.BCELoss()(output, torch.full_like(output, 0))
        fake_pred_loss_accumulate += fake_pred_loss.item()
        fake_pred_loss.backward()
        optimizerD.step() # Update discriminator

        # Generation phase
        netG.zero_grad()
        fake_pizzas = netG(noise) # Get new pizzas since the previous  were used
        output = netD(fake_pizzas).view(-1) # Get new result after update netD
        # To update generator, assume that generated pizzas are real
        gen_loss = torch.nn.BCELoss()(output, torch.full_like(output, 1))
        gen_loss_accumulate += gen_loss.item()
        gen_loss.backward()
        optimizerG.step()
        
    records['real'].append(real_pred_loss_accumulate/len(train_dataloader))
    records['fake'].append(fake_pred_loss_accumulate/len(train_dataloader))
    records['total'].append(records['real'][-1] + records['fake'][-1])
    records['gen'].append(gen_loss_accumulate/len(train_dataloader))
    if (epoch+1) % 10 == 0: # Tracking learning progress
        fid = utils.cal_fid_score('BCE-GAN', m, s, netG, device)
        records['fid'].append((epoch+1, fid))
        print(f'Epoch No. {" "*5}Real Pizza'
              f'{" "*5}Fake Pizza'
              f'{" "*5}Generation'
              f'{" "*5}FID Score')
        print(f'  {epoch+1:4}'
              f'{" "*10}{real_pred_loss_accumulate/(len(train_dataloader)):.7f}'
              f'{" "*6}{fake_pred_loss_accumulate/(len(train_dataloader)):.7f}'
              f'{" "*6}{gen_loss_accumulate/(len(train_dataloader)):7.4f}'
              f'{" "*6}{fid:8.2f}')
        utils.show_pizzas(
            netG(torch.randn(batch_size, LETENT, 1, 1, device=device)), 
            f'BCE-GAN {epoch+1:03}'
        )
        with open('BCE-GAN.json', 'w') as fp:
            fp.write(json.dumps(records))

