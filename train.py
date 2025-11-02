import numpy as np
import pandas as pd
import os
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
import evaluation
from loss import InpaintingLoss
from net import PConvUNet
from dataset import DIV2K
from utils import save_ckpt

import importlib
importlib.reload(evaluation)
from evaluation import evaluate


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

def train(train_dataset_dir="DIV2K_train_HR", 
          validation_dataset_dir="DIV2K_valid_HR",
          image_size=256,
          cut_boxes_dict={3: 20, 32: 2},
          save_dir="save",
          learning_rate=5e-4,
          max_iter=900,
          batch_size=16,
          n_threads=8,
          save_model_interval=35,
          evaluate_model_interval=35,
          log_interval=5,
          device='cuda'):

    torch.backends.cudnn.benchmark = True
    device = torch.device(device)

    if not os.path.exists(save_dir):
        os.makedirs('{:s}/images'.format(save_dir))
        os.makedirs('{:s}/ckpt'.format(save_dir))

    size = (image_size, image_size)
    img_tf = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor()]) # transforms.Normalize(mean=opt.MEAN, std=opt.STD)

    dataset_train = DIV2K(train_dataset_dir, img_tf, cut_boxes_dict)
    dataset_val = DIV2K(validation_dataset_dir, img_tf, cut_boxes_dict)

    iterator_train = iter(data.DataLoader(
        dataset_train, batch_size=batch_size,
        sampler=InfiniteSampler(len(dataset_train)),
        num_workers=n_threads))
    print("dataset size: ",len(dataset_train))
    model = PConvUNet().to(device)

    start_iter = 0
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = InpaintingLoss().to(device)

    best = -float("inf")
    best_iter = None
    for i in tqdm(range(start_iter, max_iter)):
        model.train()

        image, mask, gt = [x.to(device) for x in next(iterator_train)]
        output, _ = model(image, mask)
        loss = criterion(image, mask, output, gt)

        if (i + 1) % log_interval == 0:
            print('Loss: ', loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(save_dir, i + 1),
                    [('model', model)], [('optimizer', optimizer)], i + 1)

        if (i + 1) % evaluate_model_interval == 0:
          model.eval()
          results = evaluate(model, dataset_val, device, '{:s}/images'.format(save_dir), None)
          print(results)

          df = pd.DataFrame(results)
          if not df.empty:
              mean_psnr = float(df['psnr_pred'].mean())

              if mean_psnr > best:
                  best = mean_psnr
                  best_iter = i + 1
                  torch.save({
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'iter': i + 1,
                      'mean_psnr': mean_psnr,
                  }, 'best_inpaint.pth')
                  print(f"New best model saved (iter {i+1}) mean PSNR {mean_psnr:.4f}")
          else:
              print("Eval returned no results — skipping best-model update.")



if __name__ == "__main__":
    train()
