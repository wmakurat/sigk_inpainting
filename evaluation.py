import torch
from torchvision.utils import make_grid, save_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import random
from tqdm import tqdm

def tensor_to_uint8_image(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def compute_psnr(hr_uint8, pred_uint8, data_range=255.0):
    return compare_psnr(hr_uint8, pred_uint8, data_range=data_range)

def compute_ssim_safe(hr_uint8, pred_uint8):
    try:
        return compare_ssim(hr_uint8, pred_uint8, data_range=255.0, channel_axis=2, win_size=7)
    except TypeError:
        return compare_ssim(hr_uint8, pred_uint8, data_range=255.0, multichannel=True, win_size=7)

def compute_snr_db(hr_tensor, pred_tensor):
    hr = hr_tensor.cpu().numpy()
    pr = pred_tensor.cpu().numpy()
    signal_power = np.sum(hr**2)
    noise_power = np.sum((hr - pr)**2)
    if noise_power <= 1e-12:
        return float('inf')
    return 10.0 * np.log10(signal_power / noise_power)

def compute_lpips(lpips_fn, hr_tensor, pred_tensor):
    hr_n = hr_tensor.unsqueeze(0) * 2.0 - 1.0
    pr_n = pred_tensor.unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        d = lpips_fn(hr_n, pr_n, normalize=True)
    return float(d.mean().cpu().numpy())

def evaluate(
    model,
    loader,
    device,
    lpips_fn=None,
):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            masked = batch[0]
            mask = batch[1]            
            clean = batch[2]

            pred = model(masked, mask)

            b = masked.size(0)
            for i in range(b):
                clean_uint8 = tensor_to_uint8_image(clean[i])
                masked_uint8 = tensor_to_uint8_image(masked[i])
                pred_uint8 = tensor_to_uint8_image(pred[i])

                psnr_pred = compute_psnr(clean_uint8, pred_uint8)
                ssim_pred = compute_ssim_safe(clean_uint8, pred_uint8)
                snr_pred = compute_snr_db(clean[i], pred[i])

                lp_pred = None
                if lpips_fn is not None:
                    lp_pred = compute_lpips(lpips_fn, clean[i], pred[i])

                results.append({
                    'path': paths[i],
                    'psnr_pred': psnr_pred,
                    'ssim_pred': ssim_pred,
                    'snr_pred_db': snr_pred,
                    'lpips_pred': lp_pred
                })
    return results
                
