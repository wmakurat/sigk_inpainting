import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

try:
    import lpips
except Exception:
    lpips = None


def tensor_to_uint8_image(tensor):
    """
    tensor: (C, H, W), wartości w [0,1] (po normalizacji odwróć zanim tu trafisz)
    """
    t = tensor.detach().cpu().permute(1, 2, 0).float().numpy()
    t = np.clip(t, 0.0, 1.0)
    return (t * 255.0).astype(np.uint8)


def compute_psnr(hr_uint8, pred_uint8, data_range=255.0):
    return compare_psnr(hr_uint8, pred_uint8, data_range=data_range)


def compute_ssim_safe(hr_uint8, pred_uint8):
    try:
        return compare_ssim(hr_uint8, pred_uint8, data_range=255.0, channel_axis=2, win_size=7)
    except TypeError:
        return compare_ssim(hr_uint8, pred_uint8, data_range=255.0, multichannel=True, win_size=7)


def compute_snr_db(hr_tensor, pred_tensor):
    hr = hr_tensor.detach().cpu().float().numpy()
    pr = pred_tensor.detach().cpu().float().numpy()
    signal_power = np.sum(hr ** 2)
    noise_power = np.sum((hr - pr) ** 2)
    if noise_power <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_lpips(lpips_fn, hr_tensor, pred_tensor):
    if lpips_fn is None:
        return None
    hr_n = hr_tensor.unsqueeze(0) * 2.0 - 1.0
    pr_n = pred_tensor.unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        d = lpips_fn(hr_n, pr_n, normalize=True)
    return float(d.mean().detach().cpu().numpy())


def _wrap_loader_if_needed(loader, batch_size=16, num_workers=0):
    if isinstance(loader, DataLoader):
        return loader
    else:
        return DataLoader(loader, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate(model, loader, device, lpips_fn=None, batch_size=16, num_workers=0):
    """
    Zwraca listę dictów z metrykami dla każdego obrazu w walidacji.
    Wywołanie w Twoim trainie:  results = evaluate(model, dataset_val, device, None)
    """
    dl = _wrap_loader_if_needed(loader, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Eval"):
            masked, mask, clean = [x.to(device) for x in batch[:3]]

            pred, _ = model(masked, mask)

            b = masked.size(0)
            paths = None
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                paths = batch[3]

            for i in range(b):
                clean_u8 = tensor_to_uint8_image(clean[i])
                pred_u8  = tensor_to_uint8_image(pred[i])

                psnr_pred = compute_psnr(clean_u8, pred_u8)
                ssim_pred = compute_ssim_safe(clean_u8, pred_u8)
                snr_pred  = compute_snr_db(clean[i], pred[i])
                lp_pred   = compute_lpips(lpips_fn, clean[i], pred[i]) if lpips_fn is not None else None

                item = {
                    'psnr_pred': psnr_pred,
                    'ssim_pred': ssim_pred,
                    'snr_pred_db': snr_pred,
                    'lpips_pred': lp_pred
                }
                if paths is not None:
                    try:
                        item['path'] = str(paths[i])
                    except Exception:
                        item['path'] = None

                results.append(item)

    return results
