import torch
from torchvision.utils import make_grid, save_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import random

def evaluate(model, dataset, device, filename, num_samples=8):
    model.eval()

    n_total = len(dataset)
    n = min(num_samples, n_total)

    selected_indices = random.sample(range(n_total), n)
    samples = [dataset[i] for i in selected_indices]

    noisy = torch.stack([s['noisy'] for s in samples])
    clean = torch.stack([s['clean'] for s in samples])
    paths = [s['path'] for s in samples]

    print("Images used for evaluation:")
    for p in paths:
        print(" -", p)

    lpips_model = lpips.LPIPS(net='alex').to(device)

    with torch.no_grad():
        noisy_device = noisy.to(device)
        output = model(noisy_device)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output_cpu = output.cpu()

    psnr_list = []
    ssim_list = []
    lpips_list = []

    for i in range(n):
        denoised_np = output_cpu[i].permute(1, 2, 0).numpy()
        clean_np = clean[i].permute(1, 2, 0).numpy()

        psnr_list.append(compare_psnr(clean_np, denoised_np, data_range=1.0))

        ssim_val = compare_ssim(clean_np, denoised_np, data_range=1.0, channel_axis=2)
        ssim_list.append(ssim_val)

        lpips_val = lpips_model(
            output_cpu[i].unsqueeze(0) * 2 - 1,
            clean[i].unsqueeze(0) * 2 - 1
        ).item()
        lpips_list.append(lpips_val)

    print("\nMetrics (average over {} samples):".format(n))
    print(" - PSNR: {:.2f}".format(np.mean(psnr_list)))
    print(" - SSIM: {:.4f}".format(np.mean(ssim_list)))
    print(" - LPIPS: {:.4f}".format(np.mean(lpips_list)))

    grid = make_grid(
        torch.cat((noisy, output_cpu, clean), dim=0),
        nrow=n
    )
    save_image(grid, filename)


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     checkpoint = torch.load("best_denoise.pth", map_location=device)

#     denoiser = UNetDenoise()
#     denoiser.to(device)
#     denoiser.load_state_dict(checkpoint['model'])


#     val_dataset = DenoiseDataset('./DIV2K_valid_HR', 256, (0.01, 0.03), augment=False)

#     evaluate(denoiser, val_dataset, device, 'denoised.png', num_samples=8)