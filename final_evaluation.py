import torch
import matplotlib.pyplot as plt
import numpy as np
import lpips
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import random
from torchvision import transforms

from dataset import DIV2K
from net import PConvUNet


def tensor_to_uint8_image(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def inpaint_reference(masked_tensor, mask_tensor):
    masked_img = tensor_to_uint8_image(masked_tensor)

    mask_np = mask_tensor[0].detach().cpu().numpy()

    if mask_np.shape != masked_img.shape[:2]:
        mask_np = cv2.resize(mask_np, (masked_img.shape[1], masked_img.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    mask_uint8 = ((1 - mask_np) * 255).astype(np.uint8)

    result = cv2.inpaint(masked_img, mask_uint8, 3, cv2.INPAINT_TELEA)

    result = torch.from_numpy(result.astype(np.float32) / 255.0).permute(2, 0, 1)
    return result
def evaluate_inpainting(
        model,
        dataset,
        device,
        filename,
        num_samples=8,
        reference_implementation=inpaint_reference
    ):

    model.eval()

    n_total = len(dataset)
    n = min(num_samples, n_total)
    indices = random.sample(range(n_total), n)
    samples = [dataset[i] for i in range(n)]

    masked = torch.stack([s[0] for s in samples])
    masks = torch.stack([s[1] for s in samples])
    clean = torch.stack([s[2] for s in samples])

    lpips_model = lpips.LPIPS(net='alex').to(device)

    with torch.no_grad():
        masked_device = masked.to(device)
        masks_device = masks.to(device)
        output = model(masked_device, masks_device)
        if isinstance(output, (tuple, list)):
            output = output[0]

        print(output.shape)
        print(output.min().item(), output.max().item())
        print(output[1, :, 100, 100].to("cpu"))

        #output = torch.clamp(output, 0, 1)
        output_cpu = output.cpu()

        if reference_implementation is not None:
            ref_outputs = []
            for i in range(n):
                ref_out = reference_implementation(masked[i], masks[i])
                ref_outputs.append(ref_out)
            ref_output_cpu = torch.stack(ref_outputs)
        else:
            ref_output_cpu = output_cpu

    def compute_metrics(pred, gt):
        psnr_list, ssim_list, lpips_list = [], [], []
        for i in range(n):
            pred_np = np.clip(pred[i].permute(1, 2, 0).numpy(), 0, 1)
            gt_np = np.clip(gt[i].permute(1, 2, 0).numpy(), 0, 1)
            psnr_list.append(compare_psnr(gt_np, pred_np, data_range=1.0))
            ssim_list.append(compare_ssim(gt_np, pred_np, data_range=1.0, channel_axis=2))
            lpips_val = lpips_model(
                pred[i].unsqueeze(0) * 2 - 1,
                gt[i].unsqueeze(0) * 2 - 1
            ).item()
            lpips_list.append(lpips_val)
        return {
            "PSNR": np.mean(psnr_list),
            "SSIM": np.mean(ssim_list),
            "LPIPS": np.mean(lpips_list),
        }

    masked_metrics = compute_metrics(masked, clean)
    model_metrics = compute_metrics(output_cpu, clean)
    ref_metrics = compute_metrics(ref_output_cpu, clean)

    print("\n=== Evaluation Results over {} samples ===".format(n))
    print("Masked Input:")
    print(" - PSNR: {:.2f}".format(masked_metrics["PSNR"]))
    print(" - SSIM: {:.4f}".format(masked_metrics["SSIM"]))
    print(" - LPIPS: {:.4f}".format(masked_metrics["LPIPS"]))

    print("\nOur Model:")
    print(" - PSNR: {:.2f}".format(model_metrics["PSNR"]))
    print(" - SSIM: {:.4f}".format(model_metrics["SSIM"]))
    print(" - LPIPS: {:.4f}".format(model_metrics["LPIPS"]))

    print("\nReference (OpenCV Telea):")
    print(" - PSNR: {:.2f}".format(ref_metrics["PSNR"]))
    print(" - SSIM: {:.4f}".format(ref_metrics["SSIM"]))
    print(" - LPIPS: {:.4f}".format(ref_metrics["LPIPS"]))

    num_rows = 4
    row_labels = ['Masked', 'Our', 'Telea', 'Ground Truth']
    tensors_per_row = lambda c: [masked, output_cpu, ref_output_cpu, clean]

    fig, axes = plt.subplots(num_rows, n, figsize=(max(4, n * 3), num_rows * 3))
    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for col in range(n):
        imgs_for_col = tensors_per_row(col)
        for row, img_tensor in enumerate(imgs_for_col):
            img = img_tensor[col].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        axes[0, col].set_title(f"Sample {col+1}", fontsize=10)

    left_margin = 0.08
    plt.tight_layout(rect=[left_margin, 0, 1, 1])

    for row, label in enumerate(row_labels):
        y = 1.0 - (row + 0.5) / num_rows
        fig.text(
            left_margin / 2.0, y, label,
            va='center', ha='center',
            fontsize=14, fontweight='bold',
            rotation='vertical', rotation_mode='anchor'
        )

    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("4000.pth", map_location=device)

model = PConvUNet()
model.load_state_dict(checkpoint['model'])
model.to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
dataset = DIV2K("../DIV2K_valid_HR", transform, sizes_max_counts={8: 10, 32: 5})

evaluate_inpainting(
    model,
    dataset,
    device,
    filename="inpainting_eval_full.png",
    num_samples=100,
    reference_implementation=inpaint_reference
)
