#!/usr/bin/env python3
"""Complete all TODO cells in the project notebooks."""
import json
import os

def update_notebook(path, cell_updates):
    """Update specific cells in a notebook."""
    with open(path) as f:
        nb = json.load(f)
    
    for idx, new_source in cell_updates.items():
        if idx < len(nb['cells']):
            nb['cells'][idx]['source'] = new_source
    
    # Ensure all code cells have required fields
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if 'outputs' not in cell:
                cell['outputs'] = []
            if 'execution_count' not in cell:
                cell['execution_count'] = None
    
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {path}")


# === 01_cGAN_training.ipynb ===
cgan_updates = {
    # Cell 7: Extract data
    7: [
        "# Load MNIST data\n",
        "train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)\n",
        "print(f\"Training batches: {len(train_loader)} | Test batches: {len(test_loader)}\")\n"
    ],
    # Cell 10: Initialize models and optimizers
    10: [
        "# Initialize Generator and Discriminator\n",
        "generator = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=img_shape).to(device)\n",
        "discriminator = Discriminator(num_classes=num_classes, img_shape=img_shape).to(device)\n",
        "\n",
        "# Initialize optimizers\n",
        "optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))\n",
        "optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))\n",
        "\n",
        "# Loss function (Binary Cross Entropy)\n",
        "criterion = nn.BCELoss()\n",
        "\n",
        "print(f\"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}\")\n",
        "print(f\"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}\")\n"
    ],
    # Cell 12: Training loop
    12: [
        "from training.train_cgan import train_cgan\n",
        "\n",
        "# Train the cGAN\n",
        "train_cgan(\n",
        "    generator=generator,\n",
        "    discriminator=discriminator,\n",
        "    dataloader=train_loader,\n",
        "    optimizer_G=optimizer_G,\n",
        "    optimizer_D=optimizer_D,\n",
        "    criterion=criterion,\n",
        "    device=device,\n",
        "    z_dim=z_dim,\n",
        "    num_classes=num_classes,\n",
        "    epochs=epochs,\n",
        ")\n"
    ],
    # Cell 13: Save final model
    13: [
        "# Save final Generator and Discriminator\n",
        "save_checkpoint(generator, optimizer_G, epoch='final', name='cgan_generator')\n",
        "save_checkpoint(discriminator, optimizer_D, epoch='final', name='cgan_discriminator')\n",
        "print('Final cGAN models saved!')\n"
    ],
    # Cell 15: Fix show_batch call (was truncated)
    15: [
        "### Generate and visualize samples\n",
        "def generate_samples(generator, n_classes=10, n_per_class=8):\n",
        "    generator.eval()\n",
        "    with torch.no_grad():\n",
        "        z = torch.randn(n_classes * n_per_class, z_dim).to(device)\n",
        "        labels = torch.tensor([i for i in range(n_classes) for _ in range(n_per_class)]).to(device)\n",
        "        gen_imgs = generator(z, labels)\n",
        "    generator.train()\n",
        "    return gen_imgs, labels\n",
        "\n",
        "gen_imgs, gen_labels = generate_samples(generator)\n",
        "show_batch(gen_imgs.cpu(), gen_labels.cpu(), n=16)\n"
    ]
}
update_notebook('01_cGAN_training.ipynb', cgan_updates)


# === 02_diffusion_training.ipynb ===
diffusion_updates = {
    # Cell 6: Extract data
    6: [
        "# Load MNIST data\n",
        "train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)\n",
        "print(f\"Training batches: {len(train_loader)} | Test batches: {len(test_loader)}\")\n"
    ],
    # Cell 9: Instantiate model and train
    9: [
        "# Instantiate Conditional UNet model\n",
        "model = ConditionalUNet(num_classes=num_classes).to(device)\n",
        "print(f\"Model parameters: {sum(p.numel() for p in model.parameters()):,}\")\n",
        "\n",
        "# Train the diffusion model\n",
        "train_diffusion(\n",
        "    model=model,\n",
        "    dataloader=train_loader,\n",
        "    device=device,\n",
        "    num_classes=num_classes,\n",
        "    timesteps=timesteps,\n",
        "    epochs=epochs,\n",
        "    lr=lr,\n",
        ")\n"
    ],
    # Cell 10: Just a note cell, make it a proper comment
    10: [
        "# Notice: the loss slowly decreases. Unlike GANs, diffusion training\n",
        "# is much more stable but typically slower to converge.\n",
        "print('Diffusion training complete!')\n"
    ],
}
update_notebook('02_diffusion_training.ipynb', diffusion_updates)


# === 03_evaluation.ipynb ===
# First read the full notebook to understand all cells
with open('03_evaluation.ipynb') as f:
    eval_nb = json.load(f)
for i, c in enumerate(eval_nb['cells']):
    src = ''.join(c.get('source', []))[:100]
    print(f"  Cell {i} [{c['cell_type']}]: {src}")

eval_updates = {
    # Cell 6: get_real_examples_per_class - need to see if it's complete
    # Cell 9: Initialize models
    9: [
        "# Initialize models\n",
        "os.makedirs('../checkpoints', exist_ok=True)\n",
        "\n",
        "# Initialize cGAN Generator\n",
        "z_dim = 100\n",
        "num_classes = 10\n",
        "img_shape = (1, 28, 28)\n",
        "cgan_gen = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=img_shape).to(device)\n",
        "\n",
        "# Initialize Conditional UNet\n",
        "diff_model = ConditionalUNet(num_classes=num_classes).to(device)\n",
        "\n",
        "# Load latest checkpoints\n",
        "load_checkpoint(cgan_gen, path='../checkpoints/cgan_generator_epochfinal.pt', map_location=device)\n",
        "load_checkpoint(diff_model, path='../checkpoints/diffusion_unet_epochfinal.pt', map_location=device)\n",
        "\n",
        "cgan_gen.eval()\n",
        "diff_model.eval()\n"
    ],
    # Cell 10: Generate samples
    10: [
        "def generate_conditioned_samples_cgan(generator, device, z_dim=100, n_classes=10, n_per_class=100):\n",
        "    \"\"\"Generate class-conditioned samples from cGAN.\"\"\"\n",
        "    generator.eval()\n",
        "    all_imgs, all_labels = [], []\n",
        "    with torch.no_grad():\n",
        "        for cls in range(n_classes):\n",
        "            z = torch.randn(n_per_class, z_dim).to(device)\n",
        "            labels = torch.full((n_per_class,), cls, dtype=torch.long, device=device)\n",
        "            imgs = generator(z, labels)\n",
        "            all_imgs.append(imgs)\n",
        "            all_labels.append(labels)\n",
        "    return torch.cat(all_imgs), torch.cat(all_labels)\n",
        "\n",
        "def generate_conditioned_samples_diffusion(model, device, n_classes=10, n_per_class=100, timesteps=200):\n",
        "    \"\"\"Generate class-conditioned samples from diffusion model.\"\"\"\n",
        "    model.eval()\n",
        "    all_imgs, all_labels = [], []\n",
        "    for cls in range(n_classes):\n",
        "        class_labels = torch.full((n_per_class,), cls, dtype=torch.long, device=device)\n",
        "        imgs, labels = sample_images(model, device, num_samples=n_per_class,\n",
        "                                     num_classes=n_classes, timesteps=timesteps,\n",
        "                                     class_labels=class_labels)\n",
        "        all_imgs.append(imgs)\n",
        "        all_labels.append(labels)\n",
        "    return torch.cat(all_imgs), torch.cat(all_labels)\n",
        "\n",
        "real_imgs, real_labels = get_real_examples_per_class()\n",
        "\n",
        "# Generate cGAN samples\n",
        "cgan_imgs, cgan_labels = generate_conditioned_samples_cgan(cgan_gen, device=device)\n",
        "print(f'cGAN samples: {cgan_imgs.shape}')\n",
        "\n",
        "# Generate diffusion samples\n",
        "diff_imgs, diff_labels = generate_conditioned_samples_diffusion(diff_model, device=device)\n",
        "print(f'Diffusion samples: {diff_imgs.shape}')\n"
    ],
    # Cell 11: Plot comparison
    11: [
        "def plot_real_cgan_diffusion(real_imgs, real_labels, cgan_imgs, cgan_labels, diff_imgs, diff_labels, n_classes=10):\n",
        "    \"\"\"Plot side-by-side comparison of real, cGAN, and diffusion samples.\"\"\"\n",
        "    fig, axes = plt.subplots(3, n_classes, figsize=(20, 6))\n",
        "    row_labels = ['Real', 'cGAN', 'Diffusion']\n",
        "    \n",
        "    for cls in range(n_classes):\n",
        "        # Real\n",
        "        r_mask = (real_labels == cls)\n",
        "        if r_mask.any():\n",
        "            axes[0, cls].imshow(real_imgs[r_mask][0].squeeze().cpu().numpy(), cmap='gray')\n",
        "        axes[0, cls].set_title(f'{cls}')\n",
        "        axes[0, cls].axis('off')\n",
        "        \n",
        "        # cGAN\n",
        "        c_mask = (cgan_labels == cls)\n",
        "        if c_mask.any():\n",
        "            axes[1, cls].imshow(cgan_imgs[c_mask][0].squeeze().cpu().detach().numpy(), cmap='gray')\n",
        "        axes[1, cls].axis('off')\n",
        "        \n",
        "        # Diffusion\n",
        "        d_mask = (diff_labels == cls)\n",
        "        if d_mask.any():\n",
        "            axes[2, cls].imshow(diff_imgs[d_mask][0].squeeze().cpu().detach().numpy(), cmap='gray')\n",
        "        axes[2, cls].axis('off')\n",
        "    \n",
        "    for i, label in enumerate(row_labels):\n",
        "        axes[i, 0].set_ylabel(label, fontsize=14, rotation=0, labelpad=60)\n",
        "    \n",
        "    plt.suptitle('Real vs cGAN vs Diffusion Samples', fontsize=16)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "plot_real_cgan_diffusion(real_imgs, real_labels, cgan_imgs, cgan_labels, diff_imgs, diff_labels)\n"
    ],
    # Cell 14: Choose subset for FID
    14: [
        "# Choose subset for FID computation\n",
        "n_fid = min(1000, len(cgan_imgs), len(diff_imgs))\n",
        "\n",
        "# Get real images for FID\n",
        "real_test = datasets.MNIST(root='../data', train=False, download=True,\n",
        "                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))\n",
        "real_loader = DataLoader(real_test, batch_size=n_fid, shuffle=True)\n",
        "real_subset, _ = next(iter(real_loader))\n",
        "real_subset = real_subset[:n_fid]\n",
        "\n",
        "cgan_subset = cgan_imgs[:n_fid].detach().cpu()\n",
        "diff_subset = diff_imgs[:n_fid].detach().cpu()\n",
        "\n",
        "print(f'FID evaluation on {n_fid} samples each')\n",
        "print(f'Real: {real_subset.shape}, cGAN: {cgan_subset.shape}, Diffusion: {diff_subset.shape}')\n"
    ],
    # Cell 15: Compute FID
    15: [
        "# Compute FID for cGAN\n",
        "cgan_fid = compute_fid(real_subset, cgan_subset, device=device)\n",
        "\n",
        "# Compute FID for Diffusion\n",
        "diff_fid = compute_fid(real_subset, diff_subset, device=device)\n",
        "\n",
        "print(f' cGAN FID: {cgan_fid:.2f}')\n",
        "print(f'Diffusion FID: {diff_fid:.2f}')\n"
    ],
    # Cell 16: FID analysis
    16: [
        "# FID Score Analysis\n",
        "print('FID Score Analysis')\n",
        "print('=' * 50)\n",
        "print(f'cGAN FID: {cgan_fid:.2f}')\n",
        "print(f'Diffusion FID: {diff_fid:.2f}')\n",
        "print()\n",
        "if diff_fid < cgan_fid:\n",
        "    print('The Diffusion model achieves a LOWER FID score, indicating better')\n",
        "    print('image quality and diversity compared to the cGAN.')\n",
        "    print('This is expected: diffusion models learn a more stable denoising process,')\n",
        "    print('while GANs can suffer from mode collapse and training instability.')\n",
        "else:\n",
        "    print('The cGAN achieves a LOWER FID score, indicating competitive or better')\n",
        "    print('image quality. This can happen with well-tuned GAN training and sufficient epochs.')\n",
        "print()\n",
        "print('Lower FID = better quality and diversity.')\n",
        "print('FID measures the distance between real and generated image distributions')\n",
        "print('using features from a pretrained Inception network.')\n"
    ],
}
update_notebook('03_evaluation.ipynb', eval_updates)

print("\nAll notebooks updated!")
