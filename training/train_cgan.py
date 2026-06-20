"""
train_cgan.py
-------------
Reusable training function for cGAN experiments.
"""

import torch


def train_cgan(
    generator,
    discriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    criterion,
    device,
    z_dim,
    num_classes,
    epochs=50,
):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)

            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, z_dim).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs, labels), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(
            f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            from utils.checkpoint import save_checkpoint
            save_checkpoint(generator, optimizer_G, epoch + 1, g_loss.item(), name="cgan_generator")
            save_checkpoint(discriminator, optimizer_D, epoch + 1, d_loss.item(), name="cgan_discriminator")
            print(f"Saved checkpoints at epoch {epoch+1}")

    # Save final checkpoints
    from utils.checkpoint import save_checkpoint
    save_checkpoint(generator, optimizer_G, epoch="final", loss=g_loss.item(), name="cgan_generator")
    save_checkpoint(discriminator, optimizer_D, epoch="final", loss=d_loss.item(), name="cgan_discriminator")
