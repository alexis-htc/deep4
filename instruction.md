For this project, you will step into the role of a Generative AI Specialist at SuperCognition Labs, a cybersecurity company focused on advanced bot detection and human verification systems.

SuperCognition’s current CAPTCHA system has a critical weakness: it can be defeated by bots that rely on simple geometric fonts. To address this vulnerability, the company needs a large-scale, highly diverse dataset of synthetic, human-like handwriting—a style that is significantly harder for rule-based or low-complexity algorithms to replicate.
Your Challenge

To solve this problem, you will:

    Design and implement two generative models:
        A Conditional Generative Adversarial Network (cGAN)
        A Conditional Diffusion Model
    Use both models to generate a massive and highly varied dataset of synthetic handwriting.
    Compare the models based on:
        Output quality
        Diversity
        Conditioning performance
        Practical suitability for large-scale data generation

The dataset you produce will serve as training data for a next-generation CAPTCHA classifier that must remain robust against increasingly sophisticated bots.
Your Objective

By the end of this project, you will have:

    Built and trained two state-of-the-art conditional generative models
    Evaluated their strengths and weaknesses in a real-world cybersecurity context
    Produced a scalable pipeline for generating synthetic handwriting data

In short: you’re not just generating samples—you’re strengthening a system designed to distinguish humans from bots.

Project Instructions

This project explores conditional generative models to ceate synthetic handwritten digits for CAPTCHA-style human verification

    A Conditional GAN (cGAN)
    A Conditional Diffusion Model

Both models are trained on MNIST and evaluated using:

    Frechet Inception Distance (FID) - measures realism & diversity
    Downstream classifier accuracy - how useful the synthetic data is for training a classifier

This project is structured with starter code with ToDos in both python modules and notebooks.
Project Structure

project /
|-- data /
    |-- dataloader.py
|-- model /
    |-- __init__.py
    |-- cgan.py
    |-- diffusion.py
|-- training / 
    | -- __init__.py
    | -- train_cgan.py
    | -- train_diffusion.py
|-- utils /
   |-- __init__.py
   |-- checkpoint.py
   |-- metrics.py
   |-- visualize.py
|-- 00_data_preparation.ipynb
|-- 01_cGAN_training.ipynb
|-- 02_diffusion_training.ipynb
|-- 03_evaluation.ipynb
|-- README.md

Data /

    dataloader.py
        Loads the MNIST Dataset
        Applies standard transform
        Helper functions to load data in notebooks

Model /

    cgan.py Defines:

        Generator(G) - takes noise + class label and outputs a 28 x 28 image.

        Discriminator(D) - takes image + class label and outputs real/fake probability
        Student ToDos:
            Implement the label - conditioning logic in the G and D forward passes.
            Make sure image tensors are reshaped correctly on output/input

    diffusion.py Defines:

        timestep_embedding - converts diffusion step t into an embedding vector

        ResidualBlock - convolutional block with time + label conditioning and residual connection

        ConditionalUNet - encoder-decoder UNet that predicts noise for a given noisy image, timestep, and label
        Student ToDos:
            Implement the UNet forward pass: time embedding -> down path -> bottleneck -> up path -> output noise prediction

Training /

    train_cgan.py Implements the adversarial training loop for the cGAN
        Student ToDos:
            Complete the Generator update step (Sample noise/labels, generate images, compute generator loss, update weights)
            Complete the Discriminator update step (compute real vs fake loss, average, backprop)
            Add Checkpoint saving

    train_diffusion.py Defines:

        linear_beta_schedule - builds a simple noise schedule

        sample_images - runs the reverse diffusion process to sample from noise

        train_diffusion - trains the diffusion model via noise prediction
        Student ToDos:
            In train_diffusion: Implement forward diffusion (add noise) and the MSE noise-prediction training step
            In sample_images: Implement the reverse diffusion update loop to gradually denoise images

Utils /

checkpoint.py: Helper functions to save and load model + optimizer states metrics.py: Helper functions to calculate evaluation metrics visualize.py: Helper functions for plotting batches of Images and comparison grids
Notebooks Overview

The noteboks are the main entry point for running the project. They use the modules above and contain additional notebook-level ToDos to wire piece together and interpret the results

    00_data_preparation.ipynb - Downloads and loads MNIST data, visualizes sample digits
    01_cGAN_training.ipynb - Imports Generator, Discriminator and train_cGAN, sets up models, training loops for specified epocs and visualizes generated digits per class
    02_diffusion_training.ipynb - Imports ConditionalUNet, train_diffusion, and sample_images, trains the difusion model and samples class-conditioned digits
    03_evaluation.ipynb - Loads Real MNIST test set, synthetic samples from cGAN and Diffision, runs visual comparison (real vs cGAN vs Diffusion per class), FID computation between real vs synthetic, a simple CNN classifier trained on synthetic data and evaluated on real data

    Use this project rubric to understand and assess the project criteria.
Generative Model Architectures
Criteria	Submission Requirements

Implement class-conditional logic in GAN architectures
	

    model/cgan.py contains a Generator that successfully concatenates the latent vector z with a class embedding.
    model/cgan.py contains a `Discriminator` that integrates the class label with the image input tensor.
    Generator output uses a Tanh activation to produce 1x28x28 tensors in the [-1, 1] range.

Construct a Conditional U-Net for noise prediction
	

    model/diffusion.py implements a ConditionalUNet that utilizes both a time-step embedding and a class label embedding.
    The architecture includes skip connections between the contracting (down) and expansive (up) paths.
    The model correctly outputs a noise prediction tensor of the same shape as the input noisy image.

Model Training & Sampling
Criteria	Submission Requirements

Implement adversarial training loops for cGANs
	

    01_cGAN_training.ipynb implements the standard GAN minimax objective (BCE loss).
    The training loop includes a separate step for updating the Discriminator (on real and fake data) and the Generator.
    `01_cGAN_training.ipynb` displays a grid of generated digits (0-9) that clearly represent their respective class labels.

Implement forward and reverse diffusion processes
	

    02_diffusion_training.ipynb implements the forward noising process ($q$ sampling) based on a linear or cosine beta schedule.
    The `sample_images` function implements the reverse iterative denoising loop from $T$ to $0$.
    The training objective correctly calculates the MSE loss between predicted and actual added noise.