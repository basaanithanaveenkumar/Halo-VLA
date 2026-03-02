Halo-VLA: Building a Vision-Language-Action Model from Scratch

Naveen | February 2026


I'm excited to share Halo-VLA — a Vision-Language-Action model I built entirely from scratch in PyTorch.

Instead of fine-tuning a massive pretrained LLM, I designed every component from the ground up: the vision encoder, decoder transformer, Mixture-of-Experts layers, state encoder, and action decoder. The result? A fully transparent, hackable, and lightweight architecture at ~393M parameters.

Let me walk you through how it works. 👇


🔍 The Problem

Recent models like RT-2, OpenVLA, and π₀ have shown that Vision-Language-Action models — language models that can also output robot actions — are a promising path toward generalist robot policies.

But most existing VLAs are built by fine-tuning massive pretrained LLMs (7B+ parameters), making them expensive to train and hard to modify.

I wanted to take a different approach.


🏗️ Architecture at a Glance

Halo-VLA processes three modalities — images, language, and proprioceptive robot states — through a unified token sequence and produces two outputs:

✅ Next-token logits for language understanding
✅ Continuous action predictions for robot control

The flow:

  RGB Images → ViT Encoder → Image Projector ─┐
  Text Tokens → Token Embedding ───────────────┤→ Decoder Transformer (12 layers, MoE)
  Robot States → State Encoder ────────────────┘         |
                                                  ┌──────┴──────┐
                                               LM Head    Action Decoder
                                              (language)  (robot actions)


🧩 Key Components

1. Vision Encoder (ViT)

Each image is split into 16×16 patches and projected into the embedding space. With 224×224 images, each frame becomes 196 patch tokens. For 5 frames of a robot trajectory, that's 980 visual tokens fed into the model.


2. State Encoder

Raw proprioceptive vectors (joint positions, velocities, gripper state) are mapped into the transformer's embedding space through a lightweight MLP with LayerNorm and GELU activations. State embeddings are injected in-place at <state> token positions — preserving temporal alignment with surrounding tokens.


3. DeepSeek-Style Mixture-of-Experts (MoE)

This is one of the design choices I'm most excited about.

Each of the 12 decoder blocks uses a DeepSeek-style MoE instead of a standard FFN:

→ Shared experts (2) always run — they capture universal patterns like basic language understanding
→ Routed experts (8, top-2 selected) specialize in modality-specific or task-specific computation
→ Noisy top-k routing adds Gaussian noise during training for better load balancing

Each expert uses a SwiGLU-style gated linear unit. The result: more model capacity without proportional compute increase.


4. Action Chunking

Instead of predicting one action at a time, each <halo_action> token produces 16 future action steps in a single forward pass. This dramatically improves temporal consistency and reduces compounding errors.

The Action Decoder is an MLP that maps a single hidden state to a flat vector of size chunk_size × action_dim, then reshapes it into a trajectory.


5. Dual-Head Output

After the transformer:

→ A Language Head (LayerNorm + linear) produces vocabulary logits for next-token prediction
→ An Action Decoder (MLP) extracts hidden states at <halo_action> positions and predicts continuous action trajectories


📐 The Unified Sequence Design

The key design insight:

1️⃣ Image patches are prepended — placed at the start so the transformer always has full visual context
2️⃣ State embeddings replace <state> placeholders — I inject them at their natural positions in the conversation, maintaining temporal coherence
3️⃣ <halo_action> tokens remain as markers — their hidden states are extracted after the transformer for action decoding


🎯 Training

I train Halo-VLA with a dual-objective loss:

𝓛 = 𝓛_language + λ · 𝓛_action

→ Language loss: Cross-entropy on next-token prediction (padding and image patches masked out)
→ Action loss: L1 loss between predicted and ground-truth action trajectories

Optimizer: AdamW with cosine annealing and gradient clipping at 1.0.

Dataset: I train on the interleave-temporal subset of EO-Data1.5M (huggingface.co/datasets/IPEC-COMMUNITY/EO-Data1.5M), which provides multi-turn robot conversations with interleaved images, proprioceptive states, and action trajectories. My dataloader splits multi-turn conversations into individual user-assistant pairs and builds a cumulative-sum index for efficient random access.


🧠 Design Decisions & Why They Matter

→ Prepend images, inject states in-place: Images provide global context (always attend). States are temporally local (preserve alignment).
→ DeepSeek MoE with shared experts: More capacity without proportional compute. Shared handles common patterns; routed experts specialize.
→ Action chunking (16 steps): Reduces compounding errors and temporal jitter.
→ Continuous action head (not tokenized): Avoids discretization artifacts in fine-grained motor control.
→ Qwen2.5 tokenizer: 151K pretrained vocab + 3 custom tokens (<image>, <halo_action>, <state>).
→ ~393M parameters: Trainable on a single GPU. Fast iteration for research.


🚀 Try It Yourself

Clone the repo: github.com/SHAILAB-IPEC/Halo-VLA

Install: pip install -e ".[dev]"
Train: python scripts/train.py --max_samples 100 --epochs 5 --batch_size 2
Visualize: python scripts/visualize.py --checkpoint checkpoints/halo_vla_epoch5.pt
Inference: python scripts/inference.py --checkpoint checkpoints/halo_vla_epoch5.pt

All code is open source under the MIT License.


🔮 What's Next

→ Scaling experiments with larger embeddings and more experts
→ Real robot deployment with sim-to-real transfer
→ Multi-task training — VQA, captioning, and action prediction in one model
→ Attention-based action decoder replacing the MLP with cross-attention


If you're interested in building VLAs from scratch, or want to understand how these systems work at every level — check out the repo:

🔗 github.com/SHAILAB-IPEC/Halo-VLA

I'd love to hear your thoughts. Drop a comment or reach out!

#MachineLearning #Robotics #VLA #DeepLearning #PyTorch #OpenSource #AI #ComputerVision #NLP #TransformerModels
