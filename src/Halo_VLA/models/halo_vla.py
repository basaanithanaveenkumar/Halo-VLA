from config import HaloVLMConfig
from models.vit import VisTransformer
from models.transformer import DecoderTransformer
from models.lm_head import LMHead
from models.image_proj import ImageProjector
from models.state_encoder import StateEncoder
from models.action_decoder import ActionDecoder
import torch
import torch.nn as nn

from models.positional_embeddings import SinusoidalPositionalEmbedding


class HaloVLM(nn.Module):
    def __init__(self, config: HaloVLMConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = HaloVLMConfig(**kwargs)
        self.config = config
        network_config = config

        self.vis_enc = VisTransformer(
            img_size=network_config.img_size,
            p_size=network_config.patch_size,
            in_chans=network_config.in_chans,
            emb_dim=network_config.emb_dim,
            num_layers=network_config.vit_num_layers,
            num_heads=network_config.vit_num_heads,
            mlp_dim=network_config.vit_mlp_dim,
            drop_fact=network_config.vit_drop,
        )
        self.decoder_transformer = DecoderTransformer(
            num_layers=network_config.dec_num_layers,
            emb_dim=network_config.emb_dim,
            num_heads=network_config.dec_num_heads,
            mlp_dim=network_config.dec_mlp_dim,
            drop_fact=network_config.dec_drop,
            use_moe=network_config.use_moe,
            moe_hid_scale=network_config.moe_hid_scale,
            moe_num_routed_experts=network_config.moe_num_routed_experts,
            moe_top_k=network_config.moe_top_k,
            moe_num_shared_experts=network_config.moe_num_shared_experts,
        )
        self.token_emb = nn.Embedding(network_config.vocab_size, network_config.emb_dim)
        self.pos_embed = nn.Embedding(network_config.max_position_embeddings, network_config.emb_dim)
        self.layer_norm = nn.LayerNorm(network_config.emb_dim)
        self.lm_head = LMHead(hidden_size=network_config.emb_dim, vocab_size=network_config.vocab_size)
        self.image_projector = ImageProjector(
            vision_dim=network_config.proj_vision_dim or network_config.emb_dim,
            llm_dim=network_config.proj_llm_dim or network_config.emb_dim,
        )
        self.state_encoder = StateEncoder(config=network_config)
        self.action_decoder = ActionDecoder(config=network_config)
    

    def forward(self, images, input_ids, attention_mask, states):
        """
        Forward pass with interleaved multi-image, state-injection, and
        action-decoding support.

        The input_ids sequence contains three kinds of special tokens:
          • <image>  (image_token_id)  — replaced by ViT patch embeddings
          • <state>  (state_token_id)  — replaced by StateEncoder output
          • <action> (action_token_id) — used *after* the transformer to
            extract hidden states for ActionDecoder

        Args:
            images:         [B, N_img, 3, H, W]  — N_img images per sample.
            input_ids:      [B, seq_len]          — token ids with <image>,
                            <state>, and <action> placeholders.
            attention_mask: [B, seq_len]          — 1 for real tokens, 0 pad.
            states:         [B, N_state, state_dim] — proprioceptive states,
                            one per <state> token.

        Returns:
            logits:         [B, total_len, vocab_size] — next-token logits.
            action_preds:   [B, n_action_tokens, chunk_size, action_dim]
                            — decoded actions at every <action> position.
                            None if no <action> tokens are present.
        """

        B = input_ids.size(0)
        device = input_ids.device
        network_config = self.config
        image_token_id = network_config.image_token_id    # e.g. 151665
        state_token_id = network_config.state_token_id    # e.g. 151667
        action_token_id = network_config.action_token_id  # e.g. 151666

        # ------------------------------------------------------------------ #
        # 1. IMAGES — count <image> tokens, encode each image through ViT,
        #    project into LLM space, and concatenate all patch features.
        #    img_proj → [B, N_img * num_patches, emb_dim]
        # ------------------------------------------------------------------ #
        num_images = (input_ids == image_token_id).sum(dim=1)  # [B]
        N_img = num_images.max().item()                        # scalar

        all_img_proj = []
        for i in range(N_img):
            img_i = images[:, i]                               # [B, 3, H, W]
            feat_i = self.vis_enc(img_i)                       # [B, num_patches, emb_dim]
            proj_i = self.image_projector(feat_i)              # [B, num_patches, emb_dim]
            all_img_proj.append(proj_i)

        img_proj = torch.cat(all_img_proj, dim=1)              # [B, N_img*num_patches, emb_dim]

        # ------------------------------------------------------------------ #
        # 2. STATES — count <state> tokens, encode each state vector through
        #    the StateEncoder MLP to get a single embedding per state.
        #    state_embeds → [B, N_state, emb_dim]
        # ------------------------------------------------------------------ #
        num_states = (input_ids == state_token_id).sum(dim=1)  # [B]
        N_state = num_states.max().item()                      # scalar

        all_state_embeds = []
        for i in range(N_state):
            state_i = states[:, i]                             # [B, state_dim]
            emb_i = self.state_encoder(state_i)                # [B, emb_dim]
            all_state_embeds.append(emb_i.unsqueeze(1))        # [B, 1, emb_dim]

        if all_state_embeds:
            state_embeds = torch.cat(all_state_embeds, dim=1)  # [B, N_state, emb_dim]
        else:
            state_embeds = torch.empty(B, 0, network_config.emb_dim, device=device)

        # ------------------------------------------------------------------ #
        # 3. TEXT EMBEDDINGS — embed all tokens, then zero out positions that
        #    correspond to <image> or <state> placeholders (their actual
        #    information comes from the encoders above).
        # ------------------------------------------------------------------ #
        special_mask = (input_ids != image_token_id) & (input_ids != state_token_id)
        text_embeds = self.token_emb(input_ids)                # [B, seq_len, emb_dim]
        text_embeds = text_embeds * special_mask.unsqueeze(-1)  # zero out placeholders

        # ------------------------------------------------------------------ #
        # 4. INJECT STATE EMBEDDINGS — replace the zeroed-out <state> token
        #    positions in text_embeds with the encoded state embeddings.
        #    This keeps states at their original position in the sequence
        #    (rather than prepending like images).
        # ------------------------------------------------------------------ #
        for b in range(B):
            state_positions = (input_ids[b] == state_token_id).nonzero(as_tuple=True)[0]
            for idx, pos in enumerate(state_positions):
                if idx < N_state:
                    text_embeds[b, pos] = state_embeds[b, idx]

        # ------------------------------------------------------------------ #
        # 5. CONCATENATE — prepend image patch features before the text+state
        #    sequence so the decoder can attend to visual context first.
        #    combined_embeds → [B, N_img*num_patches + seq_len, emb_dim]
        # ------------------------------------------------------------------ #
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)

        # ------------------------------------------------------------------ #
        # 6. POSITIONAL EMBEDDINGS — add learned positions over the full
        #    combined sequence.
        # ------------------------------------------------------------------ #
        total_len = combined_embeds.size(1)
        pos_emb = self.pos_embed(torch.arange(total_len, device=device)).unsqueeze(0)
        combined_embeds = combined_embeds + pos_emb

        # ------------------------------------------------------------------ #
        # 7. DECODER TRANSFORMER — causal self-attention + MoE FFN.
        # ------------------------------------------------------------------ #
        transformer_out = self.decoder_transformer(combined_embeds)
        transformer_out = self.layer_norm(transformer_out)

        # ------------------------------------------------------------------ #
        # 8. LANGUAGE HEAD — project every position to vocab logits.
        #    logits → [B, total_len, vocab_size]
        # ------------------------------------------------------------------ #
        logits = self.lm_head(transformer_out)

        # ------------------------------------------------------------------ #
        # 9. ACTION DECODING — find <action> token positions in the
        #    *original* input_ids (offset by the prepended image patches).
        #    Extract the transformer hidden states at those positions and
        #    pass them through the ActionDecoder MLP to predict actions.
        #
        #    action_preds → [B, n_action_tokens, chunk_size, action_dim]
        #    Returns None if no <action> tokens are present.
        # ------------------------------------------------------------------ #
        num_prepended = img_proj.size(1)  # N_img * num_patches
        action_mask = (input_ids == action_token_id)           # [B, seq_len]
        n_action_tokens = action_mask.sum(dim=1).max().item()  # scalar

        action_preds = None
        if n_action_tokens > 0:
            action_hidden_list = []
            for b in range(B):
                action_positions = action_mask[b].nonzero(as_tuple=True)[0]  # indices in input_ids
                # Offset by prepended image patches to index into transformer_out
                action_positions = action_positions + num_prepended
                hiddens = transformer_out[b, action_positions]              # [n_act, emb_dim]
                # Pad if this sample has fewer <action> tokens than the batch max
                if hiddens.size(0) < n_action_tokens:
                    pad = torch.zeros(
                        n_action_tokens - hiddens.size(0), network_config.emb_dim,
                        device=device,
                    )
                    hiddens = torch.cat([hiddens, pad], dim=0)
                action_hidden_list.append(hiddens)

            action_hiddens = torch.stack(action_hidden_list, dim=0)  # [B, n_act, emb_dim]
            # Decode each <action> position independently
            B_a, T_a, D_a = action_hiddens.shape
            action_preds = self.action_decoder(
                action_hiddens.view(B_a * T_a, D_a)
            )  # [B*n_act, chunk_size, action_dim]
            action_preds = action_preds.view(
                B, T_a, self.action_decoder.chunk_size, self.action_decoder.action_dim
            )  # [B, n_act, chunk_size, action_dim]

        return logits, action_preds