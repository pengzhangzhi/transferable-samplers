"""
Invertibility Diagnostics for TarFlowSingleSystem

Diagnoses why z → x → z_recon fails by monitoring scale parameters per metablock.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.neural_networks.tarflow.tarflow_single_system import TarFlow

# Numerical precision settings (same as eval.py)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_model(ckpt_path: str) -> TarFlow:
    """Load TarFlowSingleSystem from checkpoint."""
    # Model config from fastflow_Ace-A-Nme.yaml
    model = TarFlow(
        in_channels=1,
        img_size=66,
        patch_size=1,
        channels=256,
        num_blocks=4,
        layers_per_block=4,
        nvp=True,
    )
    
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats
    if any(k.startswith("net.model.") for k in state_dict.keys()):
        # EMA checkpoint format
        print("Detected EMA checkpoint, extracting model weights...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if "shadow_params" in k or "num_updates" in k:
                continue
            if k.startswith("net.model."):
                new_key = k[len("net.model."):]
                new_state_dict[new_key] = v
            elif k.startswith("net."):
                new_key = k[len("net."):]
                new_state_dict[new_key] = v
        state_dict = new_state_dict
    elif any(k.startswith("net.") for k in state_dict.keys()):
        # Standard Lightning checkpoint
        print("Detected Lightning checkpoint, stripping 'net.' prefix...")
        state_dict = {k[len("net."):]: v for k, v in state_dict.items() if k.startswith("net.")}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_xa_from_block(block, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run forward through a single MetaBlock and extract xa (log-scale param).
    Returns: (output_z, xa, logdet)
    """
    with torch.no_grad():
        # Replicate MetaBlock.forward() but capture xa
        x_perm = block.permutation(x)
        pos_embed = block.permutation(block.pos_embed, dim=0)
        x_in = x_perm
        h = block.proj_in(x_perm) + pos_embed
        
        if block.class_embed is not None:
            h = h + block.class_embed.mean(dim=0)
        
        for attn_block in block.attn_blocks:
            h = attn_block(h, block.attn_mask)
        
        h = block.proj_out(h)
        h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        
        if block.nvp:
            xa, xb = h.chunk(2, dim=-1)
        else:
            xb = h
            xa = torch.zeros_like(h)
        
        scale = (-xa.float()).exp().type(xa.dtype)
        x_out = block.permutation((x_in - xb) * scale, inverse=True)
        logdet = -xa.mean(dim=[1, 2])
        
        return x_out, xa, logdet


def diagnose(model: TarFlow, batch_size: int = 64, device: str = "cuda"):
    """Run full invertibility diagnostics."""
    model = model.to(device)
    
    # Sample z from standard normal (prior)
    torch.manual_seed(42)
    z = torch.randn(batch_size, 66, 1, device=device)
    
    print("\n" + "=" * 76)
    print("                     INVERTIBILITY DIAGNOSTICS")
    print("=" * 76)
    
    # === Global stats ===
    print(f"\n{'='*76}")
    print("GLOBAL STATISTICS")
    print(f"{'='*76}")
    
    z_l2 = z.pow(2).sum(dim=[1, 2]).sqrt().mean().item()
    print(f"Input z:   L2={z_l2:.4f}  min={z.min().item():.4f}  max={z.max().item():.4f}")
    
    # Forward pass z → x
    with torch.no_grad():
        x, fwd_logdet = model(z)
    
    x_l2 = x.pow(2).sum(dim=[1, 2]).sqrt().mean().item()
    print(f"Output x:  L2={x_l2:.4f}  min={x.min().item():.4f}  max={x.max().item():.4f}")
    
    # Reverse pass x → z_recon
    with torch.no_grad():
        z_recon, rev_logdet = model.reverse(x, return_logdets=True)
    
    diff = (z - z_recon).abs()
    z_recon_l2 = z_recon.pow(2).sum(dim=[1, 2]).sqrt().mean().item()
    print(f"Recon z:   L2={z_recon_l2:.4f}  max_diff={diff.max().item():.6f}  mse={diff.pow(2).mean().item():.6e}")
    print(f"Logdet:    fwd={fwd_logdet.mean().item():.4f}  rev={rev_logdet.mean().item():.4f}  sum={( fwd_logdet + rev_logdet).mean().item():.6f}")
    
    # === Per-block xa statistics ===
    print(f"\n{'='*76}")
    print("PER-BLOCK SCALE PARAMETER (xa) STATISTICS")
    print(f"{'='*76}")
    print(f"{'Block':^6} │ {'xa_min':^10} │ {'xa_max':^10} │ {'xa_std':^10} │ {'scale_min':^12} │ {'scale_max':^12} │ {'|xa|>5':^8} │ {'|xa|>10':^8}")
    print("-" * 76)
    
    z_curr = z.clone()
    block_xa_stats = []
    
    for i, block in enumerate(model.blocks):
        z_next, xa, _ = extract_xa_from_block(block, z_curr)
        
        xa_min = xa.min().item()
        xa_max = xa.max().item()
        xa_std = xa.std().item()
        xa_mean = xa.mean().item()
        
        scale = (-xa.float()).exp()
        scale_min = scale.min().item()
        scale_max = scale.max().item()
        
        extreme_5 = (xa.abs() > 5).sum().item()
        extreme_10 = (xa.abs() > 10).sum().item()
        
        z_curr_l2 = z_next.pow(2).sum(dim=[1, 2]).sqrt().mean().item()
        
        block_xa_stats.append({
            'xa_min': xa_min, 'xa_max': xa_max, 'xa_std': xa_std, 'xa_mean': xa_mean,
            'scale_min': scale_min, 'scale_max': scale_max,
            'extreme_5': extreme_5, 'extreme_10': extreme_10,
            'z_l2': z_curr_l2
        })
        
        print(f"{i:^6} │ {xa_min:^10.4f} │ {xa_max:^10.4f} │ {xa_std:^10.4f} │ {scale_min:^12.6f} │ {scale_max:^12.4f} │ {extreme_5:^8} │ {extreme_10:^8}")
        
        z_curr = z_next
    
    # === Per-block round-trip test ===
    print(f"\n{'='*76}")
    print("PER-BLOCK ROUND-TRIP ERROR (block.forward(block.reverse(x)))")
    print(f"{'='*76}")
    print(f"{'Block':^6} │ {'roundtrip_mse':^14} │ {'roundtrip_max':^14} │ {'z_l2_after':^12} │ {'status':^12}")
    print("-" * 76)
    
    # Start from x, go backwards through blocks
    x_curr = x.clone()
    
    for i, block in enumerate(reversed(model.blocks)):
        block_idx = len(model.blocks) - 1 - i
        
        with torch.no_grad():
            # Reverse this block
            z_rev = block.reverse(x_curr, return_logdets=False)
            # Forward again
            x_roundtrip, _ = block(z_rev)
        
        error = (x_curr - x_roundtrip).abs()
        mse = error.pow(2).mean().item()
        max_err = error.max().item()
        z_l2 = z_rev.pow(2).sum(dim=[1, 2]).sqrt().mean().item()
        
        status = "OK" if max_err < 1e-4 else "⚠ WARN" if max_err < 1e-2 else "❌ BAD"
        
        print(f"{block_idx:^6} │ {mse:^14.6e} │ {max_err:^14.6e} │ {z_l2:^12.4f} │ {status:^12}")
        
        x_curr = z_rev
    
    # === Detailed error analysis ===
    print(f"\n{'='*76}")
    print("ERROR BREAKDOWN BY POSITION")
    print(f"{'='*76}")
    
    diff = (z - z_recon).abs()
    
    # Per-token error
    token_error = diff.mean(dim=[0, 2])  # Average over batch and channels
    worst_tokens = torch.topk(token_error, min(5, len(token_error)))
    print(f"Worst 5 tokens (by mean abs error):")
    for idx, err in zip(worst_tokens.indices.tolist(), worst_tokens.values.tolist()):
        print(f"  Token {idx}: error={err:.6f}")
    
    # Per-sample error
    sample_error = diff.mean(dim=[1, 2])  # Average over tokens and channels
    print(f"\nPer-sample error stats: min={sample_error.min().item():.6f}  max={sample_error.max().item():.6f}  std={sample_error.std().item():.6f}")
    
    # === Summary ===
    print(f"\n{'='*76}")
    print("SUMMARY")
    print(f"{'='*76}")
    
    total_extreme_5 = sum(s['extreme_5'] for s in block_xa_stats)
    total_extreme_10 = sum(s['extreme_10'] for s in block_xa_stats)
    total_elements = batch_size * 66 * 1 * len(model.blocks)  # batch * tokens * channels * blocks
    
    print(f"Total scale params with |xa| > 5:  {total_extreme_5} / {total_elements} ({100*total_extreme_5/total_elements:.2f}%)")
    print(f"Total scale params with |xa| > 10: {total_extreme_10} / {total_elements} ({100*total_extreme_10/total_elements:.2f}%)")
    
    max_scale = max(s['scale_max'] for s in block_xa_stats)
    min_scale = min(s['scale_min'] for s in block_xa_stats)
    print(f"Scale range across all blocks: [{min_scale:.6e}, {max_scale:.4f}]")
    
    if max_scale > 100 or min_scale < 1e-4:
        print("\n⚠️  WARNING: Extreme scale values detected! This likely causes invertibility issues.")
        print("   Consider adding scale clamping: scale = scale.clamp(min=0.01, max=100)")
    
    if diff.max().item() > 0.01:
        print(f"\n❌ INVERTIBILITY FAILURE: max reconstruction error = {diff.max().item():.6f}")
    else:
        print(f"\n✓ Invertibility OK: max reconstruction error = {diff.max().item():.6e}")
    
    print("=" * 76 + "\n")


if __name__ == "__main__":
    ckpt_path = "scratch/transferable-samplers/logs/train/runs/fastflow_Ace-A-Nme-dist_loss_kl/checkpoints/last_weights_only.ckpt"
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Please provide the correct path.")
        sys.exit(1)
    
    print(f"Loading model from: {ckpt_path}")
    model = load_model(ckpt_path)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    diagnose(model, batch_size=64, device=device)

