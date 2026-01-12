"""Test EVAN multi-modality implementation."""

import torch
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from evan_main import EVAN

# Set up logging to see the verbose output
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def test_single_rgb_tensor():
    """Test backward compatibility: single RGB tensor input."""
    print("\n" + "="*80)
    print("Test 1: Single RGB tensor (backward compatibility)")
    print("="*80)

    model = EVAN(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        tz_fusion_time=3,
        device='cpu'
    )

    # Single tensor input (should be converted to {'rgb': tensor} internally)
    x = torch.randn(2, 3, 224, 224)

    output = model.forward_features(x)

    print(f"Input shape: {x.shape}")
    print(f"Output modalities: {list(output.keys())}")
    print(f"Output CLS token shape (rgb): {output['rgb']['x_norm_clstoken'].shape}")
    print(f"Output patch tokens shape (rgb): {output['rgb']['x_norm_patchtokens'].shape}")
    print("✓ Test passed!")


def test_dict_single_modality():
    """Test dict input with single RGB modality."""
    print("\n" + "="*80)
    print("Test 2: Dict input with single RGB modality")
    print("="*80)

    model = EVAN(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        tz_fusion_time=3,
        device='cpu'
    )

    # Dict input with single modality
    x = {'rgb': torch.randn(2, 3, 224, 224)}

    output = model.forward_features(x)

    print(f"Input modalities: {list(x.keys())}")
    print(f"Input shape (rgb): {x['rgb'].shape}")
    print(f"Output modalities: {list(output.keys())}")
    print(f"Output CLS token shape (rgb): {output['rgb']['x_norm_clstoken'].shape}")
    print(f"Output patch tokens shape (rgb): {output['rgb']['x_norm_patchtokens'].shape}")
    print("✓ Test passed!")


def test_new_modality_creation():
    """Test dynamic creation of new modality components."""
    print("\n" + "="*80)
    print("Test 3: Dynamic new modality creation (infrared with 4 channels)")
    print("="*80)

    model = EVAN(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        tz_fusion_time=3,
        device='cpu'
    )

    # First pass: RGB only
    x1 = {'rgb': torch.randn(2, 3, 224, 224)}
    output1 = model.forward_features(x1)
    print(f"\nFirst batch - Input modalities: {list(x1.keys())}")

    # Second pass: RGB + new infrared modality
    x2 = {
        'rgb': torch.randn(2, 3, 224, 224),
        'infrared': torch.randn(2, 4, 224, 224)  # 4 channels
    }

    print(f"\nSecond batch - Input modalities: {list(x2.keys())}")
    print(f"Input shape (rgb): {x2['rgb'].shape}")
    print(f"Input shape (infrared): {x2['infrared'].shape}")

    output2 = model.forward_features(x2)

    print(f"\nOutput modalities: {list(output2.keys())}")
    print(f"Output CLS token shape (rgb): {output2['rgb']['x_norm_clstoken'].shape}")
    print(f"Output CLS token shape (infrared): {output2['infrared']['x_norm_clstoken'].shape}")
    print(f"Output patch tokens shape (rgb): {output2['rgb']['x_norm_patchtokens'].shape}")
    print(f"Output patch tokens shape (infrared): {output2['infrared']['x_norm_patchtokens'].shape}")

    # Expected: 196 patches per modality (each modality kept separate)
    expected_patches = (224 // 16) ** 2
    actual_patches_rgb = output2['rgb']['x_norm_patchtokens'].shape[1]
    actual_patches_infrared = output2['infrared']['x_norm_patchtokens'].shape[1]
    print(f"Expected patch count per modality: {expected_patches}")
    print(f"Actual patches (rgb): {actual_patches_rgb}, (infrared): {actual_patches_infrared}")
    assert actual_patches_rgb == expected_patches, f"RGB patch count mismatch!"
    assert actual_patches_infrared == expected_patches, f"Infrared patch count mismatch!"

    print("✓ Test passed!")


def test_multi_modality_fusion():
    """Test fusion of multiple modalities."""
    print("\n" + "="*80)
    print("Test 4: Multi-modality fusion (RGB + Infrared + Thermal)")
    print("="*80)

    model = EVAN(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        tz_fusion_time=3,
        device='cpu'
    )

    # Three modalities with different channel counts
    x = {
        'rgb': torch.randn(2, 3, 224, 224),
        'infrared': torch.randn(2, 4, 224, 224),
        'thermal': torch.randn(2, 1, 224, 224)
    }

    print(f"Input modalities: {list(x.keys())}")
    print(f"Channel counts: rgb={x['rgb'].shape[1]}, infrared={x['infrared'].shape[1]}, thermal={x['thermal'].shape[1]}")

    output = model.forward_features(x)

    print(f"\nOutput modalities: {list(output.keys())}")
    for modality in output.keys():
        print(f"  {modality}: CLS shape = {output[modality]['x_norm_clstoken'].shape}, "
              f"patches shape = {output[modality]['x_norm_patchtokens'].shape}")

    # Expected: 196 patches per modality (each modality kept separate)
    expected_patches = (224 // 16) ** 2
    for modality in output.keys():
        actual_patches = output[modality]['x_norm_patchtokens'].shape[1]
        assert actual_patches == expected_patches, f"{modality} patch count mismatch! Expected {expected_patches}, got {actual_patches}"
        print(f"  ✓ {modality}: {actual_patches} patches (correct)")

    print("✓ Test passed!")


def test_parameter_counts():
    """Test that parameters are correctly counted and frozen."""
    print("\n" + "="*80)
    print("Test 5: Parameter counting and freezing")
    print("="*80)

    model = EVAN(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        tz_fusion_time=3,
        device='cpu'
    )

    # Count initial parameters
    initial_params = sum(p.numel() for p in model.parameters())
    initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Initial total parameters: {initial_params:,}")
    print(f"Initial trainable parameters: {initial_trainable:,}")

    # Add a new modality
    x = {'infrared': torch.randn(2, 4, 224, 224)}
    _ = model.forward_features(x)

    # Count parameters after adding new modality
    final_params = sum(p.numel() for p in model.parameters())
    final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nAfter adding 'infrared' modality:")
    print(f"Total parameters: {final_params:,}")
    print(f"Trainable parameters: {final_trainable:,}")
    print(f"New parameters added: {final_params - initial_params:,}")

    # Check that shared blocks are frozen
    for i, block in enumerate(model.blocks):
        for name, param in block.named_parameters():
            assert not param.requires_grad, f"Block {i} param {name} should be frozen!"

    print("✓ All shared blocks are frozen!")
    print("✓ Test passed!")


if __name__ == '__main__':
    test_single_rgb_tensor()
    test_dict_single_modality()
    test_new_modality_creation()
    test_multi_modality_fusion()
    test_parameter_counts()

    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
