"""
Region Visualizer Node - Visualize regional attention masks from coupled model
✅ IMPROVED: Real text labels with PIL
"""

import torch
import torch.nn.functional as F
from typing import Any, Tuple, List, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .couple_utils import (
    log_debug, log_warning, detect_model_architecture, 
    ArchitectureInfo, ModelType, MASK_EPSILON
)
from .couple_flux_patching import RegionalMask


class ComfyCoupleRegionVisualizer:
    """
    Visualize regional attention masks from a coupled model
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coupled_model": ("MODEL", {
                    "tooltip": "Model with regional attention patches applied"
                }),
                "visualization_mode": ([
                    "individual_regions",
                    "overlay_all", 
                    "grid_layout",
                    "attention_heatmap"
                ], {
                    "default": "overlay_all",
                    "tooltip": (
                        "individual_regions: Each region as separate image | "
                        "overlay_all: All regions combined with colors | "
                        "grid_layout: Grid of all regions | "
                        "attention_heatmap: Attention strength heatmap"
                    )
                }),
                "output_resolution": (["latent", "512", "1024", "2048"], {
                    "default": "1024",
                    "tooltip": "Output image resolution (latent=mask native resolution)"
                }),
            },
            "optional": {
                "color_scheme": ([
                    "rainbow", 
                    "pastel", 
                    "neon", 
                    "monochrome"
                ], {
                    "default": "rainbow",
                    "tooltip": "Color palette for region visualization"
                }),
                "alpha": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Transparency for overlay mode (0=transparent, 1=opaque)"
                }),
                "show_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show region index numbers (Region 0, Region 1, etc.) on visualization"
                }),
                "label_size": (["small", "medium", "large"], {
                    "default": "medium",
                    "tooltip": "Size of region index labels"
                }),
                "model_type": (["auto", "sd15", "sdxl", "flux"], {
                    "default": "auto",
                    "tooltip": "Force specific architecture detection"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("visualization", "region_info")
    FUNCTION = "visualize"
    CATEGORY = "ToyxyzTestNodes"

    # Color schemes (RGB normalized 0-1)
    COLOR_SCHEMES = {
        "rainbow": [
            (1.0, 0.0, 0.0),   # Red
            (1.0, 0.5, 0.0),   # Orange
            (1.0, 1.0, 0.0),   # Yellow
            (0.0, 1.0, 0.0),   # Green
            (0.0, 0.5, 1.0),   # Cyan
            (0.0, 0.0, 1.0),   # Blue
            (0.5, 0.0, 1.0),   # Purple
            (1.0, 0.0, 0.5),   # Magenta
        ],
        "pastel": [
            (1.0, 0.7, 0.7),   # Pastel red
            (1.0, 0.9, 0.6),   # Pastel orange
            (1.0, 1.0, 0.7),   # Pastel yellow
            (0.7, 1.0, 0.7),   # Pastel green
            (0.7, 0.9, 1.0),   # Pastel cyan
            (0.7, 0.7, 1.0),   # Pastel blue
            (0.9, 0.7, 1.0),   # Pastel purple
            (1.0, 0.7, 0.9),   # Pastel magenta
        ],
        "neon": [
            (1.0, 0.0, 0.3),   # Neon red
            (1.0, 0.3, 0.0),   # Neon orange
            (0.9, 1.0, 0.0),   # Neon yellow
            (0.0, 1.0, 0.3),   # Neon green
            (0.0, 1.0, 1.0),   # Neon cyan
            (0.3, 0.3, 1.0),   # Neon blue
            (0.8, 0.0, 1.0),   # Neon purple
            (1.0, 0.0, 0.8),   # Neon magenta
        ],
        "monochrome": [
            (0.2, 0.2, 0.2),   # Dark gray
            (0.4, 0.4, 0.4),   # Medium gray
            (0.6, 0.6, 0.6),   # Light gray
            (0.8, 0.8, 0.8),   # Lighter gray
            (1.0, 1.0, 1.0),   # White
            (0.3, 0.3, 0.3),   # Dark-medium gray
            (0.5, 0.5, 0.5),   # Medium-light gray
            (0.7, 0.7, 0.7),   # Light-lighter gray
        ],
    }

    def visualize(self, coupled_model: Any, visualization_mode: str, 
                  output_resolution: str, color_scheme: str = "rainbow",
                  alpha: float = 0.7, show_labels: bool = True,
                  label_size: str = "medium", model_type: str = "auto") -> Tuple[torch.Tensor, str]:
        """
        Visualize regional attention masks
        """
        # Detect architecture
        arch_info = self._detect_architecture(coupled_model, model_type)
        is_flux = arch_info.is_flux
        
        print(f"[RegionVisualizer] Architecture: {arch_info.type.value.upper()}")
        print(f"[RegionVisualizer] Mode: {visualization_mode}, Colors: {color_scheme}")
        
        # Extract region data
        if is_flux:
            region_masks, region_info = self._extract_flux_regions(coupled_model)
        else:
            region_masks, region_info = self._extract_sd_sdxl_regions(coupled_model)
        
        if not region_masks:
            blank = torch.zeros(1, 512, 512, 3)
            return (blank, "No regional masks found in model")
        
        print(f"[RegionVisualizer] Found {len(region_masks)} regions")
        
        # Get colors
        colors = self.COLOR_SCHEMES.get(color_scheme, self.COLOR_SCHEMES["rainbow"])
        
        # Create visualization
        if visualization_mode == "individual_regions":
            vis_image = self._visualize_individual(
                region_masks, output_resolution, colors, show_labels, label_size
            )
        elif visualization_mode == "overlay_all":
            vis_image = self._visualize_overlay(
                region_masks, output_resolution, colors, alpha, show_labels, label_size
            )
        elif visualization_mode == "grid_layout":
            vis_image = self._visualize_grid(
                region_masks, output_resolution, colors, show_labels, label_size
            )
        else:  # attention_heatmap
            vis_image = self._visualize_heatmap(
                region_masks, output_resolution, show_labels, label_size
            )
        
        return (vis_image, region_info)

    def _detect_architecture(self, model: Any, model_type_str: str) -> ArchitectureInfo:
        """Detect or force model architecture"""
        if model_type_str == "auto":
            return detect_model_architecture(model)
        elif model_type_str == "flux":
            return ArchitectureInfo(type=ModelType.FLUX, use_cfg=True, dim=4096, is_flux=True)
        elif model_type_str == "sdxl":
            return ArchitectureInfo(type=ModelType.SDXL, use_cfg=True, dim=2048, is_flux=False)
        else:  # sd15
            return ArchitectureInfo(type=ModelType.SD15, use_cfg=True, dim=768, is_flux=False)

    def _extract_flux_regions(self, coupled_model: Any) -> Tuple[List[torch.Tensor], str]:
        """Extract region masks from Flux model"""
        try:
            transformer_options = coupled_model.model_options.get("transformer_options", {})
            patches_replace = transformer_options.get("patches_replace", {})
            
            regional_mask_module = None
            
            for block_type in ["double", "single"]:
                if block_type in patches_replace:
                    for block_id, patch_content in patches_replace[block_type].items():
                        if isinstance(patch_content, dict):
                            for key, module in patch_content.items():
                                if key == "mask_fn" or (isinstance(key, tuple) and key[0] == "mask_fn"):
                                    regional_mask_module = module
                                    break
                        elif isinstance(patch_content, RegionalMask):
                            regional_mask_module = patch_content
                            break
                    
                    if regional_mask_module is not None:
                        break
            
            if regional_mask_module is None or not isinstance(regional_mask_module, RegionalMask):
                log_warning("No Flux regional masks found")
                return [], "No Flux regional masks found"
            
            num_regions = regional_mask_module.num_regions
            region_masks = []
            
            for i in range(num_regions):
                mask = getattr(regional_mask_module, f'region_mask_{i}')
                region_masks.append(mask.cpu().float())
            
            region_info = (f"Flux: {num_regions} regions, "
                          f"timesteps: {regional_mask_module.start_percent*100:.0f}%-"
                          f"{regional_mask_module.end_percent*100:.0f}%")
            
            log_debug(f"Extracted {num_regions} Flux regions")
            return region_masks, region_info
        
        except Exception as e:
            log_warning(f"Failed to extract Flux regions: {e}")
            return [], f"Error: {e}"

    def _extract_sd_sdxl_regions(self, coupled_model: Any) -> Tuple[List[torch.Tensor], str]:
        """Extract region masks from SD/SDXL model"""
        try:
            transformer_options = coupled_model.model_options.get("transformer_options", {})
            patches_replace = transformer_options.get("patches_replace", {})
            attn2_patches = patches_replace.get("attn2", {})
            
            if not attn2_patches:
                return [], "No SD/SDXL regional patches found"
            
            first_patch_key = list(attn2_patches.keys())[0]
            first_patch = attn2_patches[first_patch_key]
            
            region_masks = self._extract_masks_from_closure(first_patch)
            
            if not region_masks:
                return [], "Could not extract masks from attention patches"
            
            region_info = f"SD/SDXL: {len(region_masks)} regions"
            
            log_debug(f"Extracted {len(region_masks)} SD/SDXL regions")
            return region_masks, region_info
        
        except Exception as e:
            log_warning(f"Failed to extract SD/SDXL regions: {e}")
            return [], f"Error: {e}"

    def _extract_masks_from_closure(self, patch_function) -> List[torch.Tensor]:
        """Extract masks from attention patch closure"""
        try:
            closure = patch_function.__closure__
            if closure is None:
                return []
            
            closure_vars = {}
            for i, cell in enumerate(closure):
                try:
                    closure_vars[patch_function.__code__.co_freevars[i]] = cell.cell_contents
                except:
                    pass
            
            pos_masks = closure_vars.get('pos_masks', [])
            
            if not pos_masks:
                return []
            
            extracted_masks = []
            for mask in pos_masks:
                if isinstance(mask, torch.Tensor):
                    extracted_masks.append(mask.cpu().float())
            
            return extracted_masks
        
        except Exception as e:
            log_warning(f"Failed to extract from closure: {e}")
            return []

    def _resize_mask(self, mask: torch.Tensor, target_size: int) -> torch.Tensor:
        """Resize mask to target resolution"""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        resized = F.interpolate(
            mask,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        )
        
        return resized.squeeze(0).squeeze(0)

    def _get_target_size(self, masks: List[torch.Tensor], resolution: str) -> int:
        """Get target resolution for visualization"""
        if resolution == "latent":
            return max(masks[0].shape[-2:])
        else:
            return int(resolution)

    def _visualize_individual(self, masks: List[torch.Tensor], 
                             resolution: str, colors: List[Tuple],
                             show_labels: bool, label_size: str) -> torch.Tensor:
        """Visualize each region as separate image (batch)"""
        target_size = self._get_target_size(masks, resolution)
        
        batch_images = []
        
        for idx, mask in enumerate(masks):
            resized_mask = self._resize_mask(mask, target_size)
            
            # Create colored image
            color = colors[idx % len(colors)]
            img = torch.zeros(target_size, target_size, 3)
            
            for c in range(3):
                img[:, :, c] = resized_mask * color[c]
            
            # Add label if requested
            if show_labels:
                label_text = f"Region {idx}"
                if idx == 0:
                    label_text += " (BG)"  # Background indicator
                img = self._add_text_label(img, label_text, color, label_size)
            
            batch_images.append(img)
        
        return torch.stack(batch_images, dim=0)

    def _visualize_overlay(self, masks: List[torch.Tensor],
                          resolution: str, colors: List[Tuple],
                          alpha: float, show_labels: bool, label_size: str) -> torch.Tensor:
        """Overlay all regions with transparency"""
        target_size = self._get_target_size(masks, resolution)
        
        img = torch.zeros(target_size, target_size, 3)
        
        # Store label positions to avoid overlap
        label_positions = []
        
        for idx, mask in enumerate(masks):
            resized_mask = self._resize_mask(mask, target_size)
            color = colors[idx % len(colors)]
            
            # Apply alpha blending
            for c in range(3):
                img[:, :, c] = img[:, :, c] * (1 - resized_mask * alpha) + \
                               resized_mask * color[c] * alpha
            
            # Store label position (centroid of mask)
            if show_labels and resized_mask.sum() > MASK_EPSILON:
                label_positions.append((idx, resized_mask, color))
        
        # Add labels after rendering all regions
        if show_labels:
            img = self._add_multiple_labels(img, label_positions, label_size)
        
        return img.unsqueeze(0)

    def _visualize_grid(self, masks: List[torch.Tensor],
                       resolution: str, colors: List[Tuple],
                       show_labels: bool, label_size: str) -> torch.Tensor:
        """Create grid layout of all regions"""
        target_size = self._get_target_size(masks, resolution)
        
        num_regions = len(masks)
        grid_size = int(np.ceil(np.sqrt(num_regions)))
        
        cell_size = target_size // grid_size
        
        img = torch.zeros(target_size, target_size, 3)
        
        for idx, mask in enumerate(masks):
            row = idx // grid_size
            col = idx % grid_size
            
            if row >= grid_size:
                break
            
            resized_mask = self._resize_mask(mask, cell_size)
            color = colors[idx % len(colors)]
            
            y_start = row * cell_size
            x_start = col * cell_size
            y_end = min(y_start + cell_size, target_size)
            x_end = min(x_start + cell_size, target_size)
            
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            
            for c in range(3):
                img[y_start:y_end, x_start:x_end, c] = \
                    resized_mask[:actual_h, :actual_w] * color[c]
            
            # Add label in cell
            if show_labels:
                cell_img = img[y_start:y_end, x_start:x_end, :].clone()
                label_text = f"{idx}"
                if idx == 0:
                    label_text = "0(BG)"
                cell_img = self._add_text_label(cell_img, label_text, color, label_size)
                img[y_start:y_end, x_start:x_end, :] = cell_img
        
        return img.unsqueeze(0)

    def _visualize_heatmap(self, masks: List[torch.Tensor],
                          resolution: str, show_labels: bool, label_size: str) -> torch.Tensor:
        """Create attention strength heatmap"""
        target_size = self._get_target_size(masks, resolution)
        
        combined = torch.zeros(target_size, target_size)
        
        for mask in masks:
            resized_mask = self._resize_mask(mask, target_size)
            combined += resized_mask
        
        if combined.max() > 0:
            combined = combined / combined.max()
        
        img = self._apply_jet_colormap(combined)
        
        return img.unsqueeze(0)

    def _apply_jet_colormap(self, intensity: torch.Tensor) -> torch.Tensor:
        """Apply jet colormap to intensity values"""
        h, w = intensity.shape
        img = torch.zeros(h, w, 3)
        
        # Red channel
        img[:, :, 0] = torch.clamp((intensity - 0.5) * 4, 0, 1)
        
        # Green channel
        img[:, :, 1] = torch.where(
            intensity < 0.5,
            intensity * 2,
            2 - intensity * 2
        )
        
        # Blue channel
        img[:, :, 2] = torch.clamp((0.5 - intensity) * 4, 0, 1)
        
        return img

    def _add_text_label(self, img: torch.Tensor, text: str, 
                       color: Tuple[float, float, float], size: str) -> torch.Tensor:
        """Add text label to image using PIL - WHITE TEXT with improved sizing"""
        try:
            # Convert torch tensor to PIL Image
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)
            
            # Determine font size (more conservative sizing)
            h, w = img.shape[:2]
            if size == "small":
                font_size = max(16, h // 35)
            elif size == "large":
                font_size = max(32, h // 20)
            else:  # medium
                font_size = max(24, h // 28)
            
            # Try to load a bold font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Calculate text position (top-left with padding)
            padding = max(8, h // 80)
            
            # ✅ WHITE TEXT with black outline for visibility
            text_color = (255, 255, 255)  # White
            outline_color = (0, 0, 0)     # Black
            outline_width = max(2, font_size // 12)
            
            # Draw thick black outline
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((padding + dx, padding + dy), text, font=font, fill=outline_color)
            
            # Draw white text on top
            draw.text((padding, padding), text, font=font, fill=text_color)
            
            # Convert back to torch tensor
            img_with_label = torch.from_numpy(np.array(pil_img)).float() / 255.0
            
            return img_with_label
        
        except Exception as e:
            log_warning(f"Failed to add text label: {e}")
            # Fallback: add colored corner marker
            h, w, _ = img.shape
            corner_size = min(h, w) // 20
            for c in range(3):
                img[:corner_size, :corner_size, c] = color[c] * 0.8
            return img

    def _add_multiple_labels(self, img: torch.Tensor, 
                            label_positions: List[Tuple[int, torch.Tensor, Tuple]],
                            size: str) -> torch.Tensor:
        """Add multiple labels at mask centroids - WHITE TEXT"""
        try:
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)
            
            h, w = img.shape[:2]
            if size == "small":
                font_size = max(16, h // 35)
            elif size == "large":
                font_size = max(32, h // 20)
            else:
                font_size = max(24, h // 28)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # ✅ WHITE TEXT with black outline
            text_color = (255, 255, 255)  # White
            outline_color = (0, 0, 0)     # Black
            outline_width = max(2, font_size // 12)
            
            for idx, mask, color in label_positions:
                # Calculate centroid
                mask_np = mask.cpu().numpy()
                y_coords, x_coords = np.where(mask_np > 0.5)
                
                if len(y_coords) == 0:
                    continue
                
                centroid_y = int(np.mean(y_coords))
                centroid_x = int(np.mean(x_coords))
                
                # Draw label
                label_text = f"{idx}"
                if idx == 0:
                    label_text = "0(BG)"
                
                # Draw thick black outline
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((centroid_x + dx, centroid_y + dy), label_text, 
                                     font=font, fill=outline_color)
                
                # Draw white text on top
                draw.text((centroid_x, centroid_y), label_text, font=font, fill=text_color)
            
            img_with_labels = torch.from_numpy(np.array(pil_img)).float() / 255.0
            return img_with_labels
        
        except Exception as e:
            log_warning(f"Failed to add multiple labels: {e}")
            return img


# Node registration
NODE_CLASS_MAPPINGS = {
    "ComfyCoupleRegionVisualizer": ComfyCoupleRegionVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyCoupleRegionVisualizer": "Region Visualizer (Comfy Couple)",
}
