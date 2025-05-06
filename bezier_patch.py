import torch
import torchvision
import argparse


# Setup CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Bernstein basis for cubic Bézier ---
def bernstein_basis_3(t):
    return torch.stack([
        (1 - t)**3,
        3 * t * (1 - t)**2,
        3 * t**2 * (1 - t),
        t**3
    ], dim=-1)  # shape: (..., 4)

# --- 4x4 control point grid with noise ---
def make_control_grid(grid_height=4, grid_width=4, deviation=0.1):
    base_height = torch.linspace(0.25, 0.75, grid_height, device=device)  # (4,)
    base_width = torch.linspace(0.25, 0.75, grid_width, device=device)  # (4,)
    x, y = torch.meshgrid(base_width, base_height, indexing="ij")
    control_points = torch.stack([x, y], dim=-1)  # (4, 4, 2)
    control_points += torch.randn_like(control_points) * deviation
    return control_points  # (4, 4, 2)

def hsv_to_rgb(h, s, v):
    """Convert HSV (N, 3) tensor to RGB (N, 3)"""
    h = h * 6  # scale hue to [0, 6]
    i = torch.floor(h).long()
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    i = i % 6

    conditions = [
        (i == 0, torch.stack((v, t, p), dim=1)),
        (i == 1, torch.stack((q, v, p), dim=1)),
        (i == 2, torch.stack((p, v, t), dim=1)),
        (i == 3, torch.stack((p, q, v), dim=1)),
        (i == 4, torch.stack((t, p, v), dim=1)),
        (i == 5, torch.stack((v, p, q), dim=1)),
    ]

    rgb = torch.zeros_like(h).unsqueeze(-1).repeat(1, 3)
    for cond, val in conditions:
        rgb[cond] = val[cond]
    return rgb


# --- Construct 4x4 color control grid from corner colors only ---
def make_corner_color_grid():
    # Step 1: Random hue values for 4 corners
    hue = torch.rand(4, 1, device=device)  # values in [0, 1)

    # Step 2: Fixed saturation and value
    saturation = torch.ones_like(hue) * 0.9  # or try 1.0
    value = torch.ones_like(hue) * 1.0       # or slightly less if you want less brightness

    hsv = torch.cat([hue, saturation, value], dim=1)  # (4, 3)
    C = hsv_to_rgb(hsv[:, 0], hsv[:, 1], hsv[:, 2])    # (4, 3) in RGB

    s = torch.tensor([0.0, 1/3, 2/3, 1.0], device=device)
    Bu = bernstein_basis_3(s)  # (4, 4)

    # Build color grid using only corner colors and Bezier weights
    control_colors = torch.zeros(4, 4, 3, device=device)
    for i in range(4):
        for j in range(4):
            control_colors[i, j] = (
                C[0] * Bu[i, 0] * Bu[j, 0] +  # top-left
                C[1] * Bu[i, 0] * Bu[j, 3] +  # top-right
                C[2] * Bu[i, 3] * Bu[j, 0] +  # bottom-left
                C[3] * Bu[i, 3] * Bu[j, 3]    # bottom-right
            )
    return control_colors  # (4, 4, 3)

# --- Evaluate Bézier patch over (H x W) grid ---
def evaluate_patch(control_points, control_colors, H=128, W=128):
    u = torch.linspace(0, 1, H, device=device)
    v = torch.linspace(0, 1, W, device=device)
    Bu = bernstein_basis_3(u)  # (H, 4)
    Bv = bernstein_basis_3(v)  # (W, 4)

    patch_coords = torch.einsum('hi,wj,ijc->hwc', Bu, Bv, control_points)  # (H, W, 2)
    patch_colors = torch.einsum('hi,wj,ijc->hwc', Bu, Bv, control_colors)  # (H, W, 3)
    return patch_coords, patch_colors

# --- Rasterize patch onto image canvas ---
def rasterize_patch(patch_coords, patch_colors, H=128, W=128):
    img = torch.zeros(3, H, W, device=device)

    coords = patch_coords.round().long().clamp(0, H - 1)
    y = coords[..., 1].view(-1)
    x = coords[..., 0].view(-1)
    colors = patch_colors.view(-1, 3)

    img[:, y, x] = colors.T  # Set RGB per pixel
    return img

def render_bezier_patch_image(H=256, W=256, height_patches=2, width_patches=2):
    control_pts = make_control_grid(grid_height=height_patches*3+1, grid_width=width_patches*3+1, deviation=0.05)

    image = torch.zeros(3, H, W, device=device)

    for i in range(height_patches):  # row (0, 1)
        for j in range(width_patches):  # col (0, 1)
            subgrid = control_pts[j*3:j*3+4, i*3:i*3+4]  # 4x4, with shared points
            control_colors = make_corner_color_grid()

            patch_coords, patch_colors = evaluate_patch(subgrid, control_colors, H, W)

            # Offset into image quadrant
            patch_coords[..., 0] = patch_coords[..., 0] * W
            patch_coords[..., 1] = patch_coords[..., 1] * H

            patch_img = rasterize_patch(patch_coords, patch_colors, H, W)
            image = torch.maximum(image, patch_img)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Bézier patch image")
    parser.add_argument("--height", type=int, default=2, help="Number of Bézier patches vertically")
    parser.add_argument("--width", type=int, default=2, help="Number of Bézier patches horizontally")

    args = parser.parse_args()

    image = render_bezier_patch_image(
        H=args.height*128,
        W=args.width*128,
        height_patches=args.height,
        width_patches=args.height
    )

    torchvision.utils.save_image(image, f"bezier_patch_{args.height}x{args.width}.png")
    print("Image saved to bezier_patch.png")

