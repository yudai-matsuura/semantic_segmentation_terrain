import numpy as np

def load_pfm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header != 'Pf':
            raise ValueError('Not a valid PFM file (must be grayscale Pf)')

        dims = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dims.split())

        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        data = data.reshape((height, width))
        data = np.flipud(data)  # Flip vertically as PFM format stores from bottom row
        return data

# --- 使用例 ---
pfm_path = "/media/go2laptop/T7/Moon_9/image0/depth/1692601811687322368.pfm"
depth = load_pfm(pfm_path)

print(f"✅ 画像の shape: {depth.shape}")  # -> (H, W)
print(f"✅ dtype: {depth.dtype}")         # -> float32 (should be float32)
