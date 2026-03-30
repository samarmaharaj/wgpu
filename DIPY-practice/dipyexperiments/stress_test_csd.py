import numpy as np
from dipy.data import read_stanford_labels
from dipy.core.gradients import gradient_table
# CORRECTED IMPORT HERE:
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

print("Loading Stanford dataset...")
# Using the dataset from the commented-out benchmark
img, gtab, labels_img = read_stanford_labels()
data = img.get_fdata()

# Select a chunk of the brain to process (12x12x12 = 1728 voxels)
center = (50, 40, 40)
width = 12
a, b, c = center
hw = width // 2
idx = (slice(a - hw, a + hw), slice(b - hw, b + hw), slice(c - hw, c + hw))

data_small = data[idx].copy()
mask_small = np.ones(data_small.shape[:-1], dtype=bool) 

print("Setting up CSD Model...")
model = ConstrainedSphericalDeconvModel(gtab, response=(np.array([0.0015, 0.0003, 0.0003]), 100), sh_order_max=8)

print("Starting heavy CPU math (CSD fit)...")
# We will run the fit 5 times to force the CPU to do massive work
for i in range(5):
    print(f"  Running iteration {i+1}/5...")
    # The fixed line
    csd_fit = model.fit(data_small, mask=mask_small)

print("Done!")