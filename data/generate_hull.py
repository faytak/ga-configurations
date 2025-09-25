import os
import numpy as np
from tqdm import trange

data_root = "datasets"

if __name__ == "__main__":
    dataroot = os.path.join(data_root, "hulls")

    n = 6 # O(6)-Convex Hull
    num_train_samples = 16384
    num_val_samples = 16384
    num_test_samples = 16384

    if not os.path.exists(dataroot):
        os.makedirs(dataroot)

    from scipy.spatial import ConvexHull

    def _generate_meshes(n_points, n):
        meshes = []
        for i in trange(n_points):
            points = np.random.randn(16, n)
            hull = ConvexHull(points)
            volume = hull.volume
            meshes.append((points, volume))

        meshes, volumes = zip(*meshes)
        meshes = np.stack(meshes).astype(np.float32)
        volumes = np.stack(volumes).astype(np.float32)
        return meshes, volumes

    train_meshes, train_volumes = _generate_meshes(num_train_samples, n)
    val_meshes, val_volumes = _generate_meshes(num_val_samples, n)
    test_meshes, test_volumes = _generate_meshes(num_test_samples, n)
    np.save(os.path.join(dataroot, "hulls_train_input.npy"), train_meshes)
    np.save(os.path.join(dataroot, "hulls_train_target.npy"), train_volumes)
    np.save(os.path.join(dataroot, "hulls_val_input.npy"), val_meshes)
    np.save(os.path.join(dataroot, "hulls_val_target.npy"), val_volumes)
    np.save(os.path.join(dataroot, "hulls_test_input.npy"), test_meshes)
    np.save(os.path.join(dataroot, "hulls_test_target.npy"), test_volumes)