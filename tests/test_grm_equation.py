import numpy as np

from grm_tcm_dynamic_grm import DynamicGRMConfig, spectral_grm
from grm_tcm_train import GRMTCMTrainer, GRMTrainConfig


def test_spectral_grm_matches_entrywise_green_kernel():
    W = np.array(
        [
            [0.0, 1.0, 0.3, 0.0],
            [1.0, 0.0, 0.7, 0.2],
            [0.3, 0.7, 0.0, 1.1],
            [0.0, 0.2, 1.1, 0.0],
        ],
        dtype=float,
    )
    r_s = 1.7
    cfg = DynamicGRMConfig(max_modes=3, energy_threshold=1.0)

    G, lambdas, psi, selected, _ = spectral_grm(W, r_s, cfg)

    weights = 1.0 / (1.0 + (r_s**2) * lambdas)
    expected = np.zeros_like(G)
    for m in range(selected):
        expected += weights[m] * np.outer(psi[:, m], psi[:, m])

    squared_weight_kernel = (psi * weights.reshape(1, -1)) @ (psi * weights.reshape(1, -1)).T

    np.testing.assert_allclose(G, expected, atol=1e-12)
    assert not np.allclose(G, squared_weight_kernel)


def test_static_embedding_inner_product_matches_grm_kernel():
    eigenvalues = np.array([0.2, 0.8, 1.4], dtype=float)
    eigenvectors = np.array(
        [
            [0.5, -0.2, 0.4],
            [0.1, 0.7, -0.3],
            [-0.6, 0.2, 0.1],
            [0.2, -0.5, -0.8],
        ],
        dtype=float,
    )
    rho = 1.3
    trainer = GRMTCMTrainer(GRMTrainConfig(rho=rho))

    embeddings = trainer._make_grm_embeddings(eigenvalues, eigenvectors)

    grm_weights = 1.0 / (1.0 + (rho**2) * eigenvalues)
    expected_kernel = (eigenvectors * grm_weights.reshape(1, -1)) @ eigenvectors.T
    old_coords = eigenvectors * grm_weights.reshape(1, -1)
    old_squared_kernel = old_coords @ old_coords.T

    np.testing.assert_allclose(embeddings @ embeddings.T, expected_kernel, atol=1e-12)
    assert not np.allclose(embeddings @ embeddings.T, old_squared_kernel)
