import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class BinaryImageEvaluatorNp:
    def __init__(self, threshold=0.5):
        """
        NumPy version of binary image evaluator
        Args:
            threshold: binarization threshold (default 0.5, input should be in [0,1] range)
        """
        self.threshold = threshold

    def calculate_iou(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """NumPy version of IoU calculation"""
        intersection = np.logical_and(y_true, y_pred).sum(axis=(1, 2, 3))
        union = np.logical_or(y_true, y_pred).sum(axis=(1, 2, 3))
        return np.mean(intersection / (union + 1e-10))

    def calculate_dice(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """NumPy version of Dice coefficient"""
        intersection = np.logical_and(y_true, y_pred).sum(axis=(1, 2, 3))
        return np.mean(2 * intersection / (y_true.sum(axis=(1, 2, 3)) + y_pred.sum(axis=(1, 2, 3)) + 1e-10))

    def calculate_hd(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Hausdorff distance"""
        y_true = y_true.squeeze(1)  # [N,H,W]
        y_pred = y_pred.squeeze(1)
        hds = []

        for t, p in zip(y_true, y_pred):
            t_edges = np.argwhere(t > self.threshold)
            p_edges = np.argwhere(p > self.threshold)
            if len(t_edges) == 0 or len(p_edges) == 0:
                hds.append(np.inf)
            else:
                hd = max(
                    directed_hausdorff(t_edges, p_edges)[0],
                    directed_hausdorff(p_edges, t_edges)[0]
                )
                hds.append(hd)
        return float(np.mean(hds))

    def calculate_ssim(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """SSIM calculation (input should be in [0,1] range)"""
        ssim_values = []
        for t, p in zip(y_true, y_pred):
            ssim_val = ssim(t.squeeze(), p.squeeze(), data_range=1.0, channel_axis=None)
            ssim_values.append(ssim_val)
        return float(np.mean(ssim_values))

    def calculate_porosity(self, binary_image: np.ndarray) -> float:
        """
        Calculate porosity (pore area / total area)
        Args:
            binary_image: binary image [N,C,H,W]
        Returns:
            porosity (between 0-1)
        """
        pore_area = binary_image.sum(axis=(1, 2, 3))
        total_area = binary_image[0].size  # H*W
        return float(np.mean(pore_area / total_area))

    def calculate_pore_size_distribution(self, binary_image: np.ndarray, bins: int = 10) -> Dict:
        """
        Calculate pore diameter distribution
        Args:
            binary_image: binary image [N,C,H,W]
            bins: number of histogram bins
        Returns:
            Dictionary with diameter distribution statistics
        """
        binary_np = binary_image.squeeze(1)  # [N,H,W]
        all_diameters = []

        for img in binary_np:
            # Connected component analysis
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img.astype(np.uint8))
            if num_labels < 2:  # only background
                continue

            # Calculate equivalent diameter for each pore
            areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background
            diameters = 2 * np.sqrt(areas / np.pi)
            all_diameters.extend(diameters.tolist())

        # Calculate statistics
        if len(all_diameters) == 0:
            return {'mean': 0, 'std': 0, 'all_diameters': [0]}

        hist, bin_edges = np.histogram(all_diameters, bins=bins)
        return {
            'mean': float(np.mean(all_diameters)),
            'std': float(np.std(all_diameters)),
            'all_diameters': all_diameters
        }

    def plot_pore_distribution(self, real_stats: Dict, fake_stats: Dict, batches_done=None, save_path: str = None, writer=None, pshow=False):
        """
        Plot pore diameter distribution comparison
        Args:
            real_stats: pore statistics from real images
            fake_stats: pore statistics from generated images
            save_path: path to save the figure (None to not save)
        """
        real_diameters = real_stats['all_diameters']
        fake_diameters = fake_stats['all_diameters']

        # Plot histogram
        hist_fig = plt.figure(figsize=(10, 6))
        plt.hist(real_diameters, alpha=0.5, label='Real', density=True)
        plt.hist(fake_diameters, alpha=0.5, label='Generated', density=True)
        plt.xlabel('Pore Diameter (pixels)')
        plt.ylabel('Frequency')
        plt.title('Pore Size Distribution Comparison')
        plt.legend()

        # Plot violin plot
        df = pd.DataFrame({
            'Diameter': real_diameters + fake_diameters,
            'Type': ['Real']*len(real_diameters) + ['Generated']*len(fake_diameters)
        })

        v_fig = plt.figure(figsize=(10, 6))
        sns.violinplot(
            x='Type',
            y='Diameter',
            hue='Type',
            data=df,
            palette=['blue', 'orange'],
            cut=0,
            inner='quartile',
            legend=False
        )
        plt.title('Pore Size Distribution Comparison')
        plt.ylabel('Diameter (pixels)')
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if writer:
            writer.add_figure('pore_size_distribution', hist_fig, global_step=batches_done)
            writer.add_figure('pore_size_distribution_violin', v_fig, global_step=batches_done)

        plt.close(hist_fig)
        plt.close(v_fig)

    def calculate_two_point_stats(
        self,
        binary_image: np.ndarray,
        max_dx: int = 50,
        max_dy: int = 50
    ) -> Dict[str, Dict]:
        """
        Calculate two-point statistics
        Args:
            binary_image: binary image [N,C,H,W]
            max_dx: maximum displacement in x direction
            max_dy: maximum displacement in y direction
        Returns:
            Dictionary with two-point statistics
        """
        img_np = binary_image.squeeze(1)  # [N,H,W]
        results = {}

        for img in img_np:
            height, width = img.shape

            # Connected component labeling (8-connectivity)
            _, labels = cv2.connectedComponents(img.astype(np.uint8), connectivity=8)

            # Iterate over displacement vectors
            for dx in range(-max_dx, max_dx + 1):
                for dy in range(-max_dy, max_dy + 1):
                    if dx == 0 and dy == 0:
                        continue  # skip zero displacement

                    # Calculate overlapping region
                    x_start = max(0, -dx)
                    x_end = min(width, width - dx)
                    y_start = max(0, -dy)
                    y_end = min(height, height - dy)

                    if x_start >= x_end or y_start >= y_end:
                        continue  # no overlapping region

                    # Extract regions
                    region1 = img[y_start:y_end, x_start:x_end]
                    region2 = img[y_start+dy:y_end+dy, x_start+dx:x_end+dx]
                    label1 = labels[y_start:y_end, x_start:x_end]
                    label2 = labels[y_start+dy:y_end+dy, x_start+dx:x_end+dx]

                    total_pixels = region1.size

                    # Calculate probabilities
                    P00 = np.sum((region1 == 0) & (region2 == 0)) / total_pixels
                    P11 = np.sum((region1 == 1) & (region2 == 1)) / total_pixels
                    P01 = np.sum((region1 == 0) & (region2 == 1)) / total_pixels
                    P10 = np.sum((region1 == 1) & (region2 == 0)) / total_pixels

                    # Connection probability
                    mask = (region1 == 1) & (region2 == 1)
                    P_connect = np.sum((label1 == label2) & mask) / total_pixels

                    # Store results
                    key = (dx, dy)
                    if key not in results:
                        results[key] = {
                            'P00': [], 'P11': [],
                            'P01': [], 'P10': [],
                            'P_connect': []
                        }
                    results[key]['P00'].append(P00)
                    results[key]['P11'].append(P11)
                    results[key]['P01'].append(P01)
                    results[key]['P10'].append(P10)
                    results[key]['P_connect'].append(P_connect)

        # Calculate averages
        final_results = {}
        for key, values in results.items():
            final_results[key] = {
                'P00': np.mean(values['P00']),
                'P11': np.mean(values['P11']),
                'P01': np.mean(values['P01']),
                'P10': np.mean(values['P10']),
                'P_connect': np.mean(values['P_connect'])
            }

        return final_results

    def plot_two_point_stats(
        self,
        real_stats: Dict,
        fake_stats: Dict,
        batches_done: int = None,
        writer=None,
        save_path: str = None
    ):
        """
        Visualize two-point statistics comparison
        Args:
            real_stats: two-point stats from real images
            fake_stats: two-point stats from generated images
            batches_done: current training step (for TensorBoard)
            writer: TensorBoard SummaryWriter
            save_path: path to save the figure
        """
        displacements = list(real_stats.keys())
        metrics = ['P00', 'P11', 'P01', 'P10', 'P_connect']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics[:5]):
            ax = axes[i]

            real_values = [real_stats[d][metric] for d in displacements]
            fake_values = [fake_stats[d][metric] for d in displacements]
            distances = [np.sqrt(dx**2 + dy**2) for (dx, dy) in displacements]

            ax.scatter(distances, real_values, alpha=0.5, label='Real')
            ax.scatter(distances, fake_values, alpha=0.5, label='Generated')

            ax.set_xlabel('Displacement Distance (pixels)')
            ax.set_ylabel(metric)
            ax.set_title(f'Two-point {metric} Comparison')
            ax.legend()
            ax.grid(True)

        axes[-1].axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if writer and batches_done is not None:
            writer.add_figure('two_point_stats', fig, global_step=batches_done)

        plt.close(fig)

    def evaluate(
        self,
        real_images: np.ndarray,
        fake_images: np.ndarray,
        metrics: List[str] = ['iou', 'dice', 'hd', 'ssim', 'porosity', 'pore_size', 'two_point'],
    ) -> Dict:
        """
        Evaluation interface
        Args:
            real_images: real images [N,C,H,W], range [0,1]
            fake_images: generated images [N,C,H,W], range [0,1]
            metrics: metrics to calculate
        Returns:
            Dictionary with evaluation results
        """
        # Binarize images
        y_true = (real_images > self.threshold).astype(float)
        y_pred = (fake_images > self.threshold).astype(float)

        results = {}
        if 'iou' in metrics:
            results['iou'] = self.calculate_iou(y_true, y_pred)
        if 'dice' in metrics:
            results['dice'] = self.calculate_dice(y_true, y_pred)
        if 'hd' in metrics:
            results['hd'] = self.calculate_hd(y_true, y_pred)
        if 'ssim' in metrics:
            results['ssim'] = self.calculate_ssim(real_images, fake_images)
        if 'porosity' in metrics:
            results['porosity_real'] = self.calculate_porosity(y_true)
            results['porosity_fake'] = self.calculate_porosity(y_pred)
        if 'pore_size' in metrics:
            results['pore_size_real'] = self.calculate_pore_size_distribution(y_true)
            results['pore_size_fake'] = self.calculate_pore_size_distribution(y_pred)
        if 'two_point' in metrics:
            results['two_point_real'] = self.calculate_two_point_stats(y_true)
            results['two_point_fake'] = self.calculate_two_point_stats(y_pred)

        return results