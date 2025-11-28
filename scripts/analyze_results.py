"""Analyze and visualize Neural OT training results.

Usage:
  python -m scripts.analyze_results --run-dir outputs/neural_ot/run_20240101_120000
  python -m scripts.analyze_results --run-dir outputs/neural_ot/run_20240101_120000 --export-pdf
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """Analyzer for Neural OT training results."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.embeddings_dir = self.run_dir / "embeddings"
        self.plots_dir = self.run_dir / "plots"
        self.checkpoints_dir = self.run_dir / "checkpoints"

        # Load config and metrics
        self.config = self._load_config()
        self.metrics_history = self._load_metrics_history()
        self.summary = self._load_summary()

        print(f"Loaded results from: {self.run_dir}")
        print(f"Total steps: {self.summary.get('total_steps', 'N/A')}")
        print(f"Total epochs: {self.summary.get('total_epochs', 'N/A')}")
        print(f"Best loss: {self.summary.get('best_loss', 'N/A')}")

    def _load_config(self) -> Dict:
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    def _load_metrics_history(self) -> Dict:
        metrics_path = self.metrics_dir / "metrics_history.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {}

    def _load_summary(self) -> Dict:
        summary_path = self.run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
        return {}

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

        # Extract metrics (try both with and without namespace prefixes)
        steps = self.metrics_history.get('step', [])
        
        # Helper to find key with variants
        def get_metric(base_key):
            variants = [base_key, f"train/{base_key}", f"val/{base_key}", base_key.replace("loss", "_loss")]
            for k in variants:
                if k in self.metrics_history:
                    return self.metrics_history[k]
            return []

        train_loss = get_metric('loss')
        fm_loss = get_metric('fm_loss')
        cycle_loss = get_metric('cycle_loss')
        lr = get_metric('learning_rate')
        grad_norm = get_metric('grad_norm')

        # Plot 1: Total loss
        if train_loss:
            axes[0, 0].plot(steps[:len(train_loss)], train_loss, linewidth=2, color='#2E86AB')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Total Training Loss')
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Flow matching loss
        if fm_loss:
            axes[0, 1].plot(steps[:len(fm_loss)], fm_loss, linewidth=2, color='#A23B72')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Flow Matching Loss')
            axes[0, 1].set_title('Flow Matching Loss')
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Learning rate
        if lr:
            axes[1, 0].plot(steps[:len(lr)], lr, linewidth=2, color='#F18F01')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Gradient norm
        if grad_norm:
            axes[1, 1].plot(steps[:len(grad_norm)], grad_norm, linewidth=2, color='#C73E1D')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training curves to {save_path}")
        else:
            plt.savefig(self.plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            print(f"Saved training curves to {self.plots_dir / 'training_curves.png'}")

        plt.close()

    def plot_velocity_field_analysis(self, save_path: Optional[str] = None):
        """Plot velocity field statistics over training."""
        # Find all velocity profile files
        velocity_files = sorted(self.metrics_dir.glob("velocity_profile_epoch*.npz"))

        if not velocity_files:
            print("No velocity field analysis data found.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Velocity Field Analysis', fontsize=16, fontweight='bold')

        epochs = []
        mean_norms = []
        time_profiles = []

        for vf_file in velocity_files:
            data = np.load(vf_file)
            epoch = int(data['epoch'])
            epochs.append(epoch)
            velocity_norms = data['velocity_norms']
            mean_norms.append(np.mean(velocity_norms))
            time_profiles.append(velocity_norms)

        # Plot 1: Mean velocity norm over epochs
        axes[0].plot(epochs, mean_norms, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Mean Velocity Norm')
        axes[0].set_title('Velocity Field Magnitude Over Training')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Velocity profiles at different epochs
        if time_profiles:
            time_points = np.load(velocity_files[0])['time_points']

            # Plot first, middle, and last epochs
            indices = [0, len(epochs)//2, -1] if len(epochs) >= 3 else range(len(epochs))
            colors = ['#2E86AB', '#A23B72', '#F18F01']

            for idx, color in zip(indices, colors):
                if idx < len(time_profiles):
                    axes[1].plot(time_points, time_profiles[idx],
                               marker='o', linewidth=2, label=f'Epoch {epochs[idx]}',
                               color=color)

            axes[1].set_xlabel('Time t')
            axes[1].set_ylabel('Velocity Norm')
            axes[1].set_title('Velocity Field Profile Across Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved velocity field analysis to {save_path}")
        else:
            plt.savefig(self.plots_dir / "velocity_field_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Saved velocity field analysis to {self.plots_dir / 'velocity_field_analysis.png'}")

        plt.close()

    def plot_transport_analysis(self, save_path: Optional[str] = None):
        """Plot transport quality metrics over training."""
        # Find all transport analysis files
        transport_files = sorted(self.embeddings_dir.glob("transport_analysis_epoch*.npz"))

        if not transport_files:
            print("No transport analysis data found.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Transport Quality Analysis', fontsize=16, fontweight='bold')

        epochs = []
        nl_to_lean_dist = []
        lean_to_nl_dist = []
        cycle_errors = []
        original_sims = []
        transported_sims = []

        for tf_file in transport_files:
            data = np.load(tf_file, allow_pickle=True)
            metrics = data['metrics'].item()
            epoch = int(data['epoch'])

            epochs.append(epoch)
            nl_to_lean_dist.append(metrics.get('transport/nl_to_lean_distance', 0))
            lean_to_nl_dist.append(metrics.get('transport/lean_to_nl_distance', 0))
            cycle_errors.append(metrics.get('transport/cycle_error', 0))
            original_sims.append(metrics.get('transport/original_similarity', 0))
            transported_sims.append(metrics.get('transport/transported_similarity', 0))

        # Plot 1: Transport distances
        axes[0, 0].plot(epochs, nl_to_lean_dist, marker='o', linewidth=2,
                       label='NL → Lean', color='#2E86AB')
        axes[0, 0].plot(epochs, lean_to_nl_dist, marker='s', linewidth=2,
                       label='Lean → NL', color='#A23B72')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('L2 Distance')
        axes[0, 0].set_title('Transport Distance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Cycle consistency
        axes[0, 1].plot(epochs, cycle_errors, marker='o', linewidth=2, color='#C73E1D')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cycle Error')
        axes[0, 1].set_title('Cycle Consistency Error')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Cosine similarities
        axes[1, 0].plot(epochs, original_sims, marker='o', linewidth=2,
                       label='Original', color='#6C757D')
        axes[1, 0].plot(epochs, transported_sims, marker='s', linewidth=2,
                       label='After Transport', color='#28A745')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('NL-Lean Alignment')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Similarity improvement
        improvements = [t - o for t, o in zip(transported_sims, original_sims)]
        axes[1, 1].plot(epochs, improvements, marker='o', linewidth=2, color='#F18F01')
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Similarity Improvement')
        axes[1, 1].set_title('Transport Quality Improvement')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved transport analysis to {save_path}")
        else:
            plt.savefig(self.plots_dir / "transport_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Saved transport analysis to {self.plots_dir / 'transport_analysis.png'}")

        plt.close()

    def plot_embedding_visualization(self, epoch: Optional[int] = None, save_path: Optional[str] = None):
        """Visualize embeddings using PCA at a specific epoch."""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("scikit-learn not available. Install with: pip install scikit-learn")
            return

        # Find transport analysis files
        transport_files = sorted(self.embeddings_dir.glob("transport_analysis_epoch*.npz"))

        if not transport_files:
            print("No embedding data found.")
            return

        # Use specified epoch or last epoch
        if epoch is not None:
            target_file = self.embeddings_dir / f"transport_analysis_epoch{epoch}.npz"
            if not target_file.exists():
                print(f"No data for epoch {epoch}")
                return
        else:
            target_file = transport_files[-1]

        data = np.load(target_file, allow_pickle=True)
        epoch_num = int(data['epoch'])

        # Load embeddings
        x_nl = data['x_nl_sample']
        x_lean = data['x_lean_sample']
        x_nl_transported = data['x_nl_transported_sample']

        # Flatten to 2D (batch*seq_len, hidden_dim)
        x_nl_flat = x_nl.reshape(-1, x_nl.shape[-1])
        x_lean_flat = x_lean.reshape(-1, x_lean.shape[-1])
        x_nl_transported_flat = x_nl_transported.reshape(-1, x_nl_transported.shape[-1])

        # Apply PCA
        pca = PCA(n_components=2)
        all_embeddings = np.vstack([x_nl_flat, x_lean_flat, x_nl_transported_flat])
        pca.fit(all_embeddings)

        nl_2d = pca.transform(x_nl_flat)
        lean_2d = pca.transform(x_lean_flat)
        nl_transported_2d = pca.transform(x_nl_transported_flat)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Embedding Space Visualization (Epoch {epoch_num})',
                    fontsize=16, fontweight='bold')

        # Plot 1: Before transport
        axes[0].scatter(nl_2d[:, 0], nl_2d[:, 1], c='#2E86AB',
                       label='NL', alpha=0.6, s=50)
        axes[0].scatter(lean_2d[:, 0], lean_2d[:, 1], c='#A23B72',
                       label='Lean', alpha=0.6, s=50)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        axes[0].set_title('Before Transport')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: After transport
        axes[1].scatter(nl_transported_2d[:, 0], nl_transported_2d[:, 1],
                       c='#28A745', label='NL → Lean', alpha=0.6, s=50)
        axes[1].scatter(lean_2d[:, 0], lean_2d[:, 1], c='#A23B72',
                       label='Lean', alpha=0.6, s=50)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        axes[1].set_title('After Transport')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved embedding visualization to {save_path}")
        else:
            plt.savefig(self.plots_dir / f"embeddings_epoch{epoch_num}.png",
                       dpi=300, bbox_inches='tight')
            print(f"Saved embedding visualization to {self.plots_dir / f'embeddings_epoch{epoch_num}.png'}")

        plt.close()

    def generate_summary_report(self, output_path: Optional[str] = None):
        """Generate a comprehensive text summary report."""
        if output_path is None:
            output_path = self.run_dir / "analysis_report.txt"

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HERMES Neural OT Training Analysis Report\n")
            f.write("="*80 + "\n\n")

            # Basic info
            f.write("RUN INFORMATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"Total Steps: {self.summary.get('total_steps', 'N/A')}\n")
            f.write(f"Total Epochs: {self.summary.get('total_epochs', 'N/A')}\n")
            f.write(f"Final Loss: {self.summary.get('final_loss', 'N/A')}\n")
            f.write(f"Best Loss: {self.summary.get('best_loss', 'N/A')}\n\n")

            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-"*80 + "\n")
            if self.config:
                f.write(json.dumps(self.config, indent=2))
                f.write("\n\n")

            # Training metrics summary
            if self.metrics_history:
                f.write("TRAINING METRICS SUMMARY\n")
                f.write("-"*80 + "\n")

                # Try multiple key variants
                for key in ['loss', 'train_loss', 'flow_matching_loss', 'avg_fm_loss', 'cycle_loss', 'avg_cycle_loss']:
                    if key in self.metrics_history and self.metrics_history[key]:
                        values = self.metrics_history[key]
                        f.write(f"\n{key}:\n")
                        f.write(f"  Initial: {values[0]:.6f}\n")
                        f.write(f"  Final: {values[-1]:.6f}\n")
                        f.write(f"  Min: {min(values):.6f}\n")
                        f.write(f"  Max: {max(values):.6f}\n")
                        f.write(f"  Mean: {float(np.mean(values)):.6f}\n")
                        f.write(f"  Std: {float(np.std(values)):.6f}\n")

            # Transport analysis summary
            transport_files = sorted(self.embeddings_dir.glob("transport_analysis_epoch*.npz"))
            if transport_files:
                f.write("\n\nTRANSPORT QUALITY SUMMARY\n")
                f.write("-"*80 + "\n")

                # Get first and last epoch metrics
                first_data = np.load(transport_files[0], allow_pickle=True)
                last_data = np.load(transport_files[-1], allow_pickle=True)

                first_metrics = first_data['metrics'].item()
                last_metrics = last_data['metrics'].item()

                f.write("\nInitial Epoch:\n")
                for key, value in first_metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")

                f.write("\nFinal Epoch:\n")
                for key, value in last_metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")

                f.write("\nImprovement:\n")
                for key in first_metrics.keys():
                    if key in last_metrics:
                        improvement = last_metrics[key] - first_metrics[key]
                        f.write(f"  {key}: {improvement:+.6f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")

        print(f"Saved summary report to {output_path}")

    def create_comprehensive_dashboard(self, output_path: Optional[str] = None):
        """Create a single comprehensive dashboard with all visualizations."""
        if output_path is None:
            output_path = self.plots_dir / "comprehensive_dashboard.png"

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('HERMES Neural OT Training Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)

        # 1. Training loss
        ax1 = fig.add_subplot(gs[0, 0])
        steps = self.metrics_history.get('step', [])
        train_loss = self.metrics_history.get('loss', self.metrics_history.get('train_loss', []))
        if train_loss:
            ax1.plot(steps[:len(train_loss)], train_loss, linewidth=2, color='#2E86AB')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)

        # 2. Flow matching loss
        ax2 = fig.add_subplot(gs[0, 1])
        fm_loss = self.metrics_history.get('flow_matching_loss', self.metrics_history.get('avg_fm_loss', []))
        if fm_loss:
            ax2.plot(steps[:len(fm_loss)], fm_loss, linewidth=2, color='#A23B72')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('FM Loss')
            ax2.set_title('Flow Matching Loss')
            ax2.grid(True, alpha=0.3)

        # 3. Learning rate
        ax3 = fig.add_subplot(gs[0, 2])
        lr = self.metrics_history.get('learning_rate', [])
        if lr:
            ax3.plot(steps[:len(lr)], lr, linewidth=2, color='#F18F01')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)

        # 4. Gradient norm
        ax4 = fig.add_subplot(gs[1, 0])
        grad_norm = self.metrics_history.get('grad_norm', [])
        if grad_norm:
            ax4.plot(steps[:len(grad_norm)], grad_norm, linewidth=2, color='#C73E1D')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Grad Norm')
            ax4.set_title('Gradient Norm')
            ax4.grid(True, alpha=0.3)

        # 5. Transport distances
        ax5 = fig.add_subplot(gs[1, 1])
        transport_files = sorted(self.embeddings_dir.glob("transport_analysis_epoch*.npz"))
        if transport_files:
            epochs = []
            nl_to_lean = []
            for tf in transport_files:
                data = np.load(tf, allow_pickle=True)
                epochs.append(int(data['epoch']))
                metrics = data['metrics'].item()
                nl_to_lean.append(metrics.get('transport/nl_to_lean_distance', 0))

            ax5.plot(epochs, nl_to_lean, marker='o', linewidth=2, color='#2E86AB')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Distance')
            ax5.set_title('Transport Distance (NL → Lean)')
            ax5.grid(True, alpha=0.3)

        # 6. Similarity improvement
        ax6 = fig.add_subplot(gs[1, 2])
        if transport_files:
            epochs = []
            improvements = []
            for tf in transport_files:
                data = np.load(tf, allow_pickle=True)
                epochs.append(int(data['epoch']))
                metrics = data['metrics'].item()
                improvements.append(metrics.get('transport/similarity_improvement', 0))

            ax6.plot(epochs, improvements, marker='o', linewidth=2, color='#28A745')
            ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Improvement')
            ax6.set_title('Similarity Improvement')
            ax6.grid(True, alpha=0.3)

        # 7. Velocity field magnitude
        ax7 = fig.add_subplot(gs[2, 0])
        velocity_files = sorted(self.metrics_dir.glob("velocity_profile_epoch*.npz"))
        if velocity_files:
            epochs = []
            mean_norms = []
            for vf in velocity_files:
                data = np.load(vf)
                epochs.append(int(data['epoch']))
                mean_norms.append(np.mean(data['velocity_norms']))

            ax7.plot(epochs, mean_norms, marker='o', linewidth=2, color='#6C757D')
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Mean Norm')
            ax7.set_title('Velocity Field Magnitude')
            ax7.grid(True, alpha=0.3)

        # 8. Cycle consistency
        ax8 = fig.add_subplot(gs[2, 1])
        if transport_files:
            epochs = []
            cycle_errors = []
            for tf in transport_files:
                data = np.load(tf, allow_pickle=True)
                epochs.append(int(data['epoch']))
                metrics = data['metrics'].item()
                cycle_errors.append(metrics.get('transport/cycle_error', 0))

            ax8.plot(epochs, cycle_errors, marker='o', linewidth=2, color='#C73E1D')
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Cycle Error')
            ax8.set_title('Cycle Consistency')
            ax8.grid(True, alpha=0.3)

        # 9. Summary statistics (text)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        best_loss = self.summary.get('best_loss')
        final_loss = self.summary.get('final_loss')

        best_loss_str = f"{best_loss:.6f}" if best_loss is not None else 'N/A'
        final_loss_str = f"{final_loss:.6f}" if final_loss is not None else 'N/A'

        summary_text = f"""
Summary Statistics

Total Steps: {self.summary.get('total_steps', 'N/A')}
Total Epochs: {self.summary.get('total_epochs', 'N/A')}

Best Loss: {best_loss_str}
Final Loss: {final_loss_str}

Model Parameters:
- Hidden Dim: {self.config.get('neural_ot', {}).get('hidden_dim', 'N/A')}
- Num Layers: {self.config.get('neural_ot', {}).get('num_layers', 'N/A')}
- MLP Width: {self.config.get('neural_ot', {}).get('mlp_width', 'N/A')}

Learning Rate: {self.config.get('neural_ot', {}).get('learning_rate', 'N/A')}
Batch Size: {self.config.get('neural_ot', {}).get('batch_size', 'N/A')}
"""
        ax9.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive dashboard to {output_path}")
        plt.close()

    def analyze_all(self, export_pdf: bool = False):
        """Run all analysis and generate all plots."""
        print("\n" + "="*80)
        print("Running comprehensive analysis...")
        print("="*80 + "\n")

        # Generate all plots
        self.plot_training_curves()
        self.plot_velocity_field_analysis()
        self.plot_transport_analysis()
        self.plot_embedding_visualization()
        self.create_comprehensive_dashboard()
        self.generate_summary_report()

        # Export to PDF if requested
        if export_pdf:
            try:
                from matplotlib.backends.backend_pdf import PdfPages

                pdf_path = self.run_dir / "analysis_report.pdf"
                with PdfPages(pdf_path) as pdf:
                    # Re-create plots and save to PDF
                    figs = []

                    # Training curves
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    # ... (plot code)
                    pdf.savefig(fig)
                    plt.close(fig)

                print(f"Exported analysis to PDF: {pdf_path}")
            except ImportError:
                print("matplotlib.backends.backend_pdf not available for PDF export")

        print("\n" + "="*80)
        print("Analysis complete!")
        print(f"Results saved to: {self.run_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize Neural OT training results"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the run directory (e.g., outputs/neural_ot/run_20240101_120000)"
    )
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="Export analysis to PDF"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Specific epoch for embedding visualization"
    )

    args = parser.parse_args()

    # Create analyzer and run analysis
    analyzer = ResultsAnalyzer(args.run_dir)
    analyzer.analyze_all(export_pdf=args.export_pdf)


if __name__ == "__main__":
    main()
