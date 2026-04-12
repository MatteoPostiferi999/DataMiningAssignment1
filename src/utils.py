import os
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def save_figure(filename: str):
    """Save the current matplotlib figure to the figures/ directory."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=150)
    print(f"Saved: figures/{filename}")
