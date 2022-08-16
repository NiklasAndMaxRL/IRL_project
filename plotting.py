import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_reward_2d(reward: np.ndarray, title: str = "Reward function", show: bool = True, fname: str = ""):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the heatmap
    im = ax.imshow(reward)

    # Add title
    ax.set_title(title)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Reward value", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(reward.shape[1]))
    ax.set_yticks(np.arange(reward.shape[0]))
    ax.set_xlabel("i-th column")
    ax.set_ylabel("i-th row")

    fig.tight_layout()

    # Output
    if show:
        plt.show()
    if fname:
        plt.savefig(fname)


def plot_reward_3d(reward: np.ndarray, title: str = "Reward function", show: bool = True, fname: str = ""):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))

    X, Y = np.meshgrid(np.arange(reward.shape[1]), np.arange(reward.shape[0]))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, reward.T,
                           color="cyan",  # cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    # Add a wireframe for nier plots
    ax.plot_wireframe(X, Y, reward.T,
                      linewidth=0.7, color="black")

    # Add title
    ax.set_title(title)

    # Create colorbar
    # fig.colorbar(surf, shrink=0.6, aspect=8)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(reward.shape[1]))
    ax.set_yticks(np.arange(reward.shape[0]))
    ax.set_xlabel("i-th column")
    ax.set_ylabel("i-th row")
    ax.set_zlabel("Reward value")

    ax.view_init(elev=20, azim=-25)

    # Output
    if show:
        plt.show()
    if fname:
        plt.savefig(fname)


if __name__ == "__main__":
    arr = np.array([[1.0, 0.5, 0.5, 0.1],
                    [0.5, 0.6, 0.1, 0.0],
                    [0.2, -.1, -.2, -.3],
                    [0.1, 0.6, -.2, -.1]])

    plot_reward_2d(arr)
    plot_reward_3d(arr)
