import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from coco_utils import decode_captions


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    transformer_solver = instantiate(cfg.solver)
    transformer_solver.train()

    plt.plot(transformer_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()

    transformer_solver.sample()


def main():
    train()

if __name__ == "__main__":
    main()