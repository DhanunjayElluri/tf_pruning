from dataloader.data_loader import LoadData
from configs.config import Config
from model.prune_model import PruneModel


def main():
    dataset = LoadData(Config)
    dataset = dataset.load_data()
    model = PruneModel(Config, dataset)
    model.set_params()
    model.compile_prune_model()
    model.fit_prune_model()
    model.evaluate_and_save_model()
    model.quantization()
    return


if __name__ == "__main__":
    main()