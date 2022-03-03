from dataloader.data_loader import LoadData
from configs.config import Config
from model.model import Model


def main():
    dataset = LoadData(Config)
    dataset = dataset.load_data()
    model = Model(Config, dataset)
    model.build_model()
    model.compile_model()
    model.fit_model()
    model.evaluate_and_save_model()
    return


if __name__ == "__main__":
    main()
