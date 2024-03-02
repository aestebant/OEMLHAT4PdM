from river import stream
from river.datasets import base


class Ai4i(base.RemoteDataset):
    """
    The AI4I 2020 Predictive Maintenance Dataset is a synthetic dataset that reflects real predictive maintenance data encountered in industry.

    Source: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
    """

    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=10000,
            n_features=14,
            n_outputs=5,
            unpack=True,
            url="https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
            filename="ai4i2020.csv",
            size=522_048,
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target=["TWF", "HDF", "PWF", "OSF", "RNF"],
            drop=["\ufeffUDI", "Product ID", "Machine failure"],
            converters={
                "Air temperature [K]": float,
                "Process temperature [K]": float,
                "Rotational speed [rpm]": float,
                "Torque [Nm]": float,
                "Tool wear [min]": float,
                "TWF": lambda x: x == "1",
                "HDF": lambda x: x == "1",
                "PWF": lambda x: x == "1",
                "OSF": lambda x: x == "1",
                "RNF": lambda x: x == "1",
            },
        )
