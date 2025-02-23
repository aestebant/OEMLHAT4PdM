from pathlib import Path
from river.datasets import base
from river import stream


class NPS(base.FileDataset):
    """
    Condition Based Maintenance of Naval Propulsion Plants. Data have been generated from a sophisticated simulator of a Gas Turbines (GT), mounted on a Frigate characterized by a COmbined Diesel eLectric And Gas (CODLAG) propulsion plant type.

    Source: https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants
    """

    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=65473,
            n_features=23,
            n_outputs=4,
            filename="nps_sp15.csv",
            directory=Path(__file__).parent,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target=["PRP", "HLL", "GTC", "GT"],
            converters = {
                "GTT": float,
                "GT_rpm": float,
                "CPP_T_stbd": float,
                "CPP_T_port": float,
                "Q_port": float,
                "rpm_port": float,
                "Q_stdb": float,
                "rpm_stbd": float,
                "T48": float,
                "GG_rpm": float,
                "mf": float,
                "ABB_Tic": float,
                "P2": float,
                "T2": float,
                "Pext": float,
                "P48": float,
                "TCS_tic": float,
                "Kt_stbd": float,
                "rps_prop_stbd": float,
                "Kt_port": float,
                "rps_prop_port": float,
                "Q_prop_port": float,
                "Q_prop_stbd": float,
                "PRP": eval,
                "HLL": eval,
                "GTC": eval,
                "GT": eval,
            },
        )
