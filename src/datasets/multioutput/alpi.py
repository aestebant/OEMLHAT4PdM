from river.datasets import base
from river import stream
import pandas as pd
import datetime
import bisect
import numpy as np
from pathlib import Path


pd.options.mode.chained_assignment = None


class Alpi(base.FileDataset):
    """
    The Alarms Logs in Packaging Industry (ALPI) dataset consists of a sequence of alarms logged by packaging equipment in an industrial environment. The collection includes data logged by 20 machines, deployed in different plants around the world, from 2019-02-21 to 2020-06-17. There are 154 distinct alarm codes, whose distribution is highly unbalanced.

    Source: https://ieee-dataport.org/open-access/alarm-logs-packaging-industry-alpi

    Parameters:
    - machine: serial number of the machine to use, that can take numeric values from 0 to 19.
    - input_win: size of the input window in number of entries.
    - output_win: size of the output window in number of entries.
    - delta: time difference between the end of the input window and the start of the output window, in number of entries.
    - sigma: stride between consecutive input windows, in number of entries.
    """

    def __init__(self, machine, input_win=1720, output_win=480, delta=0, sigma=120, min_count=0):
        super().__init__(
            filename="alpi.csv",
            directory=Path(__file__).parent,
            task=base.MO_BINARY_CLF,
            n_features=-1,
            n_samples=-1,
        )
        self.machine = machine
        self.input_win = input_win
        self.output_win = output_win
        self.delta = delta
        self.sigma = sigma
        self.min_count = min_count
        if self.processed_dest().is_file():
            data = pd.read_csv(self.processed_dest())
            self.X = data.select_dtypes(exclude="bool")
            self.Y = data.select_dtypes(include="bool")
        else:
            self.process_raw()

        # Description of the dataset
        self.n_features = self.X.shape[1]
        self.n_samples = len(self.X)
        self.n_outputs = self.Y.shape[1]

    def processed_dest(self) -> Path:
        directory = Path(base.get_data_home(), self.__class__.__name__)
        filename = f"alpi_{self.machine}_{self.input_win}_{self.output_win}_{self.delta}_{self.sigma}_{self.min_count}.csv"
        return directory / filename

    def process_raw(self):
        df = pd.read_csv(self.directory / self.filename, index_col="timestamp", parse_dates=True)
        # Filter to the data of the current machine
        data = df.loc[df["serial"] == self.machine]
        min_timestamp = data.index.min()
        max_timestamp = data.index.max()
        delta_in = datetime.timedelta(minutes=self.input_win)
        delta_stride = datetime.timedelta(minutes=self.sigma)
        delta_out = datetime.timedelta(minutes=self.output_win)
        delta_sigma = datetime.timedelta(minutes=self.delta)

        # Date ranges for input and output
        input_starts = pd.date_range(
            min_timestamp, max_timestamp - delta_sigma - delta_out, freq=delta_stride
        ).to_list()
        input_date_range = [(start, start + delta_in) for start in input_starts]
        output_date_range = [
            (input_end + delta_sigma, input_end + delta_sigma + delta_out)
            for _, input_end in input_date_range
        ]
        input_date_range = np.asarray(input_date_range)
        output_date_range = np.asarray(output_date_range)

        # Split in samples
        def get_bins(ranges, target):
            start = bisect.bisect_right(ranges[:, 1], target)
            end = start
            for end in range(start, len(ranges)):
                if not (ranges[end, 0] <= target < ranges[end, 1]):
                    break
            return [start, end]  # [i for i in range(start-1, end+1)]

        input_bins = pd.Series(data.index).apply(lambda y: get_bins(input_date_range, y))
        input_bins.index = data.index
        data["bin_input"] = input_bins
        output_bins = pd.Series(data.index).apply(lambda y: get_bins(output_date_range, y))
        output_bins.index = data.index
        data["bin_output"] = output_bins

        # Create temporal windows of alarms
        inputs = dict()
        for bin_id, _ in enumerate(input_bins):
            alarms = (
                data.loc[data["bin_input"].apply(lambda bins: bin_id in range(*bins))]
                .alarm.sort_index()
                .values
            )
            if len(alarms) > 0:
                inputs[bin_id] = alarms
        outputs = dict()
        for bin_id, _ in enumerate(output_bins):
            alarms = data.loc[
                data["bin_output"].apply(lambda bins: bin_id in range(*bins))
            ].alarm.values
            if len(alarms) > 0:
                outputs[bin_id] = alarms

        # Match inputs and outputs
        candidates = list()
        inlens = list()
        periods_in = list(inputs.keys())
        periods_in.sort()
        for bin_id in periods_in:
            if len(inputs[bin_id]) >= self.min_count:
                # TODO: removal / relevance alarms
                x = pd.Series(inputs[bin_id])
                x = np.asarray(x[x.shift() != x].values)  # TODO: estudiar si es mejor no resumir
                # TODO: estudiar otra representación: la entrada es cuántas veces aparece cada alarma
                inlens.append(len(x))
                try:
                    y = set(outputs[bin_id])
                except KeyError:
                    y = set()
                candidates.append((x, y))

        # Construct the final dataframe
        X = list()
        Y = list()
        avg_input_seq, std_input_seq = np.mean(inlens), np.std(inlens)
        x_len = round(avg_input_seq + std_input_seq)
        for input, output in candidates:
            curr_x_len = len(input)
            if curr_x_len >= x_len:  # Trim longer input sequences
                x = input[:x_len]
            else:  # Pad shorter input sequences
                x = np.pad(input, (0, x_len - curr_x_len), "constant", constant_values=(0, 0))
            X.append({timestamp: v for timestamp, v in enumerate(x)})
            Y.append({label: True for label in output})
        self.X = pd.DataFrame(X)
        self.Y = pd.DataFrame(Y)
        self.Y.fillna(False, inplace=True)

        result = pd.concat([self.X, self.Y], axis=1)
        destination = self.processed_dest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)

    def __iter__(self):
        return stream.iter_pandas(self.X, self.Y)
