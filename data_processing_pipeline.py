import sys

from brics_toolkit.utils.config import *
import argparse
from brics_toolkit.data_containers import BRVDataClean
from brics_toolkit.data_containers import MeasurementData, MeasurementMetadata
from brics_toolkit.data_processing.initial_data_processing import handle_input_data, initial_data_processing
from brics_toolkit.data_processing.extract_features import extract_features
from brics_toolkit.database_access.database_handler import DatabaseHandler

def split_data_into_segments(input_file : Path, BRV_data_clean : BRVDataClean):
    """
        Split the resampled ADC data into segments that contain values from a specific time window,
        and save each segment into a separate JSONL file.

        Parameters
        ----------
        input_file: Path
            The path to the raw input file containing the ADC data.
        BRV_data_clean : BRVDataClean
            The BRVDataClean object containing the cleaned and resampled ADC data and timestamps further
            defined in project's DTP.

        Returns
        -------
        None          
        
        Side Effects
        ------------
        This function creates multiple JSONL files in the "./results" directory, each containing a segment of the ADC data.
    """

    segment_index = 0
    total_segments = int(np.ceil(BRV_data_clean.timestamps[-1] / SEGMENT_LENGTH_MS))
    temp_name = input_file.name
    temp = temp_name.split('.')
    temp.pop()
    filename = ".".join(temp)
    

    for segment_index in range(total_segments):
        segment_start = segment_index * SEGMENT_LENGTH_MS
        segment_end = segment_start + SEGMENT_LENGTH_MS
        with open(f"./results/clean_{filename}_{segment_index}.jsonl", 'w') as o_f:
            for i in range(len(BRV_data_clean.timestamps)):
                if segment_start <= BRV_data_clean.timestamps[i] < segment_end:
                    record = {
                        "timestamp": int(BRV_data_clean.timestamps[i]),
                        "adc_outputs": [BRV_data_clean.adc_data[a][i] for a in range(ADC_COUNT)]
                    }
                    o_f.write(json.dumps(record) + "\n")

def save_clean_data(BRV_data_clean : BRVDataClean, input_file : Path):
    os.makedirs("results/clean", exist_ok=True)
    results_path = f"results/clean/clean_{input_file}"
    with open(results_path, 'w') as f:
        for i in range(len(BRV_data_clean.timestamps)):
            record = {
                "timestamp": int(BRV_data_clean.timestamps[i]),
                "adc_outputs": [BRV_data_clean.adc_data[a][i] for a in range(ADC_COUNT)]
            }
            f.write(json.dumps(record) + "\n")
    return Path(results_path)

def clear_results_folder():
    # remove all everythong in results directory
    with os.scandir('results') as results:
        for result in results:
            if result.is_file():
                os.remove(result.path)

def plot_raw_data(input_file: Path, plot_enabled: bool = False, debug_plot: bool = False):
    if not plot_enabled:
        return

    timestamps, adc_output_data = handle_input_data(input_file)
    segment_length_ms = 5 * 60 * 1000
    segment_count = int(np.ceil(timestamps[-1] / segment_length_ms))

    os.makedirs("results/raw_segments", exist_ok=True)

    for segment_index in range(segment_count):
        segment_start = segment_index * segment_length_ms
        segment_end = segment_start + segment_length_ms
        mask = (timestamps >= segment_start) & (timestamps < segment_end)

        if not np.any(mask):
            continue

        segment_timestamps = timestamps[mask]

        fig, axes = plt.subplots(
            ADC_COUNT,
            1,
            figsize=(12, 12),
            sharex=True
        )

        for adc_index, ax in enumerate(axes):
            ax.plot(
                segment_timestamps,
                adc_output_data[adc_index][mask],
                linewidth=0.8
            )
            ax.set_title(f"ADC {adc_index} Output")
            ax.set_ylabel("Voltage (V)")
            if debug_plot:
                ax.grid(True)

        axes[-1].set_xlabel("Time from recording start (ms)")
        fig.suptitle(
            f"{input_file.name} - "
            f"minutes {segment_start // 60000}-{segment_end // 60000}"
        )
        fig.tight_layout()

        output_path = (
            f"results/raw_segments/"
            f"{input_file.stem}_segment_{segment_index + 1}.png"
        )
        fig.savefig(output_path, dpi=150)
        plt.show()
        plt.close(fig)

def main():
    input_file_str = input("Raw input file name (with extension) located in the data folder:\n")
    input_file = Path(input_file_str)
    plot_enabled = input("Enable plotting 0/1\n") == "1"
    debug_plot = input("Debug plot 0/1\n") == "1"

    measurement_data = MeasurementData()
    measurement_metadata = MeasurementMetadata()
    measurement_metadata.filepath_raw = Path(input_file)
    measurement_data.metadata = measurement_metadata

    plot_raw_data(input_file, plot_enabled=plot_enabled, debug_plot=debug_plot)
    return
    
    initial_data_processing(BRV_measurement_data = measurement_data, target_adc = TARGET_ADC, plot_enabled = plot_enabled)
    measurement_metadata.filepath_raw = Path(f"./data/{input_file}")
    measurement_data.metadata.filepath_clean = save_clean_data(measurement_data.data_clean, input_file)
    split_data_into_segments(Path(input_file), measurement_data.data_clean)
    extract_features(measurement_data=measurement_data)

    db_handler = DatabaseHandler()
    fr = measurement_data.metadata.filepath_raw
    fc = measurement_data.metadata.filepath_clean
    ff = measurement_data.metadata.filepath_features
    print("fr:", str(fr))
    print("fc:", str(fc))
    print("ff:", str(ff))
    db_handler.uploadMeasurement(filepath_raw=str(fr), filepath_clean=str(fc), filepath_features=str(ff))
    clear_results_folder()


"""
    BRV_measurement_data = poprawna iniclajizacja PUSTEGO obiektu MeasurementData
    BRV_measurement_data.BRV_data_intermediate = process_raw_file(input_file, plot_enabled=plot_enabled)
    BRV_measurement_data.BRV_data_clean = outlier_detection(BRV_measurement_data.BRV_data_intermediate, target_adc=TARGET_ADC, plot_enabled=plot_enabled)
    split_data_into_segments(input_file, BRV_measurement_data.BRV_data_clean)
    extract_features(BRV_measurement_data, target_adc=TARGET_ADC, plot_enabled=plot_enabled)
    remove_data_segments(input_file)
    input_measurement_metadata() -> ostatecznie wypełniamy measurement_metadata i chyba też measurement_data przed uploadem
    upload_measurement()
"""

if __name__ == "__main__":
    main()