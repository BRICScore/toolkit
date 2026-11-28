import sys, argparse, random
from visualize_data import FeatureData, load_features_from_database_zip

from brics_toolkit.utils.config import *
from brics_toolkit.data_containers import MeasurementData
from brics_toolkit.utils.file_processing.measurement_data_builder import MeasurementDataBuilder
from brics_toolkit.utils.measurement_dataset_control import MeasurementDatasetHook

def parse_jsonl_line(line):
    """
        Parses a single line of a JSONL file and extracts the timestamp and ADC values.

        Parameters
        ----------
        line : str
            A single line from a JSONL file containing a JSON object with 'timestamp' and 'adc_outputs'.
        
        Returns
        -------
        timestamp : int
            The timestamp extracted from the JSON object.
        adc_values : list of int
            A list of ADC output values extracted from the JSON object.
        
        Side Effects
        ------------
        This function has no side effects.
    """

    data = json.loads(line)
    timestamp = data.get('timestamp', 0)
    adc_values = [data.get('adc_outputs')[i] for i in range(ADC_COUNT)]
    return timestamp, adc_values

def extract_breath_features(signal):
    """ 
        Initializes breath features and extracts depth and length from a single breath signal.

        Parameters
        ----------
        signal : list of float
            The normalized ADC signal corresponding to a single breath.

        Returns
        -------
        dict
            A dictionary containing the initialized and extracted breath features:
            depth, length, inhale duration, inspiratory pause duration, exhale duration
            and expiratory pause duration.

        Side Effects        
        ------------
        This function has no side effects.
    """

    depth = np.max(signal) - np.min(signal)
    length = len(signal)

    return {
        "depth": float(depth),
        "length": int(length),
        "inhale": 0.0,
        "inspiratory_pause": 0.0,
        "exhale": 0.0,
        "expiratory_pause": 0.0,
    }

def detect_breath_peaks(signal):
    """
    Detects breath peaks (minima and maxima) in the given signal using the scipy.signal.find_peaks function.

    Parameters
    ----------
    signal : list of float
        The normalized ADC signal from which to detect breath peaks.

    Returns
    -------
    tuple: list of int (minima), list of int (maxima)
        A tuple containing the indices of the detected minima and maxima.

    Side Effects
    ------------
    This function has no side effects.
    """

    inverted_signal = [-x for x in signal]
    mean_signal = np.mean(signal)
    std_dev_signal = np.std(signal)
    maxima, _ = scipy.signal.find_peaks(signal, distance=MIN_DISTANCE, height=mean_signal + std_dev_signal*STD_DEV_CONST)
    minima, _ = scipy.signal.find_peaks(inverted_signal, distance=MIN_DISTANCE, height=mean_signal + std_dev_signal*STD_DEV_CONST)

    return minima, maxima

def smooth_breath_signal(signal):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < 2:
        raise ValueError("A breath signal must contain at least two samples")

    window_length = min(9, len(signal) if len(signal) % 2 else len(signal) - 1)
    if window_length < 5:
        return signal.copy()
    return scipy.signal.savgol_filter(signal, window_length=window_length, polyorder=2)

def normalize_breath_shape(signal, shape_points=64):
    smoothed = smooth_breath_signal(signal)
    resampled = np.interp(
        np.linspace(0.0, 1.0, shape_points),
        np.linspace(0.0, 1.0, len(smoothed)),
        smoothed,
    )
    lower, upper = np.percentile(resampled, [5, 95])
    scale = (upper - lower) / 2.0
    if scale <= np.finfo(float).eps:
        return np.zeros(shape_points)
    return (resampled - (lower + upper) / 2.0) / scale

def get_mode_breaths(all_breath_data, mode_values, avg_values, file_cache):
    """ 
    Selects representative breaths using scalar breath features and normalized
    waveform similarity across all ADC channels.
    
    Parameters
    ----------
    all_breath_data : list of dict
        A list of dictionaries, where each dictionary contains the features of a single breath.
    mode_values : dict
        A dictionary containing the mode values for breath features (depth, length, inhale duration,
        exhale duration, inspiratory pause duration, expiratory pause duration).
    avg_values : dict
        A dictionary containing the average values for breath features (depth, length, inhale duration,
        exhale duration, inspiratory pause duration, expiratory pause duration).
    file_cache : dict
        Parsed clean measurements indexed by their identifiers.

    Returns
    -------
    representative_breath : dict
        A dictionary containing the features of the most representative breath (mode breath) from the dataset.
    representative_breaths_weights: list of int
        A list of weights corresponding to the selected mode breaths, where the weight is inversely proportional
        to the distance from mode values

    Side Effects
    ------------
    This function has no side effects.
    """

    if not all_breath_data:
        raise ValueError("Cannot select mode breaths from an empty breath set")

    weights = {
        'depth': 0.3,
        'length': 0.3,
        'inhale': 1.0,
        'exhale': 1.0,
    }
    feature_distances = []
    candidate_shapes = []

    for breath in all_breath_data:
        dist = 0.0
        dist += weights['depth'] * abs(breath['depth'] - avg_values['avg_depth'])
        dist += weights['length'] * abs(breath['length'] - avg_values['avg_length'])
        dist += weights['inhale'] * abs(breath['inhale'] - avg_values['avg_inhale'])
        dist += weights['exhale'] * abs(breath['exhale'] - avg_values['avg_exhale'])
        dist += weights['depth'] * abs(breath['depth'] - mode_values['mode_depth'])
        dist += weights['length'] * abs(breath['length'] - mode_values['mode_length'])
        dist += weights['inhale'] * abs(breath['inhale'] - mode_values['mode_inhale'])
        dist += weights['exhale'] * abs(breath['exhale'] - mode_values['mode_exhale']) 
        # dist += (breath['inspiratory_pause'] - mode_values['mode_inspiratory_pause'])
        # dist += (breath['expiratory_pause'] - mode_values['mode_expiratory_pause'])

        feature_distances.append(dist)
        file_data = file_cache[breath["file_name"]]
        start_idx = breath["start_idx"]
        end_idx = breath["end_idx"]
        candidate_shapes.append([
            normalize_breath_shape(file_data["adc_signals"][adc][start_idx:end_idx])
            for adc in range(ADC_COUNT)
        ])

    candidate_shapes = np.asarray(candidate_shapes)
    shape_template = np.median(candidate_shapes, axis=0)
    shape_distances = np.sqrt(
        np.mean(np.square(candidate_shapes - shape_template), axis=(1, 2))
    )

    feature_distances = np.asarray(feature_distances, dtype=float)
    feature_scale = max(float(np.median(feature_distances)), np.finfo(float).eps)
    shape_scale = max(float(np.median(shape_distances)), np.finfo(float).eps)
    combined_distances = feature_distances / feature_scale + shape_distances / shape_scale
    selected_indices = np.argsort(combined_distances, kind="stable")[:MODE_BREATH_COUNT]
    selected_count = len(selected_indices)
    selected_breaths = [all_breath_data[index] for index in selected_indices]
    representative_breaths_weights = [
        1.0 - 0.5 * i / max(selected_count - 1, 1)
        for i in range(selected_count)
    ]

    mode_breaths = [[] for _ in range(ADC_COUNT)]
    for breath in selected_breaths:
        file_name = breath["file_name"]
        start_idx = breath["start_idx"]
        end_idx = breath["end_idx"]

        file_data = file_cache[file_name]
        timestamps = file_data["timestamps"][start_idx:end_idx]
        adc_signals = file_data["adc_signals"]
        for adc in range(ADC_COUNT):
            signal = smooth_breath_signal(adc_signals[adc][start_idx:end_idx])

            adc_breath = {
                "file_name": file_name,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "adc": adc,
                "signal": signal,
                "timestamps": timestamps,
            }

            if adc == TARGET_ADC:
                adc_breath["inhale"] = breath["inhale"]
                adc_breath["inspiratory_pause"] = breath["inspiratory_pause"]
                adc_breath["exhale"] = breath["exhale"]
                adc_breath["expiratory_pause"] = breath["expiratory_pause"]
                adc_breath["file_timestamps"] = file_data["timestamps"]
                adc_breath["file_signals"] = adc_signals

            mode_breaths[adc].append(adc_breath)

    return mode_breaths, representative_breaths_weights

    # return representative_breaths_target_adc, representative_breaths_weights

def calculate_breathing_phases_for_breath(breath_signal, breath_timestamps, peak_index, start_idx, end_idx):
    """ 
    Calculates the durations of the breathing phases (inhale, inspiratory pause, exhale, expiratory pause) for a single breath.

    Parameters
    ----------
    breath_signal : list of float
        The normalized ADC signal corresponding to a single breath.
    breath_timestamps : list of int
        The timestamps corresponding to the breath signal.
    peak_index : int
        The index of the peak (maximum) in the breath signal.
    start_idx : int
        The index of the start of the breath (first minimum).
    end_idx : int
        The index of the end of the breath (the minimum after the peak).

    Returns
    -------
    phases : list of float
        A list containing the durations of the breathing phases: [inhale, inspiratory pause, exhale, expiratory pause].
    
    Side Effects
    ------------
    This function has no side effects.
    """

    if len(breath_signal) < 3:
        return 0.0, 0.0, 0.0, 0.0
    
    phases = [0.0, 0.0, 0.0, 0.0]  # inhale, inspiratory_pause, exhale, expiratory_pause
    phases[0] = breath_timestamps[start_idx] - breath_timestamps[peak_index]
    phases[1] = 0.0
    phases[2] = breath_timestamps[end_idx] - breath_timestamps[peak_index]
    phases[3] = 0.0

    return phases

def extract_breath_data_from_measurement(measurement_data):
    """
    Extracts minima and maxima indices from a cleaned measurement loaded from the dataset ZIP.

    Parameters
    ----------
    measurement_data : MeasurementData
        A measurement populated by MeasurementDataBuilder for the "clean" target.

    Returns
    -------
    dict
        A dictionary containing the extracted breath data:
        - minima: A list of indices of detected minima in the ADC signal.
        - maxima: A list of indices of detected maxima in the ADC signal.
        - timestamps: A list of timestamps corresponding to the ADC signal samples.
        - adc_signals: A list of lists, each inner list contains the ADC signal values for one of the 5 ADC channels.

    Side Effects
    ------------
    This function has no side effects.
    """

    timestamps = np.asarray(measurement_data.data_clean.timestamps)
    adc_data = np.asarray(measurement_data.data_clean.adc_data)
    if adc_data.ndim != 2 or adc_data.shape[1] != ADC_COUNT:
        raise ValueError(
            f"Expected clean ADC data with {ADC_COUNT} channels; got shape {adc_data.shape}"
        )
    if len(timestamps) != len(adc_data):
        raise ValueError("Clean timestamps and ADC samples have different lengths")

    adc_signals = [adc_data[:, adc] for adc in range(ADC_COUNT)]
    minima, maxima = detect_breath_peaks(adc_signals[TARGET_ADC])
    return { "minima": minima, "maxima": maxima, "timestamps": timestamps, "adc_signals": adc_signals }

def calculate_breath_characteristics(people_files, file_cache):
    """
    Calculates breath characteristics (depth, length, start_idx, end_idx, inhale duration, exhale duration)
    for each breath in the given files.

    Parameters
    ----------
    people_files : list of str
        Identifiers for the clean measurements containing one person's data.
    file_cache : dict
        Parsed clean measurements indexed by their identifiers.

    Returns
    -------
    all_breath_data : list of dict
        A list of dictionaries, where each dictionary contains the features of a single breath extracted from the given files.

    Side Effects
    ------------
    This function has no side effects.
    """

    all_breath_data = []
    for file in people_files:
        file_breath_data = file_cache[file]
        minima = file_breath_data["minima"]
        maxima = file_breath_data["maxima"]
        timestamps = file_breath_data["timestamps"]
        adc_signals = file_breath_data["adc_signals"]

        for i in range(len(minima)-1):
            start_idx = minima[i]
            end_idx = minima[i+1]
            breath_signal = adc_signals[TARGET_ADC][start_idx:end_idx]
            breath_timestamps = timestamps[start_idx:end_idx]
            if len(breath_signal) == 0:
                continue

            breath_features = extract_breath_features(breath_signal)
            breath_features['file_name'] = file
            breath_features['start_idx'] = start_idx
            breath_features['end_idx'] = end_idx
            breath_features['signal'] = breath_signal
            breath_features['timestamps'] = breath_timestamps

            # Find all maxima within the breath
            peaks_in_breath = []
            for peak in maxima:
                if start_idx <= peak < end_idx:
                    peaks_in_breath.append(peak)
                    break

            # If there is more than one peak we want to take the "highest" one
            peak_in_breath = np.max(peaks_in_breath) if peaks_in_breath else None

            if peak_in_breath is not None:
                phases = calculate_breathing_phases_for_breath(adc_signals[TARGET_ADC], timestamps, peak_in_breath, start_idx, end_idx)
                breath_features['inhale'] = phases[0]
                breath_features['inspiratory_pause'] = phases[1]
                breath_features['exhale'] = phases[2]
                breath_features['expiratory_pause'] = phases[3]

                all_breath_data.append(breath_features)  
    return all_breath_data          

def get_mode_param(all_breath_data, param):
    """ 
    Calculates the mode value for a given breath parameter (e.g., depth, length, inhale duration) from the breath dataset.

    Parameters
    ---------- 
    all_breath_data : list of dict
        A list of dictionaries, where each dictionary contains the features of a single breath.
    param : str
        The name of the breath parameter for which to calculate the mode (e.g., 'depth', 'length', 'inhale', 'exhale').

    Returns
    -------
    mode_value : float
        The mode value for the specified breath parameter, calculated using a histogram-based approach.

    Side Effects
    ------------
    This function has no side effects.
    """

    values = [breath[param] for breath in all_breath_data]
    non_zero_values = [v for v in values if v != 0]
    if len(non_zero_values) == 0:
        return 0.0
    
    hist, bin_edges = np.histogram(non_zero_values, bins=50)
    max_count = np.max(hist)
    max_bins = np.where(hist == max_count)[0]

    if len(max_bins) > 1:
        mean_value = np.mean(values)
        closest_bin = min(max_bins, key=lambda b: abs((bin_edges[b] + bin_edges[b + 1]) / 2 - mean_value))
        mode_upper_bin = closest_bin
    else:
        mode_upper_bin = np.argmax(hist)
        
    mode_depth = (bin_edges[mode_upper_bin] + bin_edges[mode_upper_bin + 1]) / 2
    return mode_depth

def calculate_mode_and_avg_features(all_breath_data):
    """
    Calculates the average and mode values for breath characteristics (depth, length, inhale duration,
    exhale duration) from the breath dataset.

    Parameters
    ----------
    all_breath_data : list of dict
        A list of dictionaries, where each dictionary contains the features of a single breath.

    Returns
    -------
    avg_values : dict
        A dictionary containing the average values for breath characteristics (depth, length, inhale duration,
        exhale duration).
    mode_values : dict
        A dictionary containing the mode values for breath characteristics (depth, length, inhale duration,
        exhale duration).
    
    Side Effects
    ------------
    This function has no side effects.
    """

    rounding_precision = 10
    avg_values = {
        "avg_depth": round(np.mean([breath['depth'] for breath in all_breath_data]), rounding_precision),
        "avg_length": round(np.mean([breath['length'] for breath in all_breath_data]), rounding_precision),
        "avg_inhale": round(np.mean([breath['inhale'] for breath in all_breath_data]), rounding_precision),
        "avg_inspiratory_pause": round(np.mean([breath['inspiratory_pause'] for breath in all_breath_data]), rounding_precision),
        "avg_exhale": round(np.mean([breath['exhale'] for breath in all_breath_data]), rounding_precision),
        "avg_expiratory_pause": round(np.mean([breath['expiratory_pause'] for breath in all_breath_data]), rounding_precision),
    }

    mode_values = {
        "mode_depth": round(get_mode_param(all_breath_data, 'depth'), rounding_precision),
        "mode_length": round(get_mode_param(all_breath_data, 'length'), rounding_precision),
        "mode_inhale": round(get_mode_param(all_breath_data, 'inhale'), rounding_precision),
        "mode_inspiratory_pause": round(get_mode_param(all_breath_data, 'inspiratory_pause'), rounding_precision),
        "mode_exhale": round(get_mode_param(all_breath_data, 'exhale'), rounding_precision),
        "mode_expiratory_pause": round(get_mode_param(all_breath_data, 'expiratory_pause'), rounding_precision),
    }
    return avg_values, mode_values

def create_data_profiles():
    """
    Creates data profiles for each person from cleaned measurements in the dataset ZIP.

    Parameters
    ----------
    None

    Returns
    -------
    people_profiles : dict
        A dictionary where each key is a person's identifier (e.g., initials) and the value is another dictionary containing:
        - avg_values: A dictionary of average breath characteristics (depth, length, inhale duration, exhale duration).
        - mode_values: A dictionary of mode breath characteristics (depth, length, inhale duration, exhale duration).
        - mode_breaths_weights: A list of weights corresponding to the selected mode breaths.
        - mode_breaths: A list of dictionaries containing the signal, timestamps and features of mode breaths for that person.

    Side Effects
    ------------
    This function has no side effects.
    """
    dataset_hook = MeasurementDatasetHook(target="clean")
    files = list(dataset_hook)
    if not files:
        raise FileNotFoundError("The measurement dataset ZIP contains no clean measurements")

    people_files = {}
    file_cache = {}
    people_profiles = dict()
    for filepath in files:
        measurement_data = MeasurementData()
        measurement_data_builder = MeasurementDataBuilder(measurement_data)
        measurement_data_builder.build_data(filepath=filepath, target="clean")

        person = measurement_data.metadata.labels.person_data.person_id
        if not person:
            raise ValueError(f"Clean measurement is missing a person ID: {filepath.name}")
        file = str(filepath)
        if person not in people_files:
            people_files[person] = []
        people_files[person].append(file)
        file_cache[file] = extract_breath_data_from_measurement(measurement_data)

    for person in people_files:
        people_profiles[person] = {}

        target_adc_all_breath_data = calculate_breath_characteristics(
            people_files[person],
            file_cache,
        )
        avg_values, mode_values = calculate_mode_and_avg_features(target_adc_all_breath_data)   

        mode_breaths, mode_breaths_weights = get_mode_breaths(
            target_adc_all_breath_data,
            mode_values,
            avg_values,
            file_cache,
        )

        people_profiles[person] = {
            "avg_values": avg_values,
            "mode_values": mode_values,
            "mode_breaths_weights": mode_breaths_weights,
            "mode_breaths": mode_breaths
        }
    
    return people_profiles

def plot_mode_breath_on_file(mode_breath, target_adc = TARGET_ADC):
    file_name = pathlib.Path(mode_breath['file_name']).name
    timestamps = mode_breath['file_timestamps']
    adc_signals = mode_breath['file_signals']
    mode_timestamps = mode_breath['timestamps']
    mode_signal = mode_breath['signal']

    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, adc_signals[target_adc], label='Full Signal', color='blue', linewidth=1)
    plt.plot(mode_timestamps, mode_signal, label='Mode Breath', color='red', linewidth=2.5, alpha=0.8)
    plt.title(f'Mode Breath Highlighted on File: {file_name}')
    plt.xlabel('Time (ms)')
    plt.ylabel('ADC Voltage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f'./results/mode_breath_{file_name.replace(".jsonl", ".png")}')
    plt.show()

def group_plot_mode_breaths(people_profiles):
    retimed_breaths_for_people = {}

    for person in people_profiles:
        mode_breaths_target_adc = people_profiles[person]["mode_breaths"][TARGET_ADC]
        if mode_breaths_target_adc is None:
            continue

        mode_breath = mode_breaths_target_adc[0]

        signal = mode_breath['signal']
        timestamps = mode_breath['timestamps']
        timestamps = [t - timestamps[0] for t in timestamps]
        retimed_breaths_for_people[person] = (timestamps, signal)

    plt.figure(figsize=(14, 7))
    for person, (timestamps, signal) in retimed_breaths_for_people.items():
        plt.plot(timestamps, signal, label=f'Mode Breath - {person}', linewidth=2.0, alpha=0.8)
    plt.title('Mode Breaths for All People')
    plt.xlabel('Time (ms)')
    plt.ylabel('ADC Voltage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f'./results/mode_breaths_all_people.png')
    plt.show()

def plot_profiles(profiles):
    """
    Creates a plot for data from features displaying an approximated breath
    profile calculated from features.

    Parameters
    ----------
    profiles : a dictionary where the keys are people tags and the value is a dictionary with
    the mode values shown below
        - "bpm_mode"
        - "breath_depth_mode"
        - "inhale_length_mode"
        - "ip_length_mode"
        - "exhale_length_mode"
        - "ep_length_mode"

    Returns
    -------
        None

    Side Effects
    ------------
        This function has no side effects.
    """
    length = len(profiles)
    fig, ax = plt.subplots(length, 1, squeeze=False)
    ax = ax[:, 0]
    i = 0
    no_of_points = 10
    alpha = 0.3
    total_length = 0
    biggest_height = 0
    for key, values in profiles.items():
        biggest_height = max(biggest_height, values["breath_depth_mode"])
        total_length = max(total_length, (values["inhale_length_mode"]+values["ip_length_mode"]+values["exhale_length_mode"]+values["ep_length_mode"])/1000)
    for key, values in profiles.items():
        bps = 1/(values["bpm_mode"]/60)
        inhale_x = np.linspace(0, values["inhale_length_mode"]/1000, no_of_points)
        inhale_y = [values["breath_depth_mode"] * math.sin(math.pi * 2 * inhale_x[x] / (4*values["inhale_length_mode"]/1000)) for x in range(no_of_points)]
        ax[i].plot(inhale_x, inhale_y)
        ax[i].add_patch(patches.Rectangle(
        (0.0, 0.0),   # (x,y)
        values["inhale_length_mode"]/1000,          # width
        values["breath_depth_mode"],          # height
        color = "blue",
        alpha = alpha)) #transparency

        ip_start = values["inhale_length_mode"]/1000
        ip_x = np.linspace(ip_start, ip_start+values["ip_length_mode"]/1000, no_of_points)
        ip_y = [inhale_y[-1] for _ in ip_x]
        ax[i].plot(ip_x, ip_y)
        ax[i].add_patch(patches.Rectangle(
        (ip_x[0], 0.0),   # (x,y)
        values["ip_length_mode"]/1000,          # width
        values["breath_depth_mode"],          # height
        color = "orange",
        alpha = alpha)) #transparency

        exhale_start = ip_start+values["ip_length_mode"]/1000
        exhale_x = np.linspace(0, values["exhale_length_mode"]/1000, no_of_points)
        exhale_y = [values["breath_depth_mode"] * math.cos(math.pi * 2 * exhale_x[x] / (4*values["exhale_length_mode"]/1000)) for x in range(no_of_points)]
        exhale_x += exhale_start
        ax[i].add_patch(patches.Rectangle(
        (exhale_x[0], 0.0),   # (x,y)
        values["exhale_length_mode"]/1000,          # width
        values["breath_depth_mode"],          # height
        color = "red",
        alpha = alpha)) #transparency

        ep_start = exhale_x[-1]
        ep_x = np.linspace(ep_start, ep_start+values["ep_length_mode"]/1000, no_of_points)
        ep_y = [exhale_y[-1] for _ in ep_x]
        ax[i].plot(ep_x, ep_y)
        ax[i].add_patch(patches.Rectangle(
        (ep_x[0], 0.0),   # (x,y)
        values["ep_length_mode"]/1000,          # width
        values["breath_depth_mode"],          # height
        color = "green",
        alpha = alpha)) #transparency

        ax[i].plot(exhale_x, exhale_y)
        ax[i].set_xlabel("time [s]")
        ax[i].set_xlim(xmin=0.0, xmax=total_length)
        ax[i].set_ylabel("breath_depth")
        ax[i].set_ylim(ymin=0.0, ymax=biggest_height)
        ax[i].set_title(key)
        i += 1
    fig.tight_layout(pad=0.5)
    plt.show()

def create_profile_from_features(plot_enabled=False):
    """
    This function creates a dictionary with appropriate mode data in a dictionary.
    It then calls to plot the data

    Parameters
    ----------
        None
    
    Returns
    -------
        None
    
    Side Effects
    ------------
        This function has no side effects.
    """
    feature_data = FeatureData()
    feature_data.features = load_features_from_database_zip(feature_data)
    # print(feature_data.person_indices)
    people_profiles = {}
    for person in feature_data.person_initials:
        records = feature_data.features[feature_data.person_indices[person]]
        # print(records[0])
        dicts = []
        for i in range(len(feature_data.person_indices[person])):
            dicts.append({
                "bpm": records[i][0],
                "breath_depth": records[i][1],
                "inhale_length": records[i][13],
                "ip_length": records[i][14],
                "exhale_length": records[i][15],
                "ep_length": records[i][16]
            })
        
        people_profiles[person] = {
            "bpm_mode": get_mode_param(dicts, "bpm"),
            "breath_depth_mode": get_mode_param(dicts, "breath_depth"),
            "inhale_length_mode": get_mode_param(dicts, "inhale_length"),
            "ip_length_mode": get_mode_param(dicts, "ip_length"),
            "exhale_length_mode": get_mode_param(dicts, "exhale_length"),
            "ep_length_mode": get_mode_param(dicts, "ep_length")
        }
    
    if plot_enabled:
        plot_profiles(profiles=people_profiles)

def _build_smooth_breathing_signal(profile):
    mode_breaths = profile["mode_breaths"]
    weights = profile["mode_breaths_weights"]
    if len(mode_breaths) != ADC_COUNT:
        raise ValueError(f"Expected {ADC_COUNT} ADC channels, got {len(mode_breaths)}")

    breath_count = len(mode_breaths[TARGET_ADC])
    if breath_count == 0 or any(len(channel) != breath_count for channel in mode_breaths):
        raise ValueError("All ADC channels must contain the same nonzero number of mode breaths")
    if len(weights) != breath_count or sum(weights) <= 0:
        raise ValueError("Mode-breath weights must match the available breaths and be positive")

    def middle_line(signal):
        lower, upper = np.percentile(signal, [5, 95])
        return (lower + upper) / 2

    middle_lines = []
    for adc in range(ADC_COUNT):
        channel_midlines = [
            middle_line(np.asarray(breath["signal"], dtype=float))
            for breath in mode_breaths[adc]
        ]
        middle_lines.append(float(np.median(channel_midlines)))

    first_timestamps = np.asarray(mode_breaths[TARGET_ADC][0]["timestamps"], dtype=float)
    if len(first_timestamps) < 2:
        raise ValueError("Mode breaths must contain at least two timestamped samples")
    sample_intervals = np.diff(first_timestamps)
    if np.any(sample_intervals <= 0):
        raise ValueError("Mode-breath timestamps must be strictly increasing")
    sample_interval = float(np.median(sample_intervals))

    breath_durations = [
        float(breath["timestamps"][-1] - breath["timestamps"][0])
        for breath in mode_breaths[TARGET_ADC]
    ]
    mean_breath_duration = float(np.mean(breath_durations))
    if mean_breath_duration <= 0:
        raise ValueError("Mode-breath durations must be positive")
    target_duration = SIM_MINUTES_COUNT * 60 * 1000
    breath_repetitions = max(1, math.floor(target_duration / mean_breath_duration))
    blend_samples = max(2, round(500 / sample_interval))

    generated_timestamps = []
    generated_signal = [[] for _ in range(ADC_COUNT)]
    previous_end_values = [None] * ADC_COUNT
    join_indices = []
    total_time = 0.0

    for _ in range(breath_repetitions):
        breath_index = random.choices(range(breath_count), weights=weights, k=1)[0]
        target_breath = mode_breaths[TARGET_ADC][breath_index]
        timestamps = np.asarray(target_breath["timestamps"], dtype=float)
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Mode-breath timestamps must be strictly increasing")

        if generated_timestamps:
            join_indices.append(len(generated_timestamps))
        relative_timestamps = timestamps - timestamps[0]
        generated_timestamps.extend((relative_timestamps + total_time).tolist())
        total_time += float(relative_timestamps[-1]) + sample_interval

        for adc in range(ADC_COUNT):
            signal = np.asarray(mode_breaths[adc][breath_index]["signal"], dtype=float)
            if len(signal) != len(timestamps):
                raise ValueError("All ADC channels in a mode breath must have equal sample counts")

            signal = signal - middle_line(signal) + middle_lines[adc]
            previous_end = previous_end_values[adc]
            if previous_end is not None:
                blend_count = min(blend_samples, len(signal))
                blend_phase = np.linspace(0.0, 1.0, blend_count)
                blend_weights = 0.5 * (1.0 + np.cos(np.pi * blend_phase))
                signal[:blend_count] += (previous_end - signal[0]) * blend_weights

            generated_signal[adc].extend(signal.tolist())
            previous_end_values[adc] = float(signal[-1])

    return (
        np.asarray(generated_timestamps),
        [np.asarray(signal) for signal in generated_signal],
        join_indices,
    )


def generate_breathing_signals(profiles):
    """
    Generates breathing signal for each person in the profiles dictionary by stacking randomly
    (with weights) selected breaths from the person's mode breaths. For now the generated signal
    is not saved anywhere.

    Parameters
    ----------
    profiles : dict
        A dictionary where each key is a person's identifier (e.g., initials) and the value is another dictionary containing:
        - avg_values: A dictionary of average breath characteristics (depth, length, inhale duration, exhale duration).
        - mode_values: A dictionary of mode breath characteristics (depth, length, inhale duration, exhale duration).
        - mode_breaths_weights: A list of weights corresponding to the selected mode breaths.
        - mode_breaths: A list of dictionaries containing the signal, timestamps and features of mode breaths for that person.

    Returns
    -------
    None

    Side Effects
    ------------
    This function has no side effects.
    """

    for person, profile in profiles.items():
        generated_timestamps, generated_signal, _ = _build_smooth_breathing_signal(profile)
        plt.figure(figsize=(14, 7))
        for adc in range(ADC_COUNT):
            plt.plot(generated_timestamps, generated_signal[adc], label=f'Generated Breathing Signal - ADC {adc}', alpha=0.5)
        plt.legend()
        plt.title(f"breathing signal gen - {person}")
        plt.show()
        plt.close()

def create_breathing_animation(plot_enabled: bool):
    pass
    # load clean files
    # ---------------
    # create separate subplots for each person
    # ---------------
    # animate them according to data

def main():
    gen = input("Signal generation: 0/1\n")
    plot = input("Plotting: 0/1\n")
    # plot = "0"
    # gen = "1"
    plot_enabled = plot == "1"
    gen_enabled = gen == "1"

    people_profiles = create_data_profiles()

    if plot_enabled:
        plt.figure(figsize=(14, 7))
        for person in people_profiles:
            random_color = np.random.rand(3)
            # target_adc_breaths = people_profiles[person]["mode_breaths"][TARGET_ADC]
            for breath in people_profiles[person]["mode_breaths"][TARGET_ADC]:
                first_timestamp = breath['timestamps'][0]
                temp_timestamps = [t - first_timestamp for t in breath['timestamps']]
                plt.plot(temp_timestamps, breath['signal'], alpha=0.8, color=random_color)
            plt.title(f'Mode Breaths - {person}')
            plt.xlabel('Time (ms)')
            plt.ylabel('ADC Voltage')
        plt.show()

        for person in people_profiles:
            if "s194" in person:
                print(f"Plotting mode breath for {person}")
                plot_mode_breath_on_file(people_profiles[person]["mode_breaths"][TARGET_ADC][0], TARGET_ADC)
        # group_plot_mode_breaths(people_profiles)

    if gen_enabled:
        generate_breathing_signals(people_profiles)

    # create_profile_from_features(plot_enabled)
    # create_breathing_animation(plot_enabled)

if __name__ == "__main__":
    main()