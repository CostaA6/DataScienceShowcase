import scipy.signal
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch, correlate, lfilter, butter, filtfilt
from numpy.fft import fft, fftfreq


def calculate_instantaneous_attributes(seismic):
    """Calcula Amplitud Instantánea y Frecuencia Instantánea.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).

    Returns:
        tuple: Una tupla que contiene:
            - instant_amp (np.ndarray): Matriz 2D de Amplitud Instantánea (muestras x trazas).
            - instant_freq (np.ndarray): Matriz 2D de Frecuencia Instantánea (muestras x trazas).
    """
    analytic_signal = scipy.signal.hilbert(seismic, axis=0)
    instant_amp = np.abs(analytic_signal)
    instant_phase = np.unwrap(np.angle(analytic_signal), axis=0)
    instant_freq = np.diff(instant_phase, axis=0)
    instant_freq = np.vstack([instant_freq[0, :], instant_freq])  # mantener tamaño original
    return instant_amp, instant_freq

def calculate_rms_amplitude(seismic, window=25):
    """Calcula la Amplitud RMS.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window (int): Número de muestras en la ventana deslizante para el cálculo.

    Returns:
        np.ndarray: Matriz 2D de Amplitud RMS (muestras x trazas).
    """
    return np.sqrt(uniform_filter1d(seismic**2, size=window, axis=0))

def calculate_local_variance(seismic, window=25):
    """Calcula la Varianza Local.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window (int): Número de muestras en la ventana deslizante para el cálculo.

    Returns:
        np.ndarray: Matriz 2D de Varianza Local (muestras x trazas).
    """
    mean = uniform_filter1d(seismic, size=window, axis=0)
    sq_diff = (seismic - mean) ** 2
    return uniform_filter1d(sq_diff, size=window, axis=0)

def sliding_window_attribute(arr, func, window_size):
    """Aplica una función a ventanas deslizantes a lo largo del tiempo.

    Args:
        arr (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        func (callable): Función a aplicar a cada ventana. Debe tomar un array 1D como entrada y devolver un escalar.
        window_size (int): Número de muestras en la ventana deslizante.

    Returns:
        np.ndarray: Matriz 2D del atributo calculado (muestras x trazas). Los bordes tendrán valores NaN.
    """
    half = window_size // 2
    result = np.full_like(arr, np.nan, dtype=np.float32)
    for i in range(arr.shape[1]):  # por traza
        for j in range(half, arr.shape[0] - half):
            segment = arr[j - half:j + half + 1, i]
            result[j, i] = func(segment)
    return result

def calculate_zero_cross_rate(seismic, window_size=25):
    """Calcula la Tasa de Cruces por Cero usando una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window_size (int): Número de muestras en la ventana deslizante para el cálculo.

    Returns:
        np.ndarray: Matriz 2D de Tasa de Cruces por Cero (muestras x trazas).
    """
    def _zero_cross_rate(x):
        return np.mean(np.diff(np.signbit(x)) != 0)
    return sliding_window_attribute(seismic, _zero_cross_rate, window_size)

def calculate_spectral_entropy(seismic, window_size=25, nperseg=64):
    """Calcula la Entropía Espectral usando una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window_size (int): Número de muestras en la ventana deslizante para el cálculo del espectro.
        nperseg (int): Longitud del segmento utilizado en la estimación del espectro de potencia (Welch).

    Returns:
        np.ndarray: Matriz 2D de Entropía Espectral (muestras x trazas).
    """
    def _spectral_entropy(x):
        f, Pxx = welch(x, nperseg=nperseg, nfft=nperseg)
        Pxx = Pxx / np.sum(Pxx + 1e-12)  # normalizar
        return -np.sum(Pxx * np.log2(Pxx + 1e-12))
    return sliding_window_attribute(seismic, _spectral_entropy, window_size)

def calculate_gradient_amplitude(seismic):
    """Calcula el Gradiente de Amplitud a lo largo del tiempo.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).

    Returns:
        np.ndarray: Matriz 2D del Gradiente de Amplitud (muestras x trazas).
    """
    return np.abs(np.diff(seismic, axis=0, prepend=seismic[[0], :]))

def calculate_gradient_phase(seismic):
    """Calcula el Gradiente de Fase Instantánea a lo largo del tiempo.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).

    Returns:
        np.ndarray: Matriz 2D del Gradiente de Fase Instantánea (muestras x trazas).
    """
    analytic_signal_local = scipy.signal.hilbert(seismic, axis=0)
    instant_phase_local = np.unwrap(np.angle(analytic_signal_local), axis=0)
    return np.abs(np.diff(instant_phase_local, axis=0, prepend=instant_phase_local[[0], :]))

def calculate_median_amplitude_window(seismic, window_size=25):
    """Calcula la Mediana de la Amplitud en una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window_size (int): Número de muestras en la ventana deslizante para el cálculo.

    Returns:
        np.ndarray: Matriz 2D de la Mediana de Amplitud (muestras x trazas).
    """
    output = np.zeros_like(seismic)
    half_window = window_size // 2
    for i in range(seismic.shape[0]):
        start = max(0, i - half_window)
        end = min(seismic.shape[0], i + half_window + 1)
        output[i, :] = np.median(np.abs(seismic[start:end, :]), axis=0)
    return output

def calculate_range_amplitude_window(seismic, window_size=25):
    """Calcula el Rango de Amplitud en una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window_size (int): Número de muestras en la ventana deslizante para el cálculo.

    Returns:
        np.ndarray: Matriz 2D del Rango de Amplitud (muestras x trazas).
    """
    output = np.zeros_like(seismic)
    half_window = window_size // 2
    for i in range(seismic.shape[0]):
        start = max(0, i - half_window)
        end = min(seismic.shape[0], i + half_window + 1)
        window_data = seismic[start:end, :]
        output[i, :] = np.max(window_data, axis=0) - np.min(window_data, axis=0)
    return output


def calculate_coherence(seismic, window_size=15, num_neighbors=2):
    """
    Calcula la coherencia sísmica utilizando la correlación cruzada normalizada
    entre trazas adyacentes dentro de una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window_size (int): Número de muestras en la ventana deslizante para el cálculo de la correlación.
        num_neighbors (int): Número de trazas vecinas a cada lado para comparar.

    Returns:
        np.ndarray: Matriz 2D de coherencia sísmica (muestras x trazas).
                      Los valores estarán entre -1 y 1, donde valores cercanos a 1
                      indican alta coherencia.
    """
    n_samples, n_traces = seismic.shape
    coherence = np.full_like(seismic, np.nan, dtype=np.float32)
    half_window = window_size // 2

    for i in range(n_traces):
        for j in range(half_window, n_samples - half_window):
            ref_trace_segment = seismic[j - half_window:j + half_window + 1, i]
            correlations = []

            # Comparar con las trazas vecinas
            for neighbor_offset in range(-num_neighbors, num_neighbors + 1):
                neighbor_index = i + neighbor_offset
                if 0 <= neighbor_index < n_traces and neighbor_index != i:
                    neighbor_trace_segment = seismic[j - half_window:j + half_window + 1, neighbor_index]

                    # Calcular la correlación cruzada normalizada (solo el valor máximo en el lag 0)
                    correlation = correlate(ref_trace_segment, neighbor_trace_segment, mode='same')
                    norm_factor = np.sqrt(np.sum(ref_trace_segment**2) * np.sum(neighbor_trace_segment**2)) + 1e-12
                    normalized_correlation = correlation[window_size // 2] / norm_factor
                    correlations.append(normalized_correlation)

            if correlations:
                coherence[j, i] = np.mean(correlations)

    return coherence


def calculate_trace_energy(seismic, window_size=25):
    """
    Calcula la energía de la traza sísmica dentro de una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window_size (int): Número de muestras en la ventana deslizante para el cálculo de la energía.

    Returns:
        np.ndarray: Matriz 2D de energía de la traza sísmica (muestras x trazas).
    """
    n_samples, n_traces = seismic.shape
    trace_energy = np.full_like(seismic, np.nan, dtype=np.float32)
    half_window = window_size // 2

    for i in range(n_traces):
        squared_trace = seismic[:, i] ** 2  # Elevar al cuadrado la traza actual
        energy_trace = uniform_filter1d(squared_trace, size=window_size, mode='constant', cval=0.0)
        trace_energy[:, i] = energy_trace

    return trace_energy


def predictive_deconvolution(seismic, prediction_distance=1, operator_length=20, filter_order=3, cutoff_freq=0.1):
    """
    Aplica deconvolución predictiva a la sísmica.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        prediction_distance (int): Distancia en muestras entre el punto predicho y el inicio del operador.
        operator_length (int): Longitud en muestras del operador de predicción.
        filter_order (int): Orden del filtro pasa-altas para estabilizar (opcional).
        cutoff_freq (float): Frecuencia de corte normalizada (0 a 1) del filtro pasa-altas (opcional).

    Returns:
        np.ndarray: Matriz 2D de sísmica deconvolucionada (muestras x trazas).
    """
    n_samples, n_traces = seismic.shape
    deconvolved_seismic = np.zeros_like(seismic, dtype=np.float32)

    for i in range(n_traces):
        trace = seismic[:, i].astype(np.float32)

        # Estabilización con filtro pasa-altas suave (opcional)
        if filter_order > 0 and 0 < cutoff_freq < 1:
            nyquist_freq = 0.5  # Frecuencia de Nyquist normalizada
            normalized_cutoff = cutoff_freq / nyquist_freq
            b, a = butter(filter_order, normalized_cutoff, btype='high', analog=False)
            trace = filtfilt(b, a, trace)

        # Calcular la autocorrelación
        autocorr = np.correlate(trace, trace, mode='full')
        autocorr = autocorr[len(trace) - 1:]  # Tomar la parte causal

        # Formar la matriz de Toeplitz
        toeplitz_matrix = np.zeros((operator_length, operator_length))
        for j in range(operator_length):
            toeplitz_matrix[j, :] = autocorr[prediction_distance + abs(j - np.arange(operator_length))]

        # Resolver para el operador de predicción (Wiener-Levinson)
        try:
            rhs = -autocorr[prediction_distance:prediction_distance + operator_length]
            prediction_operator = np.linalg.solve(toeplitz_matrix + np.eye(operator_length) * 1e-6, rhs) # Regularización
        except np.linalg.LinAlgError:
            prediction_operator = np.zeros(operator_length)
            print(f"Advertencia: Matriz singular en traza {i}, operador de predicción en ceros.")

        # Aplicar el operador de predicción (sustracción)
        predicted_trace = np.convolve(trace, np.concatenate(([0] * prediction_distance, prediction_operator)), mode='same')
        deconvolved_seismic[:, i] = trace - predicted_trace

    return deconvolved_seismic

def calculate_deconvolution_amplitude(deconvolved_seismic):
    """Calcula la Amplitud de la sísmica deconvolucionada.

    Args:
        deconvolved_seismic (np.ndarray): Matriz 2D de datos sísmicos deconvolucionados (muestras x trazas).

    Returns:
        np.ndarray: Matriz 2D de Amplitud de la sísmica deconvolucionada (muestras x trazas).
    """
    return np.abs(deconvolved_seismic)


def calculate_variance_of_difference(seismic, window_size=15, num_neighbors=2):
    """
    Calcula la varianza de la diferencia entre trazas sísmicas adyacentes
    dentro de una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window_size (int): Número de muestras en la ventana deslizante para el cálculo.
        num_neighbors (int): Número de trazas vecinas a cada lado para comparar.

    Returns:
        np.ndarray: Matriz 2D de varianza de la diferencia (muestras x trazas).
                      Valores bajos indican alta similitud entre trazas.
    """
    n_samples, n_traces = seismic.shape
    variance_of_difference = np.full_like(seismic, np.nan, dtype=np.float32)
    half_window = window_size // 2

    for i in range(n_traces):
        for j in range(half_window, n_samples - half_window):
            ref_trace_segment = seismic[j - half_window:j + half_window + 1, i]
            differences_squared = []

            # Comparar con las trazas vecinas
            for neighbor_offset in range(-num_neighbors, num_neighbors + 1):
                neighbor_index = i + neighbor_offset
                if 0 <= neighbor_index < n_traces and neighbor_index != i:
                    neighbor_trace_segment = seismic[j - half_window:j + half_window + 1, neighbor_index]
                    difference = ref_trace_segment - neighbor_trace_segment
                    differences_squared.append(difference ** 2)

            if differences_squared:
                variance_of_difference[j, i] = np.mean(differences_squared)

    return variance_of_difference


def calculate_max_cross_correlation(seismic, window_size=15, num_neighbors=2):
    """
    Calcula la correlación cruzada máxima entre una traza y sus vecinas
    dentro de una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        window_size (int): Número de muestras en la ventana deslizante para el cálculo de la correlación.
        num_neighbors (int): Número de trazas vecinas a cada lado para comparar.

    Returns:
        np.ndarray: Matriz 2D de correlación cruzada máxima (muestras x trazas).
                      Los valores estarán entre -1 y 1, donde valores cercanos a 1
                      indican alta similitud.
    """
    n_samples, n_traces = seismic.shape
    max_cross_correlation = np.full_like(seismic, np.nan, dtype=np.float32)
    half_window = window_size // 2

    for i in range(n_traces):
        for j in range(half_window, n_samples - half_window):
            ref_trace_segment = seismic[j - half_window:j + half_window + 1, i]
            max_correlations = []

            # Comparar con las trazas vecinas
            for neighbor_offset in range(-num_neighbors, num_neighbors + 1):
                neighbor_index = i + neighbor_offset
                if 0 <= neighbor_index < n_traces and neighbor_index != i:
                    neighbor_trace_segment = seismic[j - half_window:j + half_window + 1, neighbor_index]

                    # Calcular la correlación cruzada normalizada
                    correlation = correlate(ref_trace_segment, neighbor_trace_segment, mode='same')
                    norm_factor = np.sqrt(np.sum(ref_trace_segment**2) * np.sum(neighbor_trace_segment**2)) + 1e-12
                    normalized_correlation = correlation / norm_factor

                    # Tomar el valor absoluto máximo de la correlación normalizada
                    max_correlations.append(np.max(np.abs(normalized_correlation)))

            if max_correlations:
                max_cross_correlation[j, i] = np.mean(max_correlations)

    return max_cross_correlation


def calculate_dominant_bandwidth(seismic, sampling_rate, window_size=25, power_drop_db=3):
    """
    Calcula el ancho de banda dominante de la señal sísmica en una ventana deslizante.

    Args:
        seismic (np.ndarray): Matriz 2D de datos sísmicos (muestras x trazas).
        sampling_rate (float): Frecuencia de muestreo de los datos sísmicos en Hz.
        window_size (int): Número de muestras en la ventana deslizante para el cálculo del espectro.
        power_drop_db (float): Decibelios por debajo del pico para definir el ancho de banda.

    Returns:
        np.ndarray: Matriz 2D del ancho de banda dominante (Hz) (muestras x trazas).
    """
    n_samples, n_traces = seismic.shape
    dominant_bandwidth = np.full_like(seismic, np.nan, dtype=np.float32)
    half_window = window_size // 2

    for i in range(n_traces):
        for j in range(half_window, n_samples - half_window):
            segment = seismic[j - half_window:j + half_window + 1, i]
            N = len(segment)
            yf = fft(segment)
            xf = fftfreq(N, 1 / sampling_rate)[:N//2]
            power_spectrum = np.abs(yf[0:N//2])**2

            if np.sum(power_spectrum) > 0:
                peak_power = np.max(power_spectrum)
                peak_freq_index = np.argmax(power_spectrum)
                peak_freq = xf[peak_freq_index]

                power_threshold = peak_power / (10**(power_drop_db / 10))

                lower_freq = np.nan
                for k in range(peak_freq_index - 1, -1, -1):
                    if power_spectrum[k] < power_threshold:
                        lower_freq = xf[k]
                        break
                if np.isnan(lower_freq):
                    lower_freq = xf[0]

                upper_freq = np.nan
                for k in range(peak_freq_index + 1, len(xf)):
                    if power_spectrum[k] < power_threshold:
                        upper_freq = xf[k]
                        break
                if np.isnan(upper_freq):
                    upper_freq = xf[-1]

                dominant_bandwidth[j, i] = upper_freq - lower_freq

    return dominant_bandwidth