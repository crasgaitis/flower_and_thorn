from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from flask import Flask, render_template, Response

from utils import update_buffer, get_last_data, compute_band_powers, julia, create_custom_colormap
from pylsl import StreamInlet, resolve_byprop

app = Flask(__name__)

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot')
def plot():
    matplotlib.use('Agg')
    def generate_plot():
        
        # search for active LSL streams
        print('Looking for an EEG stream...')
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('Can\'t find EEG stream.')
        else:
            print('Found it!')
            print(streams)
            
        # set active EEG stream to inlet and apply time correction
        print("Start acquiring data")
        inlet = StreamInlet(streams[0], max_chunklen=12)
        eeg_time_correction = inlet.time_correction()

        # get the stream info
        info = inlet.info()
        fs = int(info.nominal_srate())

        # init raw EEG data buffer
        eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
        filter_state = None  # for use with the notch filter

        # number of epochs in "buffer_length"
        n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                    SHIFT_LENGTH + 1))

        # init the band power buffer (for plotting)
        # bands will be ordered: [delta, theta, alpha, beta]
        band_buffer = np.zeros((n_win_test, 4))

        print('Press Ctrl-C in the console to break the while loop.')
    

        while True:
            delta = []
            theta = []
            alpha = []
            beta = []
            
            for i in range(10):
                eeg_data, _ = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))

                ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

                eeg_buffer, filter_state = update_buffer(
                    eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)

                data_epoch = get_last_data(eeg_buffer,
                                                    EPOCH_LENGTH * fs)

                band_powers = compute_band_powers(data_epoch, fs)
                band_buffer, _ = update_buffer(band_buffer,
                                                        np.asarray([band_powers]))
                delta.append(band_powers[0])
                theta.append(band_powers[1])
                alpha.append(band_powers[2])
                beta.append(band_powers[3])

            num_points = len(delta)
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False).tolist()

            scale_factors = [0.8, 0.3, 0.25]

            plt.figure(figsize=(8, 8))

            stem_length = 2
            num_segments = 30

            # stem
            stem_x = np.zeros(num_segments)
            stem_y = np.linspace(0, -stem_length, num_segments)

            # add random noise to stem
            noise_amplitude = 0.05
            random_noise = noise_amplitude * np.random.randn(num_segments)

            stem_x += random_noise

            x1 = [np.cos(angle) * num for num, angle in zip(alpha, angles)]
            y1 = [np.sin(angle) * num for num, angle in zip(alpha, angles)]
            plt.fill(x1, y1, 'r', alpha=1)
            # plt.plot(x1, y1, 'ro-', label='Alpha')

            x2 = [np.cos(angle) * num * scale_factors[0] for num, angle in zip(delta, angles)]
            y2 = [np.sin(angle) * num * scale_factors[0] for num, angle in zip(delta, angles)]
            plt.fill(x2, y2, 'b', alpha=1)
            # plt.plot(x2, y2, 'bo-', label='Beta')

            x3 = [np.cos(angle) * num * scale_factors[1] for num, angle in zip(beta, angles)]
            y3 = [np.sin(angle) * num * scale_factors[1] for num, angle in zip(beta, angles)]
            plt.fill(x3, y3, 'g', alpha=1)
            # plt.plot(x3, y3, 'go-', label='Delta')

            x4 = [np.cos(angle) * num * scale_factors[2] for num, angle in zip(theta, angles)]
            y4 = [np.sin(angle) * num * scale_factors[2] for num, angle in zip(theta, angles)]
            plt.fill(x4, y4, 'y', alpha=1)
            # plt.plot(x4, y4, 'yo-', label='Theta')

            # connecting the last point to the first point for each list
            plt.plot(stem_x, stem_y, color='green', linewidth=2, zorder=0)
            # plt.plot([x1[-1], x1[0]], [y1[-1], y1[0]], 'r-', linewidth=1)
            # plt.plot([x2[-1], x2[0]], [y2[-1], y2[0]], 'b-', linewidth=1)
            # plt.plot([x3[-1], x3[0]], [y3[-1], y3[0]], 'g-', linewidth=1)
            # plt.plot([x4[-1], x4[0]], [y4[-1], y4[0]], 'y-', linewidth=1)
            
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            plt.xticks([])
            plt.yticks([])
            
            # plt.title('Flower Visualization')
            plt.axis('equal')
            
            # saving the plot as an image
            img_data = BytesIO()
            plt.savefig(img_data, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            img_data.seek(0)

            plt.clf()

            # returning the image data
            yield (b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + img_data.getvalue() + b'\r\n')

    return Response(generate_plot(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)