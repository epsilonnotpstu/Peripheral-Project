/**
 * ECGChart — Real-time scrolling ECG waveform using Chart.js 4.
 *
 * Key performance decisions:
 *   - animation: false      → no transition overhead, mandatory for 125 Hz updates
 *   - update('none')        → skip animation frame entirely
 *   - tension: 0            → no bezier smoothing — preserves R-wave morphology
 *   - pointRadius: 0        → no dot per sample — critical at high sample rate
 *   - Rolling buffer with shift/push instead of splice — O(1) per sample
 */

class ECGChart {
  /**
   * @param {string} canvasId  - ID of the <canvas> element
   * @param {number} fs        - Sampling rate in Hz (default 125)
   * @param {number} windowSeconds - Visible time window (default 5 seconds)
   */
  constructor(canvasId, fs = 125, windowSeconds = 5) {
    this.fs = fs;
    this.windowSamples = windowSeconds * fs;

    // Pre-allocate rolling buffer filled with null (renders as gap)
    this._buffer = new Array(this.windowSamples).fill(null);

    // X-axis labels (sample indices — not shown, just spacers)
    this._labels = Array.from({ length: this.windowSamples }, (_, i) => i);

    const ctx = document.getElementById(canvasId);
    if (!ctx) {
      console.error(`ECGChart: canvas element "${canvasId}" not found`);
      return;
    }

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this._labels,
        datasets: [{
          label: 'ECG',
          data: this._buffer,
          borderColor: '#00e676',       // medical green on dark background
          borderWidth: 1.5,
          pointRadius: 0,               // no dots — critical for performance at 125 Hz
          tension: 0,                   // no bezier — preserve morphology
          fill: false,
          spanGaps: false,              // don't connect across null gaps
        }],
      },
      options: {
        animation: false,               // CRITICAL: zero animation overhead
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },  // tooltips add render overhead
        },
        scales: {
          x: {
            display: false,             // hide X axis labels (not meaningful as sample indices)
            grid: { display: false },
          },
          y: {
            min: -0.05,
            max: 1.15,
            grid: {
              color: 'rgba(41,121,255,0.08)',
              lineWidth: 0.5,
            },
            ticks: {
              color: '#4a6080',
              maxTicksLimit: 5,
              font: { size: 10 },
            },
            border: { color: '#1e2d4a' },
          },
        },
      },
    });
  }

  /**
   * Add a single sample to the rolling buffer and update the chart.
   * Called up to 12.5 times/second (every EMIT_CHUNK_SIZE=10 samples).
   * @param {number} value - ECG sample value [0, 1]
   */
  addSample(value) {
    this._buffer.shift();
    this._buffer.push(typeof value === 'number' ? value : null);
    this.chart.data.datasets[0].data = this._buffer;
    this.chart.update('none');   // 'none' = skip transition, fastest possible update
  }

  /**
   * Add multiple samples at once (when receiving batched chunks from socket).
   * @param {number[]} samples - Array of ECG sample values
   */
  addSamples(samples) {
    const n = samples.length;
    if (n >= this.windowSamples) {
      // Replace entire buffer (unlikely but safe)
      this._buffer = samples.slice(-this.windowSamples).map(v => v ?? null);
    } else {
      // Shift out old samples, push new ones
      this._buffer.splice(0, n);
      for (const v of samples) {
        this._buffer.push(typeof v === 'number' ? v : null);
      }
    }
    this.chart.data.datasets[0].data = this._buffer;
    this.chart.update('none');
  }

  /**
   * Change the visible time window.
   * @param {number} seconds - New window size in seconds
   */
  setWindow(seconds) {
    this.windowSamples = seconds * this.fs;
    this._buffer = new Array(this.windowSamples).fill(null);
    this._labels = Array.from({ length: this.windowSamples }, (_, i) => i);
    this.chart.data.labels = this._labels;
    this.chart.data.datasets[0].data = this._buffer;
    this.chart.update('none');
  }

  /** Clear the waveform (e.g., on disconnect). */
  clear() {
    this._buffer.fill(null);
    this.chart.data.datasets[0].data = this._buffer;
    this.chart.update('none');
  }

  /** Destroy chart instance (cleanup). */
  destroy() {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }
}
