/**
 * ECGSocketClient — Manages Flask-SocketIO connection on the /ecg namespace.
 *
 * Connects directly via WebSocket (skips polling handshake for lower latency).
 * All server→client events are routed to the ECGChart and UIUpdater instances.
 */

class ECGSocketClient {
  /**
   * @param {ECGChart} ecgChart - The chart instance to receive samples
   * @param {UIUpdater} uiUpdater - The UI controller
   */
  constructor(ecgChart, uiUpdater) {
    this.chart = ecgChart;
    this.ui = uiUpdater;

    this.socket = io('/ecg', {
      transports: ['websocket'],       // skip polling — go straight to WS
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 20,
      timeout: 10000,
    });

    this._bindEvents();
  }

  // ── Server → Client events ────────────────────────────────────────────────

  _bindEvents() {
    this.socket.on('connect', () => {
      console.log('[SocketIO] Connected to /ecg namespace');
      this.ui.setConnectionStatus(true);
      this.socket.emit('get_status', {});
    });

    this.socket.on('disconnect', (reason) => {
      console.warn('[SocketIO] Disconnected:', reason);
      this.ui.setConnectionStatus(false);
      this.chart.clear();
    });

    this.socket.on('connect_error', (err) => {
      console.error('[SocketIO] Connection error:', err.message);
      this.ui.setConnectionStatus(false);
    });

    // Real-time ECG waveform samples (batch of EMIT_CHUNK_SIZE samples)
    this.socket.on('ecg_chunk', (data) => {
      // data: {samples: [float,...], bpm: float, ts: int, session_id: int}
      if (Array.isArray(data.samples)) {
        this.chart.addSamples(data.samples);
      } else if (typeof data.sample === 'number') {
        this.chart.addSample(data.sample);
      }
      if (data.bpm) {
        this.ui.updateBPM(data.bpm);
      }
    });

    // Per-beat AI classification result
    this.socket.on('beat_classified', (data) => {
      /*
        data: {
          class_id, class_name, short_name, confidence, probabilities,
          alert, bpm, beat_index, motion_flag, session_id, timestamp, inference_ms
        }
      */
      this.ui.updateClassification(data);
      this.ui.addBeatHistory(data);
      this.ui.incrementBeatCounter();

      if (data.bpm) {
        this.ui.updateBPM(data.bpm);
      }
      if (data.alert) {
        this.ui.showAlert(data.class_name, data.confidence);
      }
    });

    // Motion artifact status (from MPU6050)
    this.socket.on('motion_alert', (data) => {
      this.ui.updateMotionFlag(data.motion_flag, data.level || 0);
    });

    // System state sync (recording on/off, model loaded, etc.)
    this.socket.on('system_status', (data) => {
      this.ui.syncSystemStatus(data);
    });

    // Recording lifecycle events
    this.socket.on('recording_started', (data) => {
      this.ui.onRecordingStarted(data);
    });

    this.socket.on('recording_stopped', (data) => {
      this.ui.onRecordingStopped(data);
    });

    // Server-side error
    this.socket.on('error_msg', (data) => {
      console.error('[Server error]', data.message);
      this.ui.showToast('Error', data.message, 'danger');
    });
  }

  // ── Client → Server events ─────────────────────────────────────────────────

  startRecording(patientId, notes = '') {
    this.socket.emit('start_recording', {
      patient_id: parseInt(patientId),
      notes: notes || '',
    });
  }

  stopRecording() {
    this.socket.emit('stop_recording', {});
  }

  ping() {
    this.socket.emit('ping', {});
  }

  isConnected() {
    return this.socket && this.socket.connected;
  }
}
