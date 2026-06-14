/**
 * dashboard.js — Main UI controller and page initialization.
 *
 * Responsibilities:
 *   - UIUpdater: updates all DOM elements in response to socket events
 *   - Patient form: save new or select existing patient via REST API
 *   - Start/Stop button: wires to socket client
 *   - Session timer: client-side elapsed time counter
 *   - Beat history: last 20 beats displayed as color-coded badges
 *   - Session table: refreshable via API
 *   - PDF download button: links to /api/v1/sessions/<id>/report
 */

// ── UIUpdater class ───────────────────────────────────────────────────────────

class UIUpdater {
  constructor() {
    this._beatHistory = [];
    this._maxHistory = 20;
  }

  /** Update the BPM display with color coding. */
  updateBPM(bpm) {
    const el = document.getElementById('bpm-value');
    if (!el || !bpm || bpm <= 0) return;
    el.textContent = Math.round(bpm);

    // Color coding: green 60–100, yellow <60 or >100, red <40 or >150
    el.className = 'stat-value';
    const cfg = window.APP_CONFIG || {};
    if (bpm >= (cfg.bpmGreenMin || 60) && bpm <= (cfg.bpmGreenMax || 100)) {
      el.classList.add('bpm-normal');
    } else if (bpm < 40 || bpm > 150) {
      el.classList.add('bpm-danger');
    } else {
      el.classList.add('bpm-warning');
    }
  }

  /** Update classification badge, confidence bar, and probability bars. */
  updateClassification(data) {
    const { class_name: className, confidence, probabilities } = data;
    if (!className) return;

    // Class badge
    const badge = document.getElementById('class-badge');
    const fullName = document.getElementById('class-full-name');
    if (badge) {
      badge.textContent = className;
      badge.className = `class-badge class-${className}`;
    }
    if (fullName) {
      fullName.textContent = data.short_name ? `(${data.short_name})` : '';
    }

    // Confidence %
    const confEl = document.getElementById('confidence-value');
    if (confEl) {
      confEl.textContent = confidence ? `${(confidence * 100).toFixed(1)}%` : '--%';
    }

    // Probability mini-bars
    const shorts = ['N', 'S', 'V', 'F', 'Q'];
    if (Array.isArray(probabilities)) {
      shorts.forEach((s, i) => {
        const bar = document.getElementById(`prob-bar-${s}`);
        if (bar) {
          bar.style.width = `${((probabilities[i] || 0) * 100).toFixed(1)}%`;
        }
      });
    }
  }

  /** Update motion artifact indicator. */
  updateMotionFlag(motionFlag, level = 0) {
    const icon = document.getElementById('motion-icon');
    const text = document.getElementById('motion-text');
    const bar  = document.getElementById('motion-level-bar');

    if (motionFlag) {
      if (icon) { icon.className = 'bi bi-person-walking fs-2 motion-active'; }
      if (text) { text.textContent = 'MOTION DETECTED'; text.className = 'stat-unit mt-1 motion-active'; }
    } else {
      if (icon) { icon.className = 'bi bi-person-standing fs-2 motion-stable'; }
      if (text) { text.textContent = 'Stable'; text.className = 'stat-unit mt-1 motion-stable'; }
    }
    if (bar) { bar.style.width = `${Math.min(100, level * 100).toFixed(0)}%`; }
  }

  /** Add one beat badge to the history strip. */
  addBeatHistory(data) {
    this._beatHistory.push(data);
    if (this._beatHistory.length > this._maxHistory) {
      this._beatHistory.shift();
    }
    this._renderBeatHistory();
  }

  _renderBeatHistory() {
    const container = document.getElementById('beat-history');
    if (!container) return;
    container.innerHTML = '';
    for (const beat of this._beatHistory) {
      const span = document.createElement('span');
      span.className = `class-badge class-${beat.class_name}`;
      span.style.cssText = 'font-size:0.65rem;padding:0.15rem 0.45rem;cursor:default';
      span.title = `${beat.class_name} (${(beat.confidence * 100).toFixed(1)}%) @ ${beat.bpm ? Math.round(beat.bpm) : '?'} BPM`;
      span.textContent = beat.short_name || beat.class_name[0];
      container.appendChild(span);
    }
  }

  /** Increment the beats-analyzed counter. */
  incrementBeatCounter() {
    const el = document.getElementById('beat-counter');
    if (!el) return;
    el.textContent = parseInt(el.textContent || '0') + 1;
  }

  /** Show a Bootstrap toast alert for clinical arrhythmia detection. */
  showAlert(className, confidence) {
    const msgEl = document.getElementById('alert-message');
    if (msgEl) {
      msgEl.textContent = `${className} beat detected (confidence: ${(confidence * 100).toFixed(1)}%). Please verify with a clinician.`;
    }
    const toastEl = document.getElementById('alert-toast');
    if (toastEl) {
      const toast = bootstrap.Toast.getOrCreateInstance(toastEl, { autohide: true, delay: 6000 });
      toast.show();
    }
  }

  /** Generic toast notification. */
  showToast(title, message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${title}: ${message}`);
  }

  /** Update WebSocket connection badge. */
  setConnectionStatus(connected) {
    const badge = document.getElementById('connection-badge');
    if (!badge) return;
    if (connected) {
      badge.className = 'badge bg-success';
      badge.innerHTML = '<i class="bi bi-wifi"></i> Connected';
    } else {
      badge.className = 'badge bg-danger';
      badge.innerHTML = '<i class="bi bi-wifi-off"></i> Disconnected';
    }
  }

  /** Sync full system state (e.g., after page refresh or reconnect). */
  syncSystemStatus(data) {
    const btn = document.getElementById('start-stop-btn');
    const statusText = document.getElementById('recording-status-text');
    const sessionInfo = document.getElementById('session-info-text');

    window.isRecording = data.recording || false;
    window.currentSessionId = data.session_id || null;

    if (data.recording) {
      if (btn) {
        btn.className = 'btn btn-danger';
        btn.querySelector('#start-stop-icon').className = 'bi bi-stop-fill me-1';
        btn.querySelector('#start-stop-text').textContent = 'Stop Recording';
        btn.disabled = false;
      }
      if (statusText) { statusText.textContent = 'Recording…'; statusText.style.color = '#f44336'; }
      if (sessionInfo && data.session_id) { sessionInfo.textContent = `Session #${data.session_id}`; }
    } else {
      if (btn) {
        btn.className = 'btn btn-success';
        btn.querySelector('#start-stop-icon').className = 'bi bi-play-fill me-1';
        btn.querySelector('#start-stop-text').textContent = 'Start Recording';
        btn.disabled = !window.currentPatientId;
      }
      if (statusText) { statusText.textContent = 'Stopped'; statusText.style.color = ''; }
    }

    // Update PDF download button
    const dlBtn = document.getElementById('download-report-btn');
    if (dlBtn) {
      dlBtn.disabled = !data.session_id || data.recording;
      if (data.session_id && !data.recording) {
        dlBtn.onclick = () => window.open(`/api/v1/sessions/${data.session_id}/report`, '_blank');
      }
    }
  }

  onRecordingStarted(data) {
    window.isRecording = true;
    window.currentSessionId = data.session_id;
    window.sessionStartTime = Date.now();

    // Reset counters
    const bc = document.getElementById('beat-counter');
    if (bc) bc.textContent = '0';
    const hist = document.getElementById('beat-history');
    if (hist) hist.innerHTML = '';
    this._beatHistory = [];

    this.syncSystemStatus({ recording: true, session_id: data.session_id });
    startSessionTimer();
  }

  onRecordingStopped(data) {
    window.isRecording = false;
    stopSessionTimer();
    this.syncSystemStatus({ recording: false, session_id: data.session_id });

    // Enable download button for finished session
    const dlBtn = document.getElementById('download-report-btn');
    if (dlBtn && data.session_id) {
      dlBtn.disabled = false;
      dlBtn.onclick = () => window.open(`/api/v1/sessions/${data.session_id}/report`, '_blank');
    }

    // Reload session history
    setTimeout(loadSessions, 1000);
  }
}


// ── Session timer ──────────────────────────────────────────────────────────────

let _timerInterval = null;

function startSessionTimer() {
  stopSessionTimer();
  window.sessionStartTime = window.sessionStartTime || Date.now();
  _timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - window.sessionStartTime) / 1000);
    const m = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const s = (elapsed % 60).toString().padStart(2, '0');
    const el = document.getElementById('session-timer');
    if (el) el.textContent = `${m}:${s}`;
  }, 1000);
}

function stopSessionTimer() {
  if (_timerInterval) {
    clearInterval(_timerInterval);
    _timerInterval = null;
  }
}


// ── Patient form ───────────────────────────────────────────────────────────────

function initPatientForm() {
  // Populate form from existing patient dropdown
  const existingSelect = document.getElementById('existing-patient-select');
  if (existingSelect) {
    existingSelect.addEventListener('change', function () {
      const opt = this.options[this.selectedIndex];
      if (opt.value) {
        document.getElementById('patient-name').value = opt.dataset.name || '';
        document.getElementById('patient-age').value = opt.dataset.age || '';
        document.getElementById('patient-gender').value = opt.dataset.gender || '';
        document.getElementById('patient-medical-id').value = opt.dataset.medical || '';
        window.currentPatientId = parseInt(opt.value);
        updateCurrentPatientDisplay(opt.dataset.name);
      } else {
        // Clear for new patient entry
        ['patient-name','patient-age','patient-medical-id','patient-notes'].forEach(id => {
          document.getElementById(id).value = '';
        });
        document.getElementById('patient-gender').value = '';
        window.currentPatientId = null;
      }
    });
  }

  // Save patient button
  const saveBtn = document.getElementById('save-patient-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      const name = document.getElementById('patient-name').value.trim();
      if (!name) {
        document.getElementById('patient-name').classList.add('is-invalid');
        return;
      }
      document.getElementById('patient-name').classList.remove('is-invalid');

      // If existing patient selected, just use that ID
      const existingId = window.currentPatientId;
      if (existingId && existingSelect && existingSelect.value) {
        updateCurrentPatientDisplay(name);
        bootstrap.Modal.getInstance(document.getElementById('patientModal')).hide();
        enableStartButton();
        return;
      }

      // Create new patient via REST API
      try {
        const resp = await fetch('/api/v1/patients', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name,
            age: document.getElementById('patient-age').value || null,
            gender: document.getElementById('patient-gender').value || null,
            medical_id: document.getElementById('patient-medical-id').value.trim() || null,
            notes: document.getElementById('patient-notes').value.trim() || null,
          }),
        });

        if (!resp.ok) {
          const err = await resp.json();
          alert(`Error: ${err.error || 'Could not save patient'}`);
          return;
        }

        const patient = await resp.json();
        window.currentPatientId = patient.id;
        updateCurrentPatientDisplay(patient.name);

        // Add to dropdown
        if (existingSelect) {
          const opt = new Option(patient.name, patient.id);
          opt.dataset.name = patient.name;
          existingSelect.add(opt);
          existingSelect.value = patient.id;
        }

        bootstrap.Modal.getInstance(document.getElementById('patientModal')).hide();
        enableStartButton();
      } catch (e) {
        alert('Network error saving patient: ' + e.message);
      }
    });
  }
}

function updateCurrentPatientDisplay(name) {
  const el = document.getElementById('current-patient-display');
  if (el) el.textContent = name || 'None selected';
}

function enableStartButton() {
  const btn = document.getElementById('start-stop-btn');
  if (btn && !window.isRecording) btn.disabled = false;
}


// ── Start/Stop button ─────────────────────────────────────────────────────────

function initStartStopButton() {
  const btn = document.getElementById('start-stop-btn');
  if (!btn) return;

  btn.addEventListener('click', () => {
    if (!window.socketClient) {
      alert('Not connected to server. Please refresh.');
      return;
    }

    if (window.isRecording) {
      window.socketClient.stopRecording();
    } else {
      if (!window.currentPatientId) {
        bootstrap.Modal.getOrCreateInstance(document.getElementById('patientModal')).show();
        return;
      }
      window.socketClient.startRecording(window.currentPatientId, '');
    }
  });
}


// ── Session history reload ─────────────────────────────────────────────────────

async function loadSessions() {
  try {
    const resp = await fetch('/api/v1/sessions?per_page=20');
    if (!resp.ok) return;
    const data = await resp.json();
    const tbody = document.getElementById('session-tbody');
    if (!tbody) return;

    if (!data.sessions || data.sessions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-3">No sessions yet</td></tr>';
      return;
    }

    const classMap = {
      'Normal': 'N', 'Supraventricular': 'S', 'Ventricular': 'V',
      'Fusion': 'F', 'Unknown': 'Q',
    };

    tbody.innerHTML = data.sessions.map(s => {
      const startedAt = s.started_at ? new Date(s.started_at) : null;
      const dateStr = startedAt ? `${(startedAt.getMonth()+1).toString().padStart(2,'0')}/${startedAt.getDate().toString().padStart(2,'0')} ${startedAt.getHours().toString().padStart(2,'0')}:${startedAt.getMinutes().toString().padStart(2,'0')}` : '';
      const dur = s.duration_s ? `${Math.floor(s.duration_s/60)}:${Math.floor(s.duration_s%60).toString().padStart(2,'0')}` : s.is_active ? '⏺ live' : '--';
      const dom = s.dominant_class || 'N/A';
      const short = classMap[dom] || dom[0] || '?';

      return `<tr>
        <td>${s.id}</td>
        <td>${escapeHtml(s.patient_name || 'N/A')}</td>
        <td>${dateStr}</td>
        <td>${dur}</td>
        <td>${s.total_beats || 0}</td>
        <td><span class="class-badge class-${escapeHtml(dom)}" style="font-size:.65rem;padding:.15rem .45rem">${short}</span></td>
        <td>
          ${!s.is_active ? `<a href="/api/v1/sessions/${s.id}/report" class="btn btn-sm btn-outline-info py-0 px-1" target="_blank" title="Download PDF"><i class="bi bi-file-earmark-pdf"></i></a>` : '<span class="text-muted" style="font-size:.7rem">Live</span>'}
        </td>
      </tr>`;
    }).join('');
  } catch (e) {
    console.error('Failed to load sessions:', e);
  }
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}


// ── Page initialization ────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Global state
  window.currentPatientId = null;
  window.currentSessionId = null;
  window.isRecording = false;
  window.sessionStartTime = null;

  // Initialize ECG chart
  const cfg = window.APP_CONFIG || {};
  window.ecgChart = new ECGChart('ecg-canvas', cfg.fs || 125, cfg.windowSeconds || 5);

  // Initialize UI updater
  window.uiUpdater = new UIUpdater();

  // Initialize SocketIO client
  window.socketClient = new ECGSocketClient(window.ecgChart, window.uiUpdater);

  // Initialize UI controls
  initPatientForm();
  initStartStopButton();

  // Load session history
  loadSessions();

  console.log('[ECG Monitor] Dashboard initialized');
});
