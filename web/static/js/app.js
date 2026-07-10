const state = {
  selectedRole: "signer",
  socket: null,
  apiConfig: null,
  snapshot: null,
};

const elements = {
  lobby: document.getElementById("lobby"),
  room: document.getElementById("room"),
  roleSignerBtn: document.getElementById("roleSignerBtn"),
  roleReceiverBtn: document.getElementById("roleReceiverBtn"),
  roomInput: document.getElementById("roomInput"),
  joinRoomBtn: document.getElementById("joinRoomBtn"),
  createRoomBtn: document.getElementById("createRoomBtn"),
  roomCodeDisplay: document.getElementById("roomCodeDisplay"),
  roomCodeBadge: document.getElementById("roomCodeBadge"),
  statusPill: document.getElementById("statusPill"),
  statusText: document.getElementById("statusText"),
  speechBtn: document.getElementById("speechBtn"),
  camBtn: document.getElementById("camBtn"),
  aiBtn: document.getElementById("aiBtn"),
  leaveBtn: document.getElementById("leaveBtn"),
  chatInput: document.getElementById("chatInput"),
  sendBtn: document.getElementById("sendBtn"),
  messages: document.getElementById("messages"),
  myBox: document.getElementById("myBox"),
  remoteBox: document.getElementById("remoteBox"),
  myVideo: document.getElementById("myVideo"),
  remoteVideo: document.getElementById("remoteVideo"),
  noCamMsg: document.getElementById("noCamMsg"),
  waitingMsg: document.getElementById("waitingMsg"),
  localCameraStatus: document.getElementById("localCameraStatus"),
  remoteCameraStatus: document.getElementById("remoteCameraStatus"),
  myRoleTag: document.getElementById("myRoleTag"),
  remoteRoleTag: document.getElementById("remoteRoleTag"),
  gestureOutput: document.getElementById("gestureOutput"),
  intentOutput: document.getElementById("intentOutput"),
  sentenceOutput: document.getElementById("sentenceOutput"),
};

function selectRole(role) {
  state.selectedRole = role === "receiver" ? "receiver" : "signer";
  elements.roleSignerBtn.classList.toggle("active", state.selectedRole === "signer");
  elements.roleReceiverBtn.classList.toggle("active", state.selectedRole === "receiver");
}

function apiReady() {
  return state.socket && state.socket.readyState === WebSocket.OPEN;
}

function sendAction(action, extra = {}) {
  if (!apiReady()) {
    showToast("Browser bridge is still connecting.", "error");
    return;
  }
  const payload = { type: "ui_action", session_id: state.apiConfig.sessionId, action, ...extra };
  state.socket.send(JSON.stringify(payload));
}

function connectSocket() {
  if (!state.apiConfig) {
    return;
  }
  let socketUrl = state.apiConfig.serverUrl;
  if (!socketUrl) {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    socketUrl = `${protocol}://${window.location.hostname}:${window.location.port}`;
  }
  const socket = new WebSocket(socketUrl);
  state.socket = socket;

  socket.addEventListener("message", (event) => {
    let payload = null;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      return;
    }

    if (payload.type === "snapshot") {
      state.snapshot = payload.state;
      render();
      return;
    }

    if (payload.type === "toast") {
      showToast(payload.message, payload.level);
    }
  });

  socket.addEventListener("close", () => {
    state.socket = null;
    window.setTimeout(connectSocket, 1500);
  });

  socket.addEventListener("open", () => {
    // Register this browser UI with the communication server using the worker's session id
    try {
      socket.send(JSON.stringify({ type: "ui_register", session_id: state.apiConfig.sessionId }));
    } catch (err) {
      // ignore
    }
  });
}

async function loadConfig() {
  const response = await fetch("/app-config.json", { cache: "no-store" });
  state.apiConfig = await response.json();
  elements.myVideo.src = `${state.apiConfig.streamLocalUrl}?stream=local`;
  elements.remoteVideo.src = `${state.apiConfig.streamRemoteUrl}?stream=remote`;
}

function render() {
  const snapshot = state.snapshot;
  if (!snapshot) {
    return;
  }

  selectRole(snapshot.role || state.selectedRole);

  const isRoomStage = snapshot.stage === "room";
  elements.lobby.style.display = isRoomStage ? "none" : "flex";
  elements.room.style.display = isRoomStage ? "flex" : "none";

  elements.statusPill.className = `status-pill ${snapshot.status.connected ? "online" : "offline"}`;
  elements.statusText.textContent = snapshot.status.text || "Not connected";

  elements.roomCodeDisplay.textContent = snapshot.room_code || "----";
  if (!document.activeElement || document.activeElement !== elements.roomInput) {
    elements.roomInput.value = snapshot.room_code || "";
  }

  const isSigner = snapshot.role === "signer";
  updateRoleTag(elements.myRoleTag, snapshot.role);
  updateRoleTag(elements.remoteRoleTag, snapshot.remote_role || (isSigner ? "receiver" : "signer"));
  elements.myBox.className = `camera-box ${isSigner ? "active-signer" : "active-receiver"}${snapshot.local_stream_available ? " has-stream" : ""}`;
  elements.remoteBox.className = `camera-box ${(snapshot.remote_role || "receiver") === "signer" ? "active-signer" : "active-receiver"}${snapshot.remote_stream_available ? " has-stream" : ""}`;

  elements.localCameraStatus.textContent = snapshot.camera_on ? "Waiting for camera..." : "Camera off";
  elements.remoteCameraStatus.textContent = remotePlaceholderText(snapshot);

  elements.speechBtn.classList.toggle("active", snapshot.speech_on);
  elements.speechBtn.textContent = snapshot.speech_on ? "Speech On" : "Speech";
  elements.camBtn.classList.toggle("active", snapshot.camera_on);
  elements.camBtn.textContent = snapshot.camera_on ? "Camera" : "Cam Off";
  elements.aiBtn.classList.toggle("active", snapshot.ai_on);
  elements.aiBtn.textContent = snapshot.ai_on ? "AI On" : "AI Off";

  renderTranslation(snapshot.translation || {});
  renderMessages(snapshot.messages || []);
}

function updateRoleTag(element, role) {
  const normalized = role === "receiver" ? "receiver" : "signer";
  element.textContent = normalized === "signer" ? "Signer" : "Receiver";
  element.className = `camera-role-tag ${normalized === "signer" ? "tag-signer" : "tag-receiver"}`;
}

function remotePlaceholderText(snapshot) {
  if (!snapshot.remote_connected) {
    return "Waiting for other user...";
  }
  if (!snapshot.remote_status.camera_enabled) {
    return "Remote camera is off";
  }
  if (!snapshot.remote_stream_available) {
    return "Receiving remote camera...";
  }
  return "Remote camera";
}

function renderTranslation(translation) {
  const gesture = translation.gesture || "-";
  const intent = translation.intent || "-";
  const sentence = translation.sentence || translation.placeholder || "Gesture recognition active - start signing...";

  elements.gestureOutput.textContent = gesture;
  elements.intentOutput.textContent = intent;
  elements.sentenceOutput.textContent = sentence;

  elements.gestureOutput.classList.toggle("placeholder", !translation.gesture);
  elements.intentOutput.classList.toggle("placeholder", !translation.intent);
  elements.sentenceOutput.classList.toggle("placeholder", !translation.sentence);
}

function renderMessages(messages) {
  if (!messages.length) {
    elements.messages.innerHTML = `
      <div class="empty-messages" id="emptyMsg">
        <span class="empty-icon">&#x1F4AC;</span>
        Messages will appear here instantly
      </div>
    `;
    return;
  }

  const previousHeight = elements.messages.scrollHeight;
  const previousTop = elements.messages.scrollTop;
  elements.messages.innerHTML = messages
    .map((message) => {
      const direction = message.direction || "system";
      const bubble = escapeHtml(message.text || "");
      if (direction === "system") {
        return `<div class="message system"><div class="msg-bubble">${bubble}</div></div>`;
      }
      return `
        <div class="message ${direction}">
          <div class="msg-bubble">${bubble}</div>
          <div class="msg-meta">${escapeHtml(message.time || "")}</div>
        </div>
      `;
    })
    .join("");

  const isNearBottom = previousHeight - previousTop - elements.messages.clientHeight < 80;
  if (isNearBottom) {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function showToast(message, level = "info") {
  const existing = document.querySelector(".toast");
  if (existing) {
    existing.remove();
  }
  const toast = document.createElement("div");
  toast.className = `toast ${level === "error" ? "error" : ""}`.trim();
  toast.textContent = message;
  document.body.appendChild(toast);
  window.setTimeout(() => toast.remove(), 2100);
}

function setupListeners() {
  elements.roleSignerBtn.addEventListener("click", () => selectRole("signer"));
  elements.roleReceiverBtn.addEventListener("click", () => selectRole("receiver"));

  elements.roomInput.addEventListener("input", () => {
    elements.roomInput.value = elements.roomInput.value.toUpperCase();
  });

  elements.joinRoomBtn.addEventListener("click", () => {
    sendAction("join_room", {
      role: state.selectedRole,
      room_code: elements.roomInput.value.trim(),
    });
  });

  elements.createRoomBtn.addEventListener("click", () => {
    sendAction("create_room", { role: state.selectedRole });
  });

  elements.leaveBtn.addEventListener("click", () => {
    sendAction("leave_room");
  });

  elements.speechBtn.addEventListener("click", () => {
    sendAction("toggle_speech");
  });

  elements.camBtn.addEventListener("click", () => {
    sendAction("toggle_camera");
  });

  elements.aiBtn.addEventListener("click", () => {
    sendAction("toggle_ai");
  });

  elements.sendBtn.addEventListener("click", () => {
    submitChat();
  });

  elements.chatInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    submitChat();
  });

  elements.roomCodeBadge.addEventListener("click", async () => {
    const roomCode = elements.roomCodeDisplay.textContent.trim();
    if (!roomCode || roomCode === "----") {
      return;
    }
    try {
      await navigator.clipboard.writeText(roomCode);
      showToast("Room code copied.");
    } catch (error) {
      showToast("Unable to copy room code.", "error");
    }
  });
}

function submitChat() {
  const text = elements.chatInput.value.trim();
  if (!text) {
    return;
  }
  sendAction("send_chat", { text });
  elements.chatInput.value = "";
}

async function main() {
  selectRole("signer");
  setupListeners();
  await loadConfig();
  connectSocket();
}

main().catch(() => {
  showToast("Failed to load the DualTalk UI.", "error");
});
