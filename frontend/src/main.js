const state = {
  selectedRole: "signer",
  socket: null,
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

// Hidden elements for local camera capture
const localVideoEl = document.createElement("video");
localVideoEl.autoplay = true;
localVideoEl.playsInline = true;
localVideoEl.muted = true;

const captureCanvas = document.createElement("canvas");
captureCanvas.width = 640;
captureCanvas.height = 480;
const captureCtx = captureCanvas.getContext("2d");

let localStream = null;
let captureInterval = null;
let lastSpokenText = "";
const RENDER_SOCKET_URL = "wss://dualtalk.onrender.com/ws";

function selectRole(role) {
  state.selectedRole = role === "receiver" ? "receiver" : "signer";
  elements.roleSignerBtn.classList.toggle("active", state.selectedRole === "signer");
  elements.roleReceiverBtn.classList.toggle("active", state.selectedRole === "receiver");
}

function apiReady() {
  return state.socket && state.socket.readyState === WebSocket.OPEN;
}

function sendAction(type, extra = {}) {
  if (!apiReady()) {
    showToast("WebSocket is disconnected.", "error");
    return;
  }
  const payload = { type, ...extra };
  state.socket.send(JSON.stringify(payload));
}

async function startCamera() {
  if (localStream) return;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: 15 }
    });
    localVideoEl.srcObject = localStream;
    await localVideoEl.play();
    
    if (captureInterval) clearInterval(captureInterval);
    captureInterval = setInterval(captureFrame, 200);
  } catch (error) {
    showToast("Camera access failed. Check permissions.", "error");
    console.error("Camera start error:", error);
  }
}

function stopCamera() {
  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
  }
  if (localStream) {
    localStream.getTracks().forEach(track => track.stop());
    localStream = null;
  }
  localVideoEl.srcObject = null;
  elements.myVideo.src = "";
}

function captureFrame() {
  if (!state.snapshot || state.snapshot.role !== "signer" || !state.snapshot.camera_on) {
    return;
  }

  if (localVideoEl.readyState === localVideoEl.HAVE_ENOUGH_DATA) {
    captureCtx.drawImage(localVideoEl, 0, 0, captureCanvas.width, captureCanvas.height);
    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.6);
    
    // Draw locally first if AI is off or WebSocket disconnected
    if (!state.snapshot.ai_on || !apiReady()) {
      elements.myVideo.src = dataUrl;
    }

    if (apiReady()) {
      const base64 = dataUrl.split(",")[1];
      state.socket.send(JSON.stringify({
        type: "video_frame",
        image: base64,
        role: state.snapshot.role
      }));
    }
  }
}

function connectSocket() {
  let socketUrl = import.meta.env.VITE_WS_URL || "";
  if (!socketUrl) {
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      socketUrl = "ws://localhost:8765/ws";
    } else {
      socketUrl = RENDER_SOCKET_URL;
    }
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
      syncCameraState();
      return;
    }

    if (payload.type === "toast") {
      showToast(payload.message, payload.level);
      return;
    }

    if (payload.type === "local_frame") {
      if (state.snapshot && state.snapshot.camera_on && state.snapshot.ai_on) {
        elements.myVideo.src = "data:image/jpeg;base64," + payload.image;
      }
      return;
    }

    if (payload.type === "remote_frame" || payload.type === "video_frame") {
      if (state.snapshot && state.snapshot.remote_connected && state.snapshot.remote_status.camera_enabled) {
        elements.remoteVideo.src = "data:image/jpeg;base64," + payload.image;
      }
      return;
    }

    if (payload.type === "speak") {
      if (state.snapshot && state.snapshot.speech_on && payload.text) {
        speakText(payload.text);
      }
      return;
    }
  });

  socket.addEventListener("close", () => {
    state.socket = null;
    state.snapshot = null;
    renderOffline();
    window.setTimeout(connectSocket, 2000);
  });

  socket.addEventListener("open", () => {
    // Re-join or resume state is handled by the server when we send actions or join
  });
}

function speakText(text) {
  if (!window.speechSynthesis || text === lastSpokenText) return;
  lastSpokenText = text;
  
  // Cancel active speak
  window.speechSynthesis.cancel();
  
  const utterance = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.speak(utterance);
}

function syncCameraState() {
  if (!state.snapshot) {
    stopCamera();
    return;
  }
  const shouldCameraBeOn = state.snapshot.camera_on && state.snapshot.role === "signer";
  if (shouldCameraBeOn) {
    startCamera();
  } else {
    stopCamera();
  }
}

function renderOffline() {
  elements.statusPill.className = "status-pill offline";
  elements.statusText.textContent = "Offline (Reconnecting...)";
  elements.myVideo.src = "";
  elements.remoteVideo.src = "";
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

function main() {
  selectRole("signer");
  setupListeners();
  connectSocket();
}

main();
