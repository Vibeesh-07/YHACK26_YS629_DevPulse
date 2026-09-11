// Firebase Web SDK Configuration for CryoNav14
const firebaseConfig = {
  apiKey: "AIzaSyAqD3uxKiE1qKJBYBtIVgRnxHzUqae73kg",
  authDomain: "cryonav14.firebaseapp.com",
  projectId: "cryonav14",
  storageBucket: "cryonav14.firebasestorage.app",
  messagingSenderId: "254609799079",
  appId: "1:254609799079:web:88e8ef62a05176367a2882",
  measurementId: "G-1PDSBX57NL"
};

// Expose globally for browser client scripts
if (typeof window !== "undefined") {
  window.firebaseConfig = firebaseConfig;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = firebaseConfig;
}
