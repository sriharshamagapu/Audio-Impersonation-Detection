import React, { useState, useRef, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';

const socket = io('http://localhost:3000');

const riskStyles = {
  Low: { bg: '#e6f4ea', color: '#1e7a46', border: '#1e7a46' },
  Verify: { bg: '#fff8e1', color: '#8a6d00', border: '#c9a400' },
  High: { bg: '#fff0e6', color: '#b5540a', border: '#b5540a' },
  Critical: { bg: '#fdecea', color: '#b3261e', border: '#b3261e' },
};

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [verdict, setVerdict] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [consecutiveHigh, setConsecutiveHigh] = useState(0);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    socket.on('verdict', (data) => {
      console.log('Received verdict:', data);

      if (data.error) {
        setErrorMsg(data.error);
        return;
      }

      setErrorMsg(null);
      setVerdict(data);

      setConsecutiveHigh((prev) => {
        if (data.risk === 'High' || data.risk === 'Critical') {
          return prev + 1;
        }
        return 0;
      });
    });

    socket.on('connect_error', () => {
      setErrorMsg('Cannot reach backend server. Is it running?');
    });

    return () => {
      socket.off('verdict');
      socket.off('connect_error');
    };
  }, []);

  const recordChunk = () => {
    if (!streamRef.current) return;
    const recorder = new MediaRecorder(streamRef.current);
    let chunks = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      socket.emit('audio_chunk', blob);
    };

    recorder.start();
    setTimeout(() => recorder.stop(), 1500);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setIsRecording(true);
      setErrorMsg(null);
      recordChunk();
      intervalRef.current = setInterval(recordChunk, 1500);
    } catch (err) {
      setErrorMsg('Microphone access denied or unavailable.');
    }
  };

  const stopRecording = () => {
    clearInterval(intervalRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
    setIsRecording(false);
    setConsecutiveHigh(0);
  };

  const showAlert = consecutiveHigh >= 2;
  const style = verdict ? riskStyles[verdict.risk] || riskStyles.Verify : null;

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', textAlign: 'center', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Voice Clone Detector</h1>

      <button
        onClick={isRecording ? stopRecording : startRecording}
        style={{
          padding: '10px 24px',
          fontSize: '16px',
          borderRadius: '8px',
          cursor: 'pointer',
          backgroundColor: isRecording ? '#b3261e' : '#2f6fed',
          color: 'white',
          border: 'none',
        }}
      >
        {isRecording ? 'Stop' : 'Start'} Recording
      </button>

      {errorMsg && (
        <div style={{
          marginTop: '20px',
          padding: '12px',
          borderRadius: '8px',
          backgroundColor: '#fdecea',
          color: '#b3261e',
          border: '1px solid #b3261e',
        }}>
          ⚠ {errorMsg}
        </div>
      )}

      {verdict && !errorMsg && (
        <div
          style={{
            marginTop: '30px',
            padding: '24px',
            borderRadius: '12px',
            backgroundColor: style.bg,
            border: `2px solid ${style.border}`,
            color: style.color,
          }}
        >
          <h2 style={{ margin: 0 }}>Risk Level: {verdict.risk}</h2>
          <p style={{ margin: '8px 0 0 0' }}>
            Spoof Probability: {(verdict.p_aasist_spoof * 100).toFixed(1)}%
          </p>

          {showAlert && (
            <div
              style={{
                marginTop: '16px',
                padding: '14px',
                borderRadius: '8px',
                backgroundColor: '#b3261e',
                color: 'white',
                fontWeight: 'bold',
              }}
            >
              🚨 Sustained high risk detected — recommend secondary verification
              (call-back, MFA, or supervisor escalation) before proceeding.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
