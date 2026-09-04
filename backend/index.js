const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
  }
});

app.get('/', (req, res) => {
  res.send('Voice Clone Detector Backend is running');
});

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('audio_chunk', async (data) => {
    console.log('Received audio chunk from', socket.id, '- size:', data.length || data.byteLength);

    try {
      const formData = new FormData();
      formData.append('file', Buffer.from(data), {
        filename: 'chunk.webm',
        contentType: 'audio/webm',
      });

      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: formData.getHeaders(),
      });

      socket.emit('verdict', response.data);
    } catch (err) {
      console.error('Error calling AASIST API:', err.message);
      socket.emit('verdict', { error: 'Analysis failed' });
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

const PORT = 3000;
server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});