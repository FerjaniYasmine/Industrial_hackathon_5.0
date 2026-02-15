// ============ SIMULATEUR MQTT + WEBSOCKET ============
// npm install mqtt ws express
// node mqtt-websocket-simulator.js

const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const mqtt = require('mqtt');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Configuration MQTT + WebSocket
const MQTT_BROKER = 'mqtt://broker.hivemq.com';
const WS_PORT = 8080;
const HTTP_PORT = 3000;

// État des capteurs
const pipes = ['A-12', 'A-13', 'A-14', 'A-15', 'B-01', 'B-02', 'B-03', 'C-21', 'C-22', 'D-30', 'D-31', 'E-40', 'E-41', 'F-50', 'F-51', 'G-60', 'G-61', 'H-70', 'H-71'];

const pipeStates = {};
pipes.forEach(id => {
    pipeStates[id] = {
        risk: 20 + Math.random() * 40,
        pressure: 0.3 + Math.random() * 0.5,
        vibration: 0.5 + Math.random() * 2,
        temp: 58 + Math.random() * 15,
        flowRate: 3 + Math.random() * 12
    };
});

// ========== CONNEXION MQTT ==========
const mqttClient = mqtt.connect(MQTT_BROKER, {
    clientId: 'simulator-' + Date.now(),
    clean: true,
    reconnectPeriod: 3000
});

mqttClient.on('connect', () => {
    console.log('✅ MQTT Connecté au broker HiveMQ\n');
});

mqttClient.on('error', (err) => {
    console.log('⚠️  MQTT Error:', err.message);
});

// ========== BROADCAST WEBSOCKET ==========
function broadcastToClients(data) {
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify(data));
        }
    });
}

// ========== GÉNÉRATION + PUBLICATION DONNÉES ==========
let messageCount = 0;

setInterval(() => {
    pipes.forEach(pipeId => {
        const state = pipeStates[pipeId];

        // Variations réalistes
        state.risk = Math.max(0, Math.min(100, state.risk + (Math.random() - 0.5) * 4));
        state.pressure = Math.max(0.05, Math.min(1.0, state.pressure + (Math.random() - 0.5) * 0.08));
        state.vibration = Math.max(0.1, state.vibration + (Math.random() - 0.5) * 0.5);
        state.temp = Math.max(50, Math.min(80, state.temp + (Math.random() - 0.5) * 2));
        state.flowRate = Math.max(0.5, Math.min(20, state.flowRate + (Math.random() - 0.5) * 1.5));

        // Tendance de dégradation
        if (Math.random() > 0.7) {
            state.risk += 0.3;
            state.vibration += 0.05;
        }

        // Niveau acoustique
        let acoustic = 'Basse';
        if (state.risk > 70) acoustic = 'Très haute';
        else if (state.risk > 50) acoustic = 'Haute';
        else if (state.risk > 30) acoustic = 'Moyenne';

        // Créer le payload
        const payload = {
            pipeId,
            risk: parseFloat(state.risk.toFixed(2)),
            pressure: parseFloat(state.pressure.toFixed(3)),
            vibration: parseFloat(state.vibration.toFixed(2)),
            temperature: parseFloat(state.temp.toFixed(1)),
            flowRate: parseFloat(state.flowRate.toFixed(2)),
            acoustic,
            timestamp: new Date().toISOString(),
            messageId: ++messageCount
        };

        // Publier sur MQTT
        const topic = `factory/pipes/${pipeId}/sensors`;
        mqttClient.publish(topic, JSON.stringify(payload), { qos: 1 });

        // Envoyer aussi via WebSocket
        broadcastToClients(payload);

        // Affichage terminal
        const status = payload.risk > 70 ? '🚨' : payload.risk > 30 ? '⚠️' : '✅';
        const riskBar = '█'.repeat(Math.ceil(payload.risk / 5)) + '░'.repeat(Math.ceil((100 - payload.risk) / 5));
        console.log(`${status} ${pipeId.padEnd(5)} | [${riskBar}] ${payload.risk.toFixed(1).padStart(5)}% | P:${payload.pressure.toFixed(2)} | T:${payload.temp.toFixed(1)}°C | V:${payload.vibration.toFixed(2)}`);
    });

    console.log('═'.repeat(100));
}, 1000);

// ========== WEBSOCKET SERVER ==========
wss.on('connection', (ws) => {
    console.log(`\n👤 Client WebSocket connecté`);
    ws.send(JSON.stringify({ type: 'connected', message: 'Connected to simulator' }));
});

// ========== HTTP SERVER ==========
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.send('✅ Simulator is running on ws://localhost:8080');
});

server.listen(WS_PORT, () => {
    console.log(`\n╔════════════════════════════════════════════╗`);
    console.log(`║  🚀 MQTT SIMULATOR + WEBSOCKET STARTED     ║`);
    console.log(`╚════════════════════════════════════════════╝\n`);
    console.log(`📡 MQTT Broker: ${MQTT_BROKER}`);
    console.log(`🌐 WebSocket Server: ws://localhost:${WS_PORT}`);
    console.log(`🏭 Pipes simulés: ${pipes.length}`);
    console.log(`⏱️  Update Interval: 1000ms\n`);
    console.log(`📊 Publishing to:`);
    console.log(`   - MQTT: factory/pipes/{ID}/sensors`);
    console.log(`   - WebSocket: broadcast to all clients\n`);
    console.log(`─`.repeat(100) + '\n');
});