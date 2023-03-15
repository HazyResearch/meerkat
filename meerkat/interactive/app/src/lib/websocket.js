import { writable } from "svelte/store";

export let WEBSOCKET = writable(null);

export function createWebSocket(url) {
    return new Promise((resolve, reject) => {
        const ws = new WebSocket(url);

        ws.addEventListener('open', () => {
            resolve(ws);
        });

        ws.addEventListener('error', (event) => {
            reject(event);
        });
    });
};

// Write a CustomWebSocket class that extends WebSocket
export class CustomWebSocket extends WebSocket {
    constructor(url) {
        super(url);
    }
};

export function makeWebSocketRequest(ws, payload) {
    return new Promise((resolve, reject) => {

        ws.send(JSON.stringify(payload));

        ws.addEventListener('message', (event) => {
            const response = JSON.parse(event.data);
            resolve(response);
        });

        ws.addEventListener('error', (event) => {
            reject(event);
        });
    });
};
