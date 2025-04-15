// Bun-compatible API server for message monitoring and control
import { serve } from "bun";

// Common CORS headers and helper function
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const addCorsHeaders = (response) => {
  const newHeaders = new Headers(response.headers);
  Object.entries(corsHeaders).forEach(([key, value]) => {
    newHeaders.set(key, value);
  });
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: newHeaders,
  });
};

// Store connected clients
const clients = new Set();

// HTTP Server with API endpoints and WebSocket support
const server = serve({
  port: process.env.PORT || 3000,
  async fetch(req) {
    // Add CORS headers to all responses
    if (req.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          ...corsHeaders,
          "Access-Control-Allow-Headers":
            "Content-Type, Authorization, Upgrade, Sec-WebSocket-Protocol",
        },
      });
    }

    const url = new URL(req.url);

    // WebSocket upgrade with explicit CORS handling
    if (url.pathname === "/ws") {
      console.log(
        "WebSocket connection attempt from:",
        req.headers.get("origin"),
      );

      // Handle WebSocket upgrade
      const upgraded = server.upgrade(req, {
        headers: corsHeaders,
      });

      if (upgraded) {
        console.log("WebSocket upgrade successful");
        return undefined;
      }

      console.log("WebSocket upgrade failed");
      return addCorsHeaders(
        new Response("WebSocket upgrade failed", { status: 400 }),
      );
    }

    // API status endpoint
    if (url.pathname === "/api/status" && req.method === "GET") {
      return addCorsHeaders(
        new Response(
          JSON.stringify({
            status: "running",
            connectedClients: clients.size,
          }),
          {
            headers: { "Content-Type": "application/json" },
          },
        ),
      );
    }

    // Ban agent endpoint
    if (url.pathname === "/api/ban" && req.method === "POST") {
      const authHeader = req.headers.get("authorization");
      const token = authHeader?.split(" ")[1];

      if (!token || token !== "your_secret_token") {
        return new Response("Unauthorized", {
          status: 401,
          headers: corsHeaders,
        });
      }

      return req.json().then(async (data) => {
        try {
          const response = await fetch("http://internal-api:3001/ban", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify(data),
          });

          if (!response.ok) {
            console.error("HTTP Error Response:", {
              status: response.status,
              statusText: response.statusText,
            });
            const errorText = await response.text();
            console.error("Error response body:", errorText);
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const result = await response.json();
          return addCorsHeaders(
            new Response(JSON.stringify(result), {
              status: response.status,
              headers: { "Content-Type": "application/json" },
            }),
          );
        } catch (error) {
          console.error("Error details:", {
            message: error.message,
            cause: error.cause,
            stack: error.stack,
          });
          return addCorsHeaders(
            new Response(
              JSON.stringify({
                error: "Failed to forward ban request",
                details: error.message,
              }),
              {
                status: 500,
                headers: { "Content-Type": "application/json" },
              },
            ),
          );
        }
      });
    }

    // Message endpoint for Prosody
    if (url.pathname === "/api/messages" && req.method === "POST") {
      const authHeader = req.headers.get("authorization");
      const token = authHeader?.split(" ")[1];

      if (!token || token !== "your_secret_token") {
        return addCorsHeaders(new Response("Unauthorized", { status: 401 }));
      }

      return req.json().then((messageData) => {
        console.log("Received message:", messageData);

        // Broadcast to all connected WebSocket clients
        clients.forEach((client) => {
          client.send(JSON.stringify(messageData));
        });

        return addCorsHeaders(
          new Response(JSON.stringify({ status: "ok" }), {
            headers: { "Content-Type": "application/json" },
          }),
        );
      });
    }

    // Not found
    return addCorsHeaders(new Response("Not Found", { status: 404 }));
  },
  websocket: {
    open(ws) {
      console.log("Client connected");
      clients.add(ws);
      // Send an initial message to confirm connection
      ws.send(JSON.stringify({ type: "connection", status: "connected" }));
    },
    message(ws, message) {
      console.log("Received WebSocket message:", message);
      // Echo the message back to confirm receipt
      ws.send(JSON.stringify({ type: "echo", message }));
    },
    close(ws) {
      console.log("Client disconnected");
      clients.delete(ws);
    },
    error(ws, error) {
      console.error("WebSocket error:", error);
      clients.delete(ws);
    },
  },
});

console.log(`Server running at http://localhost:${server.port}`);
