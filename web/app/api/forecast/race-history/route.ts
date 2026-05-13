// Proxy for GET /api/v1/forecast/race-history — per-race margin history series.
// Server-side fetch goes direct to the Python backend, not through the Caddy proxy.
const API_BASE = process.env.API_URL || "http://localhost:8002";

export async function GET(): Promise<Response> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race-history`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) {
      return new Response(JSON.stringify({ error: "upstream error" }), {
        status: res.status,
        headers: { "Content-Type": "application/json" },
      });
    }
    const data = await res.json();
    return new Response(JSON.stringify(data), {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
      },
    });
  } catch {
    return new Response(JSON.stringify({ error: "backend unavailable" }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }
}
