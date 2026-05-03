// Proxy for GET /api/v1/forecast/xt-impact — cross-type poll impact scores.
// Server-side fetch goes direct to the Python backend, not through the Caddy proxy.
const API_BASE = process.env.API_URL || "http://localhost:8002";

export async function GET(request: Request): Promise<Response> {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") ?? "10";

  try {
    const res = await fetch(
      `${API_BASE}/api/v1/forecast/xt-impact?limit=${limit}`,
      { next: { revalidate: 3600 } },
    );
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
