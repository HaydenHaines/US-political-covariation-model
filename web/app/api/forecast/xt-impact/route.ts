// Proxy for GET /api/v1/forecast/xt-impact — cross-type poll impact scores.
// Server-side fetch goes direct to the Python backend, not through the Caddy proxy.
const API_BASE = process.env.API_URL || "http://localhost:8002";

export async function GET(request: Request): Promise<Response> {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") ?? "10";
  const raceType = searchParams.get("race_type");

  const upstream = new URL(`${API_BASE}/api/v1/forecast/xt-impact`);
  upstream.searchParams.set("limit", limit);
  if (raceType) upstream.searchParams.set("race_type", raceType);

  try {
    const res = await fetch(
      upstream.toString(),
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
