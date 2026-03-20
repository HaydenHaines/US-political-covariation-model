"use client";
export function CommunityPanel({
  communityId,
  onClose,
}: {
  communityId: number;
  onClose: () => void;
}) {
  return (
    <div style={{
      position: "absolute",
      top: 0,
      right: 0,
      width: "320px",
      height: "100%",
      background: "white",
      borderLeft: "1px solid var(--color-border)",
      padding: "20px",
      overflow: "auto",
      boxShadow: "-2px 0 8px rgba(0,0,0,0.08)",
    }}>
      <button onClick={onClose} style={{ float: "right", border: "none", background: "none", cursor: "pointer", fontSize: "18px" }}>×</button>
      <h3 style={{ margin: "0 0 12px" }}>Community {communityId}</h3>
      <p style={{ color: "var(--color-text-muted)", fontSize: "13px" }}>Loading…</p>
    </div>
  );
}
