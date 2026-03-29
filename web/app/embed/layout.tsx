/**
 * Embed layout — intentionally minimal.
 *
 * This layout is for pages loaded inside <iframe> tags on third-party sites.
 * It relies on the root layout for the <html>/<body> shell; it only adds
 * embed-specific inline styles scoped to this subtree.
 *
 * The root layout's SiteChrome component detects /embed/* routes and suppresses
 * GlobalNav and Footer, so only the widget card renders here.
 */
export default function EmbedLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <style>{`
        /* Embed-specific overrides — scoped to iframe context */
        body {
          background: #ffffff;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #222222;
          overflow: auto;
        }
        h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; }
        a { color: #2166ac; text-decoration: none; }
        a:hover { text-decoration: underline; }
      `}</style>
      {children}
    </>
  );
}
