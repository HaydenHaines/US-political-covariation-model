/**
 * Embed layout — intentionally minimal.
 *
 * Overrides the root layout so the global `overflow: hidden` body rule (needed
 * for the full-screen map shell) does not clip the embed card.  The embed page
 * is designed to be loaded inside an <iframe> by third-party sites, so we keep
 * CSS completely self-contained here rather than pulling in globals.css.
 */
export default function EmbedLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta name="robots" content="noindex" />
        <style>{`
          *, *::before, *::after { box-sizing: border-box; }
          html, body {
            margin: 0;
            padding: 0;
            background: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #222222;
            overflow: auto;
          }
          h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; }
          a { color: #2166ac; text-decoration: none; }
          a:hover { text-decoration: underline; }
        `}</style>
      </head>
      <body>{children}</body>
    </html>
  );
}
