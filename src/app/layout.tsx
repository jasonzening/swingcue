import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SwingCue — AI Golf Swing Coach",
  description:
    "Upload your golf swing video. See the #1 thing holding you back. Get one clear fix — no jargon, in under a minute. SwingCue is the AI coach in your pocket.",
  keywords: ["golf", "swing analysis", "AI golf coach", "golf improvement", "swing fix", "golf lesson"],
  authors: [{ name: "SwingCue" }],
  openGraph: {
    title: "SwingCue — AI Golf Swing Coach",
    description: "Upload your swing. See what's wrong. Know what to fix next.",
    url: "https://swingcue.ai",
    siteName: "SwingCue",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "SwingCue — AI Golf Swing Coach",
    description: "Upload your swing. See what's wrong. Know what to fix next.",
  },
  icons: {
    icon: [
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon.ico" },
    ],
    apple: "/favicon.svg",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#080c08",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        {/* Only load weights actually used: 400 (body), 600 (nav), 700 (labels), 800 (headings) */}
        <link
          href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
