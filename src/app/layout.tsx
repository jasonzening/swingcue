import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SwingCue — AI Golf Swing Coach",
  description:
    "Upload your golf swing. See what's wrong. Get one clear fix — no jargon, no guesswork. SwingCue is the AI coach in your pocket.",
  keywords: ["golf", "swing analysis", "AI golf coach", "golf improvement", "swing fix"],
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
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#080c08",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" style={{ background: "#080c08" }}>
      <body style={{ margin: 0, background: "#080c08" }}>{children}</body>
    </html>
  );
}
